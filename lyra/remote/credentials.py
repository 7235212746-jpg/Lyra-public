from __future__ import annotations

import base64
import binascii
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


_SCHEMA_VERSION = 1
_MAX_DOCUMENT_BYTES = 256 * 1024
_MAX_SECRET_CHARACTERS = 16 * 1024
_MAX_PLAINTEXT_BYTES = 128 * 1024
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_DOCUMENT_FIELDS = {"version", "profile_id", "blob"}
_PAYLOAD_FIELDS = {"password", "private_key_passphrase"}
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class CredentialStoreError(RuntimeError):
    """Base class for credential storage failures safe to show to users."""


class CredentialPlatformError(CredentialStoreError):
    """Raised when Windows DPAPI is unavailable."""


class CredentialNotFoundError(CredentialStoreError):
    """Raised when no stored credential exists for a profile."""


class CredentialCorruptError(CredentialStoreError):
    """Raised for invalid, tampered, or undecryptable credential files."""


@dataclass(frozen=True)
class RemoteCredentials:
    password: str | None = None
    private_key_passphrase: str | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("password", self.password),
            ("private_key_passphrase", self.private_key_passphrase),
        ):
            if value is not None and (
                not isinstance(value, str) or len(value) > _MAX_SECRET_CHARACTERS
            ):
                raise ValueError(
                    f"{field} must be a string of at most {_MAX_SECRET_CHARACTERS} characters or None."
                )
        if self.password is None and self.private_key_passphrase is None:
            raise ValueError("At least one remote credential secret is required.")

    def __repr__(self) -> str:
        return "RemoteCredentials(password=<redacted>, private_key_passphrase=<redacted>)"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _require_windows() -> None:
    if os.name != "nt":
        raise CredentialPlatformError(
            "Remote credential persistence requires Windows current-user DPAPI."
        )


def _validate_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not _ID_PATTERN.fullmatch(profile_id):
        raise ValueError(
            "profile_id must use 1-64 letters, numbers, dots, underscores, or hyphens."
        )
    return profile_id


def _blob(value: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(value)
    return (
        _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))),
        buffer,
    )


def _entropy(profile_id: str) -> bytes:
    return b"Lyra remote credential v1\0" + profile_id.encode("ascii")


def _crypt32() -> Any:
    try:
        library = ctypes.WinDLL("crypt32", use_last_error=True)
    except (AttributeError, OSError) as exc:
        raise CredentialPlatformError("Windows DPAPI is unavailable.") from exc
    library.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    library.CryptProtectData.restype = wintypes.BOOL
    library.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    library.CryptUnprotectData.restype = wintypes.BOOL
    return library


def _copy_and_free(output: _DataBlob) -> bytes:
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        if output.pbData:
            ctypes.windll.kernel32.LocalFree(output.pbData)


def _protect(plaintext: bytes, profile_id: str) -> bytes:
    input_blob, input_buffer = _blob(plaintext)
    entropy_blob, entropy_buffer = _blob(_entropy(profile_id))
    output = _DataBlob()
    if not _crypt32().CryptProtectData(
        ctypes.byref(input_blob),
        "Lyra remote credential",
        ctypes.byref(entropy_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    ):
        raise CredentialStoreError(
            f"Windows DPAPI encryption failed with error {ctypes.get_last_error()}."
        )
    # Keep backing buffers alive until CryptProtectData has returned.
    del input_buffer, entropy_buffer
    return _copy_and_free(output)


def _unprotect(ciphertext: bytes, profile_id: str) -> bytes:
    input_blob, input_buffer = _blob(ciphertext)
    entropy_blob, entropy_buffer = _blob(_entropy(profile_id))
    output = _DataBlob()
    if not _crypt32().CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    ):
        raise CredentialCorruptError(
            "Remote credential could not be decrypted by the current Windows user."
        )
    del input_buffer, entropy_buffer
    return _copy_and_free(output)


def _serialize(credentials: RemoteCredentials) -> bytes:
    if not isinstance(credentials, RemoteCredentials):
        raise TypeError("credentials must be a RemoteCredentials value.")
    payload = {
        "password": credentials.password,
        "private_key_passphrase": credentials.private_key_passphrase,
    }
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(data) > _MAX_PLAINTEXT_BYTES:
        raise ValueError("Remote credential payload exceeds its size limit.")
    return data


def _deserialize(data: bytes) -> RemoteCredentials:
    if len(data) > _MAX_PLAINTEXT_BYTES:
        raise CredentialCorruptError("Decrypted remote credential exceeds its size limit.")
    try:
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
            raise ValueError
        return RemoteCredentials(**payload)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CredentialCorruptError("Decrypted remote credential payload is invalid.") from exc


class CredentialStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, profile_id: str, credentials: RemoteCredentials) -> None:
        _require_windows()
        profile_id = _validate_profile_id(profile_id)
        ciphertext = _protect(_serialize(credentials), profile_id)
        document = {
            "version": _SCHEMA_VERSION,
            "profile_id": profile_id,
            "blob": base64.b64encode(ciphertext).decode("ascii"),
        }
        data = json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        if len(data) > _MAX_DOCUMENT_BYTES:
            raise ValueError("Remote credential document exceeds its size limit.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, profile_id: str) -> RemoteCredentials:
        _require_windows()
        profile_id = _validate_profile_id(profile_id)
        try:
            size = self.path.stat().st_size
        except FileNotFoundError as exc:
            raise CredentialNotFoundError(
                f"Remote credential for profile {profile_id!r} was not found."
            ) from exc
        except OSError as exc:
            raise CredentialStoreError("Remote credential file could not be inspected.") from exc
        if size > _MAX_DOCUMENT_BYTES:
            raise CredentialCorruptError("Remote credential file exceeds its size limit.")

        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
                raise ValueError
            if document["version"] != _SCHEMA_VERSION:
                raise ValueError
            if document["profile_id"] != profile_id:
                raise ValueError
            encoded = document["blob"]
            if not isinstance(encoded, str):
                raise ValueError
            ciphertext = base64.b64decode(encoded, validate=True)
            if not ciphertext or len(ciphertext) > _MAX_DOCUMENT_BYTES:
                raise ValueError
        except FileNotFoundError as exc:
            raise CredentialNotFoundError(
                f"Remote credential for profile {profile_id!r} was not found."
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise CredentialStoreError("Remote credential file could not be read.") from exc
        except (json.JSONDecodeError, binascii.Error, ValueError) as exc:
            raise CredentialCorruptError("Remote credential document is invalid.") from exc
        return _deserialize(_unprotect(ciphertext, profile_id))

    def delete(self, profile_id: str) -> None:
        _require_windows()
        profile_id = _validate_profile_id(profile_id)
        self.load(profile_id)
        try:
            self.path.unlink()
        except FileNotFoundError as exc:
            raise CredentialNotFoundError(
                f"Remote credential for profile {profile_id!r} was not found."
            ) from exc
        except OSError as exc:
            raise CredentialStoreError("Remote credential file could not be deleted.") from exc


def save(path: str | Path, profile_id: str, credentials: RemoteCredentials) -> None:
    CredentialStore(path).save(profile_id, credentials)


def load(path: str | Path, profile_id: str) -> RemoteCredentials:
    return CredentialStore(path).load(profile_id)


def delete(path: str | Path, profile_id: str) -> None:
    CredentialStore(path).delete(profile_id)


__all__ = [
    "CredentialCorruptError",
    "CredentialNotFoundError",
    "CredentialPlatformError",
    "CredentialStore",
    "CredentialStoreError",
    "RemoteCredentials",
    "delete",
    "load",
    "save",
]

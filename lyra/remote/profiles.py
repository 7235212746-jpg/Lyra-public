from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4


_PROFILE_SCHEMA_VERSION = 1
_MAX_PROFILE_FILE_BYTES = 1024 * 1024
_PROFILE_FILE_NAME = "remote_profiles.json"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HOST_PATTERN = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9._:-]{0,254}|\[[0-9A-Fa-f:.]+\])$")
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_SSH_KEY_ALGORITHM = r"(?:ssh-[A-Za-z0-9@._+-]+|ecdsa-[A-Za-z0-9@._+-]+)"
_SHA256_BASE64 = r"[A-Za-z0-9+/]{43}=?"
_LEGACY_SHA256_BASE64 = r"[A-Za-z0-9+/]{32,}={0,2}"
_FINGERPRINT_PATTERN = re.compile(
    rf"^(?:{_SSH_KEY_ALGORITHM}\s+\d+\s+(?:{_SHA256_BASE64}|SHA256:{_LEGACY_SHA256_BASE64})"
    rf"|SHA256:{_LEGACY_SHA256_BASE64})$"
)
_PROFILE_FIELDS = {
    "profile_id",
    "name",
    "host",
    "port",
    "username",
    "host_fingerprint",
    "private_key_path",
}


def _bounded_text(value: Any, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty string of at most {maximum} characters.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} contains control characters.")
    return value


def normalize_fingerprint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Host fingerprint must be a SHA-256 fingerprint string.")
    normalized = " ".join(value.strip().split())
    if not _FINGERPRINT_PATTERN.fullmatch(normalized):
        raise ValueError("Host fingerprint must be a pinned SHA-256 SSH fingerprint.")
    return normalized


@dataclass(frozen=True)
class RemoteProfile:
    profile_id: str
    name: str
    host: str
    port: int
    username: str
    host_fingerprint: str | None = None
    private_key_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _ID_PATTERN.fullmatch(self.profile_id):
            raise ValueError("profile_id must use only letters, numbers, dots, underscores, or hyphens.")
        object.__setattr__(self, "name", _bounded_text(self.name, "name", 128))
        host = _bounded_text(self.host, "host", 255)
        username = _bounded_text(self.username, "username", 128)
        if not _HOST_PATTERN.fullmatch(host):
            raise ValueError("host must be a DNS name or an IPv4/IPv6 address, not a URI or option.")
        if not _USERNAME_PATTERN.fullmatch(username):
            raise ValueError("username contains characters that are unsafe for SSH transports.")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "username", username)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer from 1 through 65535.")
        if self.host_fingerprint is not None:
            object.__setattr__(
                self,
                "host_fingerprint",
                normalize_fingerprint(self.host_fingerprint),
            )
        if self.private_key_path is not None:
            object.__setattr__(
                self,
                "private_key_path",
                _bounded_text(self.private_key_path, "private_key_path", 4096),
            )

    def with_confirmed_fingerprint(self, fingerprint: str) -> RemoteProfile:
        return RemoteProfile(
            self.profile_id,
            self.name,
            self.host,
            self.port,
            self.username,
            normalize_fingerprint(fingerprint),
            self.private_key_path,
        )

    def to_persistent_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "host_fingerprint": self.host_fingerprint,
            "private_key_path": self.private_key_path,
        }

    @classmethod
    def from_persistent_dict(cls, value: Any) -> RemoteProfile:
        if not isinstance(value, dict) or set(value) != _PROFILE_FIELDS:
            raise ValueError("Remote profile contains unknown, missing, or secret fields.")
        return cls(**value)


@dataclass(frozen=True)
class RuntimeAuthentication:
    """Ephemeral credentials. Instances are never accepted by ProfileStore."""

    password: str | None = None
    private_key_passphrase: str | None = None

    def __repr__(self) -> str:
        return "RuntimeAuthentication(password=<redacted>, private_key_passphrase=<redacted>)"


class ProfileStore:
    def __init__(self, path: str | Path | None = None) -> None:
        uses_default_path = path is None
        self.path = Path(path) if path is not None else self.default_path()
        if uses_default_path:
            self._migrate_legacy_store()

    @staticmethod
    def default_path() -> Path:
        """Return a version-independent path in the Windows user data area."""
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        user_data = Path(root) if root else Path.home() / "AppData" / "Local"
        return user_data / "Lyra" / _PROFILE_FILE_NAME

    @staticmethod
    def _legacy_paths() -> tuple[Path, ...]:
        candidates: list[Path] = []
        roaming = os.environ.get("APPDATA")
        if roaming:
            candidates.append(Path(roaming) / "Lyra" / _PROFILE_FILE_NAME)
        legacy_home = Path.home() / ".lyra"
        candidates.extend(
            (
                legacy_home / "Lyra" / _PROFILE_FILE_NAME,
                legacy_home / _PROFILE_FILE_NAME,
            )
        )
        return tuple(candidates)

    def _migrate_legacy_store(self) -> None:
        if self.path.exists():
            return
        destination = self.path.resolve(strict=False)
        for legacy_path in self._legacy_paths():
            if legacy_path.resolve(strict=False) == destination or not legacy_path.is_file():
                continue
            data = self._read_bounded_bytes(legacy_path)
            self._decode_profiles(data)
            backup = self._available_migration_backup(legacy_path)
            self._atomic_write(backup, data)
            self._atomic_write(self.path, data)
            legacy_path.unlink()
            return

    @staticmethod
    def _available_migration_backup(source: Path) -> Path:
        backup = source.with_name(f"{source.name}.pre-migration.bak")
        sequence = 1
        while backup.exists():
            backup = source.with_name(f"{source.name}.pre-migration.{sequence}.bak")
            sequence += 1
        return backup

    @staticmethod
    def _read_bounded_bytes(path: Path) -> bytes:
        try:
            if path.stat().st_size > _MAX_PROFILE_FILE_BYTES:
                raise ValueError("Remote profile file exceeds its size limit.")
            return path.read_bytes()
        except OSError as exc:
            raise ValueError("Remote profile file could not be read.") from exc

    @staticmethod
    def _decode_profiles(data: bytes) -> tuple[RemoteProfile, ...]:
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Remote profile file is not valid UTF-8 JSON.") from exc
        if not isinstance(document, dict) or set(document) != {"schema_version", "profiles"}:
            raise ValueError("Remote profile document contains unknown or missing fields.")
        if document["schema_version"] != _PROFILE_SCHEMA_VERSION:
            raise ValueError("Unsupported remote profile schema version.")
        raw_profiles = document["profiles"]
        if not isinstance(raw_profiles, list) or len(raw_profiles) > 256:
            raise ValueError("Remote profiles must be a bounded list.")
        profiles = tuple(RemoteProfile.from_persistent_dict(item) for item in raw_profiles)
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise ValueError("Remote profile IDs must be unique.")
        return profiles

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> tuple[RemoteProfile, ...]:
        if not self.path.exists():
            return ()
        return self._decode_profiles(self._read_bounded_bytes(self.path))

    def save(self, profiles: Iterable[RemoteProfile]) -> None:
        values = tuple(profiles)
        if len(values) > 256 or any(not isinstance(item, RemoteProfile) for item in values):
            raise ValueError("ProfileStore only accepts up to 256 RemoteProfile values.")
        if len({profile.profile_id for profile in values}) != len(values):
            raise ValueError("Remote profile IDs must be unique.")
        document = {
            "schema_version": _PROFILE_SCHEMA_VERSION,
            "profiles": [profile.to_persistent_dict() for profile in values],
        }
        data = json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        if len(data) > _MAX_PROFILE_FILE_BYTES:
            raise ValueError("Remote profile file exceeds its size limit.")
        self._atomic_write(self.path, data)

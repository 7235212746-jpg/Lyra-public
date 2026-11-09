from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lyra.remote.credentials import (
    CredentialCorruptError,
    CredentialNotFoundError,
    CredentialPlatformError,
    CredentialStore,
    RemoteCredentials,
)


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is required")


def credentials() -> RemoteCredentials:
    return RemoteCredentials(
        password="unique-password-4f60",
        private_key_passphrase="unique-passphrase-8971",
    )


def test_dpapi_round_trip_persists_only_version_id_and_base64_blob(tmp_path: Path) -> None:
    target = tmp_path / "cluster-a.credential.json"
    store = CredentialStore(target)

    store.save("cluster-a", credentials())

    raw = target.read_text(encoding="utf-8")
    document = json.loads(raw)
    assert set(document) == {"version", "profile_id", "blob"}
    assert document["version"] == 1
    assert document["profile_id"] == "cluster-a"
    assert base64.b64decode(document["blob"], validate=True)
    assert "unique-password-4f60" not in raw
    assert "unique-passphrase-8971" not in raw
    assert "password" not in raw.casefold()
    assert "passphrase" not in raw.casefold()
    assert store.load("cluster-a") == credentials()


def test_dpapi_ciphertext_is_not_plaintext_or_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    CredentialStore(first).save("cluster-a", credentials())
    CredentialStore(second).save("cluster-a", credentials())

    first_blob = json.loads(first.read_text(encoding="utf-8"))["blob"]
    second_blob = json.loads(second.read_text(encoding="utf-8"))["blob"]
    assert first_blob != second_blob


def test_repr_never_discloses_secrets() -> None:
    value = credentials()

    rendered = repr(value)

    assert "unique-password-4f60" not in rendered
    assert "unique-passphrase-8971" not in rendered
    assert "redacted" in rendered


@pytest.mark.parametrize(
    "profile_id",
    ["", "../escape", "cluster/a", "cluster\\a", "-" * 65, "space here", "\x00bad"],
)
def test_profile_id_is_validated_before_file_access(tmp_path: Path, profile_id: str) -> None:
    store = CredentialStore(tmp_path / "credential.json")

    with pytest.raises(ValueError, match="profile_id"):
        store.save(profile_id, credentials())
    with pytest.raises(ValueError, match="profile_id"):
        store.load(profile_id)
    with pytest.raises(ValueError, match="profile_id"):
        store.delete(profile_id)


def test_load_rejects_missing_file_with_explicit_safe_error(tmp_path: Path) -> None:
    with pytest.raises(CredentialNotFoundError, match="not found"):
        CredentialStore(tmp_path / "missing.json").load("cluster-a")


@pytest.mark.parametrize(
    "document",
    [
        b"not-json",
        b'{"version":1,"profile_id":"cluster-a","blob":"%%%"}',
        b'{"version":2,"profile_id":"cluster-a","blob":"AA=="}',
        b'{"version":1,"profile_id":"other","blob":"AA=="}',
        b'{"version":1,"profile_id":"cluster-a","blob":"AA==","extra":1}',
    ],
)
def test_load_rejects_malformed_tampered_or_mismatched_documents(
    tmp_path: Path, document: bytes
) -> None:
    target = tmp_path / "credential.json"
    target.write_bytes(document)

    with pytest.raises(CredentialCorruptError, match="credential"):
        CredentialStore(target).load("cluster-a")


def test_load_rejects_oversized_document_before_parsing(tmp_path: Path) -> None:
    target = tmp_path / "credential.json"
    target.write_bytes(b"x" * (256 * 1024 + 1))

    with pytest.raises(CredentialCorruptError, match="size"):
        CredentialStore(target).load("cluster-a")


def test_atomic_save_preserves_existing_credentials_when_replace_fails(tmp_path: Path) -> None:
    target = tmp_path / "credential.json"
    store = CredentialStore(target)
    original = credentials()
    store.save("cluster-a", original)

    with patch("lyra.remote.credentials.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            store.save("cluster-a", RemoteCredentials(password="replacement"))

    assert store.load("cluster-a") == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_delete_removes_credentials_and_missing_delete_fails_explicitly(tmp_path: Path) -> None:
    target = tmp_path / "credential.json"
    store = CredentialStore(target)
    store.save("cluster-a", credentials())

    store.delete("cluster-a")

    assert not target.exists()
    with pytest.raises(CredentialNotFoundError, match="not found"):
        store.delete("cluster-a")


def test_platform_guard_fails_closed_before_save_load_or_delete(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credential.json")

    with patch("lyra.remote.credentials.os.name", "posix"):
        for operation in (
            lambda: store.save("cluster-a", credentials()),
            lambda: store.load("cluster-a"),
            lambda: store.delete("cluster-a"),
        ):
            with pytest.raises(CredentialPlatformError, match="Windows"):
                operation()

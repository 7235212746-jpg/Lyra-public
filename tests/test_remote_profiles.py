from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lyra.remote.profiles import (
    ProfileStore,
    RemoteProfile,
    RuntimeAuthentication,
    normalize_fingerprint,
)


WINSCP_FINGERPRINT = "ssh-ed25519 255 " + "A" * 43
FINGERPRINT = WINSCP_FINGERPRINT.replace(" 255 ", " 255 SHA256:")


def _profile(profile_id: str = "cluster-a") -> RemoteProfile:
    return RemoteProfile(
        profile_id=profile_id,
        name="Main cluster",
        host="login.example.edu",
        port=22,
        username="researcher",
        host_fingerprint=FINGERPRINT,
        private_key_path="C:/Users/me/.ssh/id_ed25519",
    )


def _write_v1(path: Path, profile: RemoteProfile) -> bytes:
    document = {
        "schema_version": 1,
        "profiles": [profile.to_persistent_dict()],
    }
    data = json.dumps(document, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def test_profile_store_round_trip_never_persists_runtime_secrets(tmp_path):
    profile = _profile()
    auth = RuntimeAuthentication(password="unique-password", private_key_passphrase="unique-passphrase")
    store = ProfileStore(tmp_path / "profiles.json")

    store.save((profile,))

    raw = (tmp_path / "profiles.json").read_text(encoding="utf-8")
    assert "unique-password" not in raw
    assert "unique-passphrase" not in raw
    assert "password" not in raw.casefold()
    assert "passphrase" not in raw.casefold()
    assert store.load() == (profile,)
    assert "unique-password" not in repr(auth)
    assert "unique-passphrase" not in repr(auth)


def test_default_store_uses_stable_local_app_data_path_across_project_updates(
    tmp_path,
    monkeypatch,
):
    local_app_data = tmp_path / "LocalAppData"
    old_checkout = tmp_path / "Lyra-0.2"
    new_checkout = tmp_path / "Lyra-0.3"
    old_checkout.mkdir()
    new_checkout.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    monkeypatch.chdir(old_checkout)
    old_store = ProfileStore()
    old_store.save((_profile(),))

    monkeypatch.chdir(new_checkout)
    updated_store = ProfileStore()

    assert old_store.path == local_app_data / "Lyra" / "remote_profiles.json"
    assert updated_store.path == old_store.path
    assert updated_store.load() == (_profile(),)


@pytest.mark.parametrize(
    "legacy_relative_path",
    [
        Path("Roaming") / "Lyra" / "remote_profiles.json",
        Path("UserHome") / ".lyra" / "Lyra" / "remote_profiles.json",
        Path("UserHome") / ".lyra" / "remote_profiles.json",
    ],
)
def test_default_store_migrates_legacy_v1_after_backing_it_up(
    tmp_path,
    monkeypatch,
    legacy_relative_path,
):
    local_app_data = tmp_path / "LocalAppData"
    app_data = tmp_path / "Roaming"
    user_home = tmp_path / "UserHome"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(app_data))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    legacy_path = tmp_path / legacy_relative_path
    original = _write_v1(legacy_path, _profile())

    store = ProfileStore()

    assert store.load() == (_profile(),)
    assert store.path == local_app_data / "Lyra" / "remote_profiles.json"
    assert store.path.read_bytes() == original
    assert not legacy_path.exists()
    backups = tuple(legacy_path.parent.glob("remote_profiles.json.pre-migration*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_migrated_schema_v1_profile_is_lossless(tmp_path, monkeypatch):
    local_app_data = tmp_path / "LocalAppData"
    legacy_path = tmp_path / "Roaming" / "Lyra" / "remote_profiles.json"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    profile = RemoteProfile(
        profile_id="gpu.cluster-01",
        name="GPU cluster A",
        host="2001:db8::1",
        port=22022,
        username="vasp_user",
        host_fingerprint=FINGERPRINT,
        private_key_path="D:/SSH keys/vasp_ed25519",
    )
    _write_v1(legacy_path, profile)

    loaded = ProfileStore().load()

    assert loaded == (profile,)
    assert loaded[0].to_persistent_dict() == profile.to_persistent_dict()


def test_atomic_save_failure_preserves_existing_profiles(tmp_path, monkeypatch):
    target = tmp_path / "profiles.json"
    store = ProfileStore(target)
    original = _write_v1(target, _profile("original"))

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        store.save((_profile("replacement"),))

    assert target.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_profile_store_rejects_secret_or_unknown_persisted_fields(tmp_path):
    target = tmp_path / "profiles.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {
                        "profile_id": "bad",
                        "name": "bad",
                        "host": "host",
                        "port": 22,
                        "username": "user",
                        "host_fingerprint": FINGERPRINT,
                        "private_key_path": None,
                        "password": "must-not-load",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown|secret"):
        ProfileStore(target).load()


def test_normalize_fingerprint_preserves_winscp_sha256_format():
    observed = f"  {WINSCP_FINGERPRINT.replace(' ', '   ')}  "

    assert normalize_fingerprint(observed) == WINSCP_FINGERPRINT


def test_profile_round_trip_preserves_winscp_fingerprint(tmp_path):
    profile = RemoteProfile(
        "winscp-host",
        "WinSCP host",
        "login.example.edu",
        22,
        "researcher",
        WINSCP_FINGERPRINT,
    )
    store = ProfileStore(tmp_path / "profiles.json")

    store.save((profile,))

    assert store.load() == (profile,)
    assert store.load()[0].host_fingerprint == WINSCP_FINGERPRINT


@pytest.mark.parametrize(
    "fingerprint",
    [
        "*",
        "accept-any",
        "SHA256:short",
        "md5:aa:bb",
        "A" * 43,
        "ssh-ed25519 255 too-short",
        "ssh-ed25519 255 abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_profile_rejects_unpinned_or_weak_fingerprints(fingerprint):
    with pytest.raises(ValueError, match="fingerprint"):
        RemoteProfile(
            profile_id="cluster",
            name="Cluster",
            host="host",
            port=22,
            username="user",
            host_fingerprint=fingerprint,
        )


@pytest.mark.parametrize(
    "host",
    ["-oProxyCommand=attack", "host name", "sftp://host", "user@host", "host/path"],
)
def test_profile_rejects_host_values_that_could_become_transport_options(host):
    with pytest.raises(ValueError, match="host"):
        RemoteProfile("cluster", "Cluster", host, 22, "user", FINGERPRINT)


@pytest.mark.parametrize("username", ["-option", "user name", "user@host", "user/name"])
def test_profile_rejects_unsafe_ssh_usernames(username):
    with pytest.raises(ValueError, match="username"):
        RemoteProfile("cluster", "Cluster", "host", 22, username, FINGERPRINT)

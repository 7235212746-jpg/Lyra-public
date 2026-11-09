from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyra.mobile.auth import AuthStore, PairingError


def test_pairing_code_is_one_time_and_only_hashes_are_persisted(tmp_path: Path) -> None:
    state_file = tmp_path / "mobile-auth.json"
    store = AuthStore(state_file=state_file, pairing_ttl=60)
    code = store.issue_pairing_code()

    credentials = store.pair(code, "My phone")

    with pytest.raises(PairingError, match="invalid or expired"):
        store.pair(code, "Second phone")
    saved = state_file.read_text(encoding="utf-8")
    assert credentials.token not in saved
    assert credentials.csrf_token not in saved
    assert code not in saved


def test_device_token_can_be_verified_and_revoked(tmp_path: Path) -> None:
    store = AuthStore(state_file=tmp_path / "auth.json")
    credentials = store.pair(store.issue_pairing_code(), "Phone")

    device = store.authenticate(credentials.token)
    assert device.device_id == credentials.device_id
    assert store.verify_csrf(device, credentials.csrf_token)

    store.revoke(credentials.device_id)
    assert store.authenticate(credentials.token) is None


def test_auth_state_contains_no_plaintext_secrets(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    store = AuthStore(state_file=path)
    credentials = store.pair(store.issue_pairing_code(), "Phone")
    payload = json.loads(path.read_text(encoding="utf-8"))

    device = payload["devices"][0]
    assert device["token_hash"]
    assert device["csrf_hash"]
    assert "token" not in device
    assert "csrf_token" not in device
    assert credentials.token != device["token_hash"]


def test_device_token_expires_and_new_pairing_code_invalidates_old_code(tmp_path: Path) -> None:
    now = [100.0]
    store = AuthStore(
        state_file=tmp_path / "auth.json",
        pairing_ttl=60,
        token_ttl=10,
        clock=lambda: now[0],
    )
    old_code = store.issue_pairing_code()
    new_code = store.issue_pairing_code()
    with pytest.raises(PairingError, match="invalid or expired"):
        store.pair(old_code, "Old phone")
    credentials = store.pair(new_code, "Phone")
    assert store.authenticate(credentials.token) is not None

    now[0] = 111.0
    assert store.authenticate(credentials.token) is None

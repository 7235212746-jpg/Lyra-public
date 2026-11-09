from __future__ import annotations

import pytest

from lyra.remote.profiles import RemoteProfile, RuntimeAuthentication
from lyra.remote.transport import (
    HostKeyMismatch,
    HostKeyUnverified,
    connect_pinned,
)
from tests.test_remote_support import FakeTransport


PIN = "ssh-ed25519 255 SHA256:abcdefghijklmnopqrstuvwx1234567890ABCDE"
OTHER_PIN = "ssh-ed25519 255 SHA256:ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210abcde"


def profile(pin=PIN):
    return RemoteProfile("p", "Cluster", "host", 22, "user", pin)


def test_first_connection_reports_fingerprint_without_connecting():
    transport = FakeTransport(PIN)

    with pytest.raises(HostKeyUnverified) as caught:
        connect_pinned(transport, profile(None), RuntimeAuthentication())

    assert caught.value.observed == PIN
    assert transport.connect_calls == []


def test_changed_host_key_is_blocked_with_old_and_new_values():
    transport = FakeTransport(OTHER_PIN)

    with pytest.raises(HostKeyMismatch) as caught:
        connect_pinned(transport, profile(), RuntimeAuthentication())

    assert caught.value.expected == PIN
    assert caught.value.observed == OTHER_PIN
    assert transport.connect_calls == []


def test_exact_pinned_key_opens_session():
    transport = FakeTransport(PIN)
    session = connect_pinned(transport, profile(), RuntimeAuthentication())

    assert session is transport.session
    assert len(transport.connect_calls) == 1

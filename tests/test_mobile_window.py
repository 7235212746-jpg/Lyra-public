from __future__ import annotations

import http.client
import socket
from pathlib import Path

import pytest

from lyra.mobile_window import MobileServiceController


def test_controller_starts_loopback_service_and_stops_cleanly(tmp_path: Path) -> None:
    controller = MobileServiceController(state_root=tmp_path, port=0)

    address = controller.start()
    connection = http.client.HTTPConnection("127.0.0.1", controller.port, timeout=2)
    connection.request("GET", "/api/health")
    response = connection.getresponse()
    response.read()
    connection.close()

    assert address == f"http://127.0.0.1:{controller.port}"
    assert response.status == 200
    assert controller.running

    controller.stop()
    assert not controller.running


def test_pairing_code_does_not_persist_in_plaintext(tmp_path: Path) -> None:
    controller = MobileServiceController(state_root=tmp_path, port=0)
    code = controller.issue_pairing_code()

    auth_file = tmp_path / "mobile" / "auth.json"
    assert len(code) == 6
    assert not auth_file.exists() or code not in auth_file.read_text(encoding="utf-8")


def test_controller_stop_is_idempotent(tmp_path: Path) -> None:
    controller = MobileServiceController(state_root=tmp_path, port=0)

    controller.stop()
    controller.start()
    controller.stop()
    controller.stop()

    assert not controller.running


def test_controller_closes_backend_when_port_binding_fails(tmp_path: Path) -> None:
    class Backend:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    backend = Backend()
    controller = MobileServiceController(
        state_root=tmp_path,
        port=blocker.getsockname()[1],
        backend=backend,
    )
    try:
        with pytest.raises(OSError):
            controller.start()
    finally:
        controller.stop()
        blocker.close()
    assert backend.closed

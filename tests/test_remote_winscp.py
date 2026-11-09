from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyra.remote.profiles import RemoteProfile, RuntimeAuthentication
from lyra.remote.winscp import WinScpInstallation, WinScpTransport


PIN = "ssh-ed25519 255 SHA256:abcdefghijklmnopqrstuvwx1234567890ABCDE"


class FakeRunner:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, command, *, input, **kwargs):
        request = json.loads(input)
        self.requests.append((command, request, kwargs))
        operation = request["operation"]
        if operation == "scan":
            payload = {"ok": True, "fingerprint": PIN}
        elif operation == "list":
            payload = {
                "ok": True,
                "entries": [{
                    "name": "OUTCAR",
                    "path": "/calc/OUTCAR",
                    "is_directory": False,
                    "size": 120,
                    "modified_ns": 10,
                }],
            }
        elif operation == "get":
            Path(request["local_path"]).write_bytes(b"data")
            payload = {"ok": True, "size": 4}
        elif operation == "home":
            payload = {"ok": True, "home": "/home/user"}
        elif operation == "runtime-preflight":
            payload = {"ok": True, "report": "{\"schema_version\":1}"}
        else:
            payload = {"ok": True}

        class Result:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        return Result()


class FailingRunner:
    def __call__(self, command, *, input, **kwargs):
        request = json.loads(input)

        class Result:
            returncode = 1
            stdout = ""
            stderr = f"authentication failed for {request['authentication']['password']}"

        return Result()


def installation(tmp_path: Path) -> WinScpInstallation:
    exe = tmp_path / "WinSCP.exe"
    dll = tmp_path / "WinSCPnet.dll"
    bridge = tmp_path / "winscp_bridge.ps1"
    for path in (exe, dll, bridge):
        path.write_text("test", encoding="ascii")
    return WinScpInstallation(exe, dll, bridge)


def profile() -> RemoteProfile:
    return RemoteProfile("cluster", "Cluster", "host.example", 22, "user", PIN)


def test_scan_and_connect_never_use_accept_any_or_command_line_secrets(tmp_path: Path) -> None:
    runner = FakeRunner()
    transport = WinScpTransport(installation(tmp_path), runner=runner)

    assert transport.observe_host_fingerprint(profile()) == PIN
    session = transport.connect(
        profile(), RuntimeAuthentication(password="runtime-password")
    )
    assert session.list_directory("/calc")[0].name == "OUTCAR"

    serialized = repr(runner.requests)
    assert "acceptany" not in serialized.casefold()
    assert "hostkey=*" not in serialized.casefold()
    assert all(PIN not in " ".join(request[0]) for request in runner.requests)


def test_runtime_password_and_passphrase_use_stdin_not_the_command_line(tmp_path: Path) -> None:
    runner = FakeRunner()
    transport = WinScpTransport(installation(tmp_path), runner=runner)
    authentication = RuntimeAuthentication(
        password="unique-runtime-password",
        private_key_passphrase="unique-runtime-passphrase",
    )

    session = transport.connect(profile(), authentication)
    assert session.list_directory("/calc")[0].name == "OUTCAR"

    command, request, _kwargs = runner.requests[-1]
    assert request["authentication"] == {
        "password": "unique-runtime-password",
        "private_key_passphrase": "unique-runtime-passphrase",
    }
    assert "unique-runtime-password" not in " ".join(command)
    assert "unique-runtime-passphrase" not in " ".join(command)


def test_runtime_secrets_are_redacted_from_bridge_errors(tmp_path: Path) -> None:
    transport = WinScpTransport(installation(tmp_path), runner=FailingRunner())
    session = transport.connect(
        profile(), RuntimeAuthentication(password="do-not-leak-this")
    )

    with pytest.raises(Exception) as captured:
        session.list_directory("/calc")

    assert "do-not-leak-this" not in str(captured.value)
    assert "<redacted>" in str(captured.value)


def test_connect_rejects_missing_password_and_private_key_before_opening_session(
    tmp_path: Path,
) -> None:
    transport = WinScpTransport(installation(tmp_path), runner=FakeRunner())

    with pytest.raises(ValueError, match="SSH 密码.*私钥"):
        transport.connect(profile(), RuntimeAuthentication())


def test_bridge_assigns_runtime_credentials_to_session_options() -> None:
    bridge = (
        Path(__file__).resolve().parents[1]
        / "lyra"
        / "remote"
        / "winscp_bridge.ps1"
    ).read_text(encoding="utf-8")

    assert "$options.Password" in bridge
    assert "$options.PrivateKeyPassphrase" in bridge


def test_read_bytes_honours_size_bound(tmp_path: Path) -> None:
    session = WinScpTransport(installation(tmp_path), runner=FakeRunner()).connect(
        profile(), RuntimeAuthentication(password="runtime-password")
    )

    assert session.read_bytes("/calc/OUTCAR", max_bytes=4) == b"data"
    with pytest.raises(ValueError, match="bound"):
        session.read_bytes("/calc/OUTCAR", max_bytes=3)


def test_remote_home_and_preflight_use_fixed_bridge_operations(tmp_path: Path) -> None:
    runner = FakeRunner()
    session = WinScpTransport(installation(tmp_path), runner=runner).connect(
        profile(), RuntimeAuthentication(password="runtime-password")
    )

    assert session.remote_home() == "/home/user"
    assert session.runtime_preflight(
        "/home/user/.cache/lyra/preflight.py",
        "/home/user/.cache/lyra",
    ) == b'{"schema_version":1}'
    operations = [request[1]["operation"] for request in runner.requests]
    assert operations[-2:] == ["home", "runtime-preflight"]


def test_bridge_runtime_preflight_quotes_only_validated_paths() -> None:
    bridge = (
        Path(__file__).resolve().parents[1]
        / "lyra"
        / "remote"
        / "winscp_bridge.ps1"
    ).read_text(encoding="utf-8")

    assert '"runtime-preflight"' in bridge
    assert "Quote-Shell ([string]$request.script_path)" in bridge
    assert "Quote-Shell ([string]$request.cache_root)" in bridge

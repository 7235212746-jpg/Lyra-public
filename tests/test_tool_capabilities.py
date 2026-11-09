from __future__ import annotations

from pathlib import Path

from lyra.tool_capabilities import ToolCapabilities, find_executable


def test_find_executable_prefers_path_lookup(tmp_path: Path) -> None:
    path_binary = tmp_path / "winscp.com"
    path_binary.write_text("test", encoding="ascii")

    result = find_executable(
        "winscp.com",
        candidates=(tmp_path / "other" / "WinSCP.com",),
        which=lambda _name: str(path_binary),
    )

    assert result == path_binary.resolve()


def test_find_executable_uses_existing_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "WinSCP" / "WinSCP.com"
    candidate.parent.mkdir()
    candidate.write_text("test", encoding="ascii")

    result = find_executable(
        "winscp.com",
        candidates=(candidate,),
        which=lambda _name: None,
    )

    assert result == candidate.resolve()


def test_detect_reports_independent_remote_and_mobile_capabilities(
    tmp_path: Path,
) -> None:
    winscp = tmp_path / "WinSCP.exe"
    winscp.with_name("WinSCPnet.dll").write_text("test", encoding="ascii")
    ssh = tmp_path / "ssh.exe"
    tailscale = tmp_path / "tailscale.exe"
    for executable in (winscp, ssh, tailscale):
        executable.write_text("test", encoding="ascii")

    locations = {
        "WinSCP.exe": winscp,
        "ssh.exe": ssh,
        "tailscale.exe": tailscale,
    }
    detected = ToolCapabilities.detect(
        finder=lambda name, **_kwargs: locations.get(name)
    )

    assert detected.winscp == winscp
    assert detected.openssh == ssh
    assert detected.tailscale == tailscale
    assert detected.remote_ready
    assert detected.mobile_private_network_ready


def test_detect_does_not_claim_remote_ready_without_winscp_assembly(tmp_path: Path) -> None:
    winscp = tmp_path / "WinSCP.exe"
    winscp.write_text("test", encoding="ascii")

    detected = ToolCapabilities.detect(
        finder=lambda name, **_kwargs: winscp if name == "WinSCP.exe" else None
    )

    assert detected.winscp is None
    assert not detected.remote_ready


def test_remote_ready_requires_the_implemented_winscp_backend() -> None:
    detected = ToolCapabilities(winscp=None, openssh=Path("ssh.exe"), tailscale=None)

    assert not detected.remote_ready
    assert not detected.mobile_private_network_ready
    assert "OpenSSH" in detected.remote_summary
    assert "WinSCP" in detected.remote_summary
    assert "Tailscale" in detected.mobile_summary

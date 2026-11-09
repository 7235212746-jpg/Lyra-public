from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable


ExecutableFinder = Callable[..., Path | None]


def find_executable(
    name: str,
    *,
    candidates: Iterable[Path] = (),
    which: Callable[[str], str | None] = shutil.which,
) -> Path | None:
    located = which(name)
    if located:
        path = Path(located).expanduser()
        if path.is_file():
            return path.resolve()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _program_roots() -> tuple[Path, ...]:
    values = (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
    )
    return tuple(Path(value) for value in values if value)


@dataclass(frozen=True)
class ToolCapabilities:
    winscp: Path | None
    openssh: Path | None
    tailscale: Path | None

    @classmethod
    def detect(cls, *, finder: ExecutableFinder = find_executable) -> "ToolCapabilities":
        roots = _program_roots()
        winscp_candidates = tuple(
            candidate
            for root in roots
            for candidate in (
                root / "WinSCP" / "WinSCP.exe",
                root / "Programs" / "WinSCP" / "WinSCP.exe",
            )
        )
        tailscale_candidates = tuple(
            candidate
            for root in roots
            for candidate in (
                root / "Tailscale" / "tailscale.exe",
                root / "Programs" / "Tailscale" / "tailscale.exe",
            )
        )
        ssh_candidates = (
            Path(os.environ.get("WINDIR", r"C:\Windows"))
            / "System32"
            / "OpenSSH"
            / "ssh.exe",
        )
        winscp = finder("WinSCP.exe", candidates=winscp_candidates)
        if winscp is not None and not winscp.with_name("WinSCPnet.dll").is_file():
            winscp = None
        return cls(
            winscp=winscp,
            openssh=finder("ssh.exe", candidates=ssh_candidates),
            tailscale=finder("tailscale.exe", candidates=tailscale_candidates),
        )

    @property
    def remote_ready(self) -> bool:
        return self.winscp is not None

    @property
    def mobile_private_network_ready(self) -> bool:
        return self.tailscale is not None

    @property
    def remote_summary(self) -> str:
        if self.winscp is not None:
            return f"WinSCP 已就绪：{self.winscp}"
        if self.openssh is not None:
            return f"OpenSSH 已就绪：{self.openssh}；Lyra 图形远程连接仍需 WinSCP"
        return "未检测到 WinSCP 或 OpenSSH，远程连接暂不可用"

    @property
    def mobile_summary(self) -> str:
        if self.tailscale is not None:
            return f"Tailscale 已就绪：{self.tailscale}"
        return "Tailscale 未检测到；手机服务只能在本机回环地址启动"

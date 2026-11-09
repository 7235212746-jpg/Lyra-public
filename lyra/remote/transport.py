from __future__ import annotations

from dataclasses import dataclass
import hmac
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from .profiles import RemoteProfile, RuntimeAuthentication, normalize_fingerprint


class RemoteError(RuntimeError):
    pass


class RemoteDisconnected(RemoteError):
    pass


class HostKeyUnverified(RemoteError):
    def __init__(self, observed: str) -> None:
        self.observed = observed
        super().__init__("The server host key has not been confirmed.")


class HostKeyMismatch(RemoteError):
    def __init__(self, expected: str, observed: str) -> None:
        self.expected = expected
        self.observed = observed
        super().__init__("The server host key differs from the pinned fingerprint.")


@dataclass(frozen=True)
class RemoteStat:
    name: str
    path: str
    is_directory: bool
    size: int
    modified_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name in {".", ".."}:
            raise ValueError("Remote entry name is invalid.")
        if "/" in self.name or "\\" in self.name or any(ord(c) < 32 for c in self.name):
            raise ValueError("Remote entry name contains a separator or control character.")
        validate_remote_path(self.path)
        if not isinstance(self.is_directory, bool):
            raise ValueError("is_directory must be boolean.")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("Remote entry size must be a non-negative integer.")
        if isinstance(self.modified_ns, bool) or not isinstance(self.modified_ns, int) or self.modified_ns < 0:
            raise ValueError("Remote entry timestamp must be a non-negative integer.")


def validate_remote_path(path: str, *, allow_root: bool = True) -> str:
    if not isinstance(path, str) or not path or len(path) > 4096:
        raise ValueError("Remote path must be a bounded absolute POSIX path.")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ValueError("Remote path contains control characters.")
    pure = PurePosixPath(path)
    if not pure.is_absolute():
        raise ValueError("Remote path must be absolute.")
    if any(part == ".." for part in pure.parts):
        raise ValueError("Remote path may not contain parent traversal.")
    normalized = str(pure)
    if normalized == "/" and not allow_root:
        raise ValueError("Remote root path is not allowed here.")
    return normalized


@runtime_checkable
class RemoteSession(Protocol):
    def list_directory(self, path: str) -> Sequence[RemoteStat]: ...

    def read_bytes(self, path: str, *, max_bytes: int) -> bytes: ...

    def write_bytes_atomic(self, path: str, data: bytes, *, mode: int = 0o600) -> None: ...

    def download_file(self, remote_path: str, local_path: Path, *, max_bytes: int) -> int: ...

    def start_job(self, job_id: UUID) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class RemoteTransport(Protocol):
    def observe_host_fingerprint(self, profile: RemoteProfile) -> str: ...

    def connect(
        self,
        profile: RemoteProfile,
        authentication: RuntimeAuthentication,
    ) -> RemoteSession:
        """Open SSH/SFTP while enforcing profile.host_fingerprint in the handshake."""
        ...


def connect_pinned(
    transport: RemoteTransport,
    profile: RemoteProfile,
    authentication: RuntimeAuthentication,
) -> RemoteSession:
    observed = normalize_fingerprint(transport.observe_host_fingerprint(profile))
    expected = profile.host_fingerprint
    if expected is None:
        raise HostKeyUnverified(observed)
    expected = normalize_fingerprint(expected)
    if not hmac.compare_digest(expected.encode("ascii"), observed.encode("ascii")):
        raise HostKeyMismatch(expected, observed)
    return transport.connect(profile, authentication)

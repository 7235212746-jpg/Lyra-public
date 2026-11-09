from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Callable
from uuid import UUID

from lyra.remote.transport import RemoteStat


class FakeSession:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.directories: dict[str, tuple[RemoteStat, ...]] = {}
        self.started: list[UUID] = []
        self.closed = False
        self.read_error: Exception | None = None
        self.download_hook: Callable[[str, Path], None] | None = None

    def list_directory(self, path: str) -> tuple[RemoteStat, ...]:
        return self.directories.get(path, ())

    def read_bytes(self, path: str, *, max_bytes: int) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        data = self.files[path]
        if len(data) > max_bytes:
            raise ValueError("remote payload exceeds bound")
        return data

    def write_bytes_atomic(self, path: str, data: bytes, *, mode: int = 0o600) -> None:
        assert PurePosixPath(path).is_absolute()
        assert mode == 0o600
        self.files[path] = bytes(data)

    def download_file(self, remote_path: str, local_path: Path, *, max_bytes: int) -> int:
        if self.download_hook is not None:
            self.download_hook(remote_path, local_path)
            return local_path.stat().st_size
        data = self.files[remote_path]
        if len(data) > max_bytes:
            raise ValueError("remote file exceeds bound")
        local_path.write_bytes(data)
        return len(data)

    def start_job(self, job_id: UUID) -> None:
        self.started.append(job_id)

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, fingerprint: str, session: FakeSession | None = None) -> None:
        self.fingerprint = fingerprint
        self.session = session or FakeSession()
        self.connect_calls = []

    def observe_host_fingerprint(self, profile):
        return self.fingerprint

    def connect(self, profile, authentication):
        self.connect_calls.append((profile, authentication))
        return self.session

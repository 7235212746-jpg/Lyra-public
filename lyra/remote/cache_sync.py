from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

from .protocol import RemoteResult
from .transport import RemoteSession


class SelectiveCacheSync:
    def __init__(self, session: RemoteSession) -> None:
        self._session = session

    def sync(
        self,
        result: RemoteResult,
        destination: str | Path,
        *,
        roles: set[str] | frozenset[str],
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> tuple[Path, ...]:
        if not isinstance(max_total_bytes, int) or isinstance(max_total_bytes, bool) or max_total_bytes < 0:
            raise ValueError("Cache synchronization limit must be a non-negative integer.")
        if not isinstance(roles, set | frozenset) or any(not isinstance(role, str) for role in roles):
            raise ValueError("Cache synchronization roles must be a set of strings.")
        selected = tuple(item for item in result.artifacts if item.role in roles)
        if sum(item.size for item in selected) > max_total_bytes:
            raise ValueError("Selected remote artifacts exceed the synchronization limit.")

        root = Path(destination).resolve()
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".lyra-remote-sync-{uuid4().hex}.tmp"
        published: list[Path] = []
        try:
            staging.mkdir()
            staged: list[tuple[Path, Path]] = []
            for item in selected:
                relative = PurePosixPath(item.relative_path)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise ValueError("Artifact relative path is unsafe.")
                local_relative = Path(*relative.parts)
                target = (root / local_relative).resolve()
                try:
                    target.relative_to(root)
                except ValueError as exc:
                    raise ValueError("Artifact relative path escapes the cache destination.") from exc
                temporary = staging / local_relative
                temporary.parent.mkdir(parents=True, exist_ok=True)
                transferred = self._session.download_file(
                    item.remote_path,
                    temporary,
                    max_bytes=item.size,
                )
                actual_size = temporary.stat().st_size
                if transferred != item.size or actual_size != item.size:
                    raise ValueError(f"Artifact size verification failed for {item.relative_path}.")
                digest = hashlib.sha256()
                with temporary.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item.sha256:
                    raise ValueError(f"Artifact hash verification failed for {item.relative_path}.")
                staged.append((temporary, target))

            for temporary, target in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, target)
                published.append(target)
            return tuple(published)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

from __future__ import annotations

from pathlib import PurePosixPath

from .transport import RemoteSession, RemoteStat, validate_remote_path


class RemoteBrowser:
    def __init__(self, session: RemoteSession) -> None:
        self._session = session

    def list_directory(self, path: str) -> tuple[RemoteStat, ...]:
        normalized = validate_remote_path(path)
        entries = tuple(self._session.list_directory(normalized))
        for entry in entries:
            if PurePosixPath(entry.path).parent != PurePosixPath(normalized):
                raise ValueError("Remote listing returned an entry outside the requested directory.")
        return tuple(
            sorted(entries, key=lambda item: (not item.is_directory, item.name.casefold(), item.name))
        )

    @staticmethod
    def parent(path: str) -> str:
        normalized = validate_remote_path(path)
        return str(PurePosixPath(normalized).parent)

from __future__ import annotations

from typing import Protocol

from lyra.lifecycle import SingleInstanceGuard


class Closeable(Protocol):
    def close(self) -> None: ...


class Saveable(Protocol):
    def save(self) -> None: ...


class ApplicationLifecycle:
    def __init__(
        self,
        worker: Closeable,
        settings: Saveable | None,
        guard: SingleInstanceGuard,
    ) -> None:
        self._worker = worker
        self._settings = settings
        self._guard = guard
        self._closed = False
        self.settings_error: Exception | None = None

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._settings is not None:
                try:
                    self._settings.save()
                except Exception as exc:
                    self.settings_error = exc
        finally:
            try:
                self._worker.close()
            finally:
                self._guard.release()

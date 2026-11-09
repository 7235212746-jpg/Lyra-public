from __future__ import annotations

from multiprocessing import freeze_support
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lyra.settings_store import SettingsStore


class SingleInstanceGuard:
    def __init__(self, name: str, *, lock_root: Path | None = None) -> None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "lyra"
        root = Path(lock_root) if lock_root is not None else Path(tempfile.gettempdir())
        self.path = root / f"{safe_name}.lock"
        self._owned = False

    def acquire(self) -> bool:
        if self._owned:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                if not self._remove_stale_lock():
                    return False
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
                handle.flush()
                os.fsync(handle.fileno())
            self._owned = True
            return True
        return False

    def _remove_stale_lock(self) -> bool:
        try:
            owner = int(self.path.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            owner = -1
        if owner > 0 and self._process_exists(owner):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _process_exists(pid: int) -> bool:
        if pid == os.getpid():
            return True
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            error_access_denied = 5
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            open_process.restype = wintypes.HANDLE
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            get_exit_code = kernel32.GetExitCodeProcess
            get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            get_exit_code.restype = wintypes.BOOL
            handle = open_process(process_query_limited_information, False, pid)
            if handle:
                try:
                    exit_code = wintypes.DWORD()
                    if not get_exit_code(handle, ctypes.byref(exit_code)):
                        return True
                    return exit_code.value == 259
                finally:
                    close_handle(handle)
            return ctypes.get_last_error() == error_access_denied
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self._owned:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._owned = False

    def __enter__(self) -> "SingleInstanceGuard":
        if not self.acquire():
            raise RuntimeError("Another Lyra instance is already running.")
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


def _default_settings_store() -> SettingsStore:
    from lyra.settings_store import SettingsStore

    return SettingsStore.default()


def _create_classic_app(*, settings_store: SettingsStore):
    from lyra.app import LyraApp

    return LyraApp(settings_store=settings_store)


def run_classic(settings_store: SettingsStore | None = None) -> int:
    freeze_support()
    guard = SingleInstanceGuard("lyra-classic")
    if not guard.acquire():
        return 0
    try:
        store = settings_store if settings_store is not None else _default_settings_store()
        app = _create_classic_app(settings_store=store)
        app.mainloop()
        return 0
    finally:
        guard.release()

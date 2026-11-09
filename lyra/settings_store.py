from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


SETTINGS_SCHEMA_VERSION = 2
_CORE_NAMESPACES = ("classic", "qt", "shared")
_LOCK_TIMEOUT_SECONDS = 10.0
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class SettingsFormatError(ValueError):
    """Raised when saving would overwrite an unreadable settings document."""


class SettingsLockTimeoutError(TimeoutError):
    """Raised when another process holds the settings transaction too long."""


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _resolved_path_key(path: Path) -> str:
    return os.path.normcase(str(_resolved_path(path)))


def _path_lock(path: Path) -> threading.RLock:
    key = _resolved_path_key(path)
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _windows_interprocess_lock(path: Path) -> Iterator[None]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    create_mutex.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    release_mutex = kernel32.ReleaseMutex
    release_mutex.argtypes = (wintypes.HANDLE,)
    release_mutex.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    digest = hashlib.sha256(_resolved_path_key(path).encode("utf-8")).hexdigest()
    handle = create_mutex(None, False, f"Local\\Lyra.Settings.{digest}")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    acquired = False
    try:
        result = wait_for_single_object(handle, int(_LOCK_TIMEOUT_SECONDS * 1000))
        if result in (0x00000000, 0x00000080):
            acquired = True
        elif result == 0x00000102:
            raise SettingsLockTimeoutError(
                f"timed out waiting for settings lock: {path}"
            )
        else:
            raise ctypes.WinError(ctypes.get_last_error())
        yield
    finally:
        if acquired:
            release_mutex(handle)
        close_handle(handle)


def _posix_lock_path(path: Path) -> Path:
    canonical = _resolved_path(path)
    return canonical.with_name(f"{canonical.name}.lock")


@contextmanager
def _posix_interprocess_lock(path: Path) -> Iterator[None]:
    import fcntl

    lock_path = _posix_lock_path(path)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SettingsLockTimeoutError(
                        f"timed out waiting for settings lock: {path}"
                    )
                time.sleep(0.05)
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _interprocess_lock(path: Path):
    if os.name == "nt":
        return _windows_interprocess_lock(path)
    return _posix_interprocess_lock(path)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SettingsFormatError(
                f"settings JSON contains duplicate object key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise SettingsFormatError(
        f"settings JSON contains non-finite numeric constant {value!r}"
    )


def _decode_document(raw: bytes) -> tuple[int, dict[str, object]]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SettingsFormatError("settings file is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise SettingsFormatError("settings document must be a JSON object")

    schema_version = document.get("schema_version")
    if type(schema_version) is not int:
        raise SettingsFormatError("settings schema version must be an integer")
    if schema_version == 1:
        if not isinstance(document.get("settings"), dict):
            raise SettingsFormatError("settings schema 1 requires an object payload")
        return 1, document
    if schema_version != SETTINGS_SCHEMA_VERSION:
        raise SettingsFormatError(
            f"settings schema {schema_version!r} is not supported for saving"
        )

    namespaces = document.get("namespaces")
    if not isinstance(namespaces, dict):
        raise SettingsFormatError("settings schema 2 requires a namespaces object")
    for name in _CORE_NAMESPACES:
        if not isinstance(namespaces.get(name), dict):
            raise SettingsFormatError(
                f"settings schema 2 requires the {name!r} namespace"
            )
    if any(
        not isinstance(name, str) or not isinstance(value, dict)
        for name, value in namespaces.items()
    ):
        raise SettingsFormatError("settings namespaces must contain JSON objects")
    return SETTINGS_SCHEMA_VERSION, document


def _new_document() -> dict[str, object]:
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "namespaces": {name: {} for name in _CORE_NAMESPACES},
    }


def _migrate_v1(document: dict[str, object]) -> dict[str, object]:
    migrated = copy.deepcopy(document)
    classic = migrated.pop("settings")
    migrated["schema_version"] = SETTINGS_SCHEMA_VERSION
    migrated["namespaces"] = {
        "classic": classic,
        "qt": {},
        "shared": {},
    }
    return migrated


def _merge_mapping(
    existing: Mapping[str, object], incoming: Mapping[str, object]
) -> dict[str, object]:
    merged = copy.deepcopy(dict(existing))
    for key, value in incoming.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mapping(previous, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _backup_candidates(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.v1.*.bak"))


def _ensure_v1_backup(path: Path, original: bytes) -> None:
    for candidate in _backup_candidates(path):
        if candidate.read_bytes() == original:
            return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_path = path.with_name(
        f"{path.name}.v1.{timestamp}.{uuid.uuid4().hex}.bak"
    )
    try:
        with backup_path.open("xb") as backup:
            backup.write(original)
            backup.flush()
            os.fsync(backup.fileno())
    except BaseException:
        try:
            backup_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write(path: Path, document: Mapping[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                document,
                temporary,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def default(cls) -> "SettingsStore":
        appdata = os.environ.get("APPDATA")
        root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return cls(root / "Lyra" / "settings.json")

    def load_namespace(self, name: str) -> dict:
        if not isinstance(name, str):
            raise TypeError("settings namespace name must be a string")
        target_path = _resolved_path(self.path)
        try:
            raw = target_path.read_bytes()
            schema_version, document = _decode_document(raw)
        except (OSError, SettingsFormatError):
            return {}

        if schema_version == 1:
            value = document["settings"] if name == "classic" else {}
        else:
            namespaces = document["namespaces"]
            assert isinstance(namespaces, dict)
            value = namespaces.get(name, {})
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def save_namespace(
        self, name: str, settings: Mapping[str, object]
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("settings namespace name must be a string")
        if not isinstance(settings, Mapping):
            raise TypeError("settings namespace payload must be a mapping")

        target_path = _resolved_path(self.path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with _path_lock(target_path):
            with _interprocess_lock(target_path):
                try:
                    original = target_path.read_bytes()
                except FileNotFoundError:
                    original = None

                if original is None:
                    document = _new_document()
                else:
                    schema_version, loaded = _decode_document(original)
                    if schema_version == 1:
                        _ensure_v1_backup(target_path, original)
                        document = _migrate_v1(loaded)
                    else:
                        document = copy.deepcopy(loaded)

                namespaces = document["namespaces"]
                assert isinstance(namespaces, dict)
                previous = namespaces.get(name, {})
                if not isinstance(previous, Mapping):
                    raise SettingsFormatError(
                        f"settings namespace {name!r} is not an object"
                    )
                namespaces[name] = _merge_mapping(previous, settings)
                _atomic_write(target_path, document)

    def load(self) -> dict:
        return self.load_namespace("classic")

    def save(self, settings: Mapping[str, object]) -> None:
        self.save_namespace("classic", settings)

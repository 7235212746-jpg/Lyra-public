from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from .transport import validate_remote_path


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?")
MAX_PREFLIGHT_BYTES = 64 * 1024


_PREFLIGHT_SCRIPT = r'''from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

def main():
    if len(sys.argv) != 2:
        raise SystemExit(2)
    cache = Path(sys.argv[1])
    cache_is_symlink = cache.is_symlink()
    if not cache_is_symlink:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            cache.chmod(0o700)
        except OSError:
            pass
    report = {
        "schema_version": 1,
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": {
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
            "PIL": package_version("Pillow"),
        },
        "cache_writable": (not cache_is_symlink and os.access(cache, os.W_OK)),
        "cache_is_symlink": cache_is_symlink,
        "free_bytes": shutil.disk_usage(cache.parent if cache_is_symlink else cache).free,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class RemoteRuntimeBundle:
    payload: bytes
    digest: str

    def as_file(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)
        return target


@dataclass(frozen=True)
class RemoteRuntimePaths:
    cache_root: str
    runtime_root: str
    runtime_zip: str
    agent_path: str
    preflight_path: str


@dataclass(frozen=True)
class RemoteRuntimeRequirements:
    python: tuple[int, int, int] = (3, 12, 0)
    numpy: tuple[int, int, int] = (2, 0, 0)
    matplotlib: tuple[int, int, int] = (3, 9, 0)
    pillow: tuple[int, int, int] = (10, 0, 0)


@dataclass(frozen=True)
class RemotePreflightReport:
    python: str
    packages: dict[str, str | None]
    cache_writable: bool
    cache_is_symlink: bool
    free_bytes: int
    compatible: bool
    reasons: tuple[str, ...]


def preflight_script_bytes() -> bytes:
    return _PREFLIGHT_SCRIPT.encode("utf-8")


def build_runtime_bundle(package_root: str | Path) -> RemoteRuntimeBundle:
    root = Path(package_root).resolve()
    if not root.is_dir() or not (root / "__init__.py").is_file():
        raise ValueError("Lyra package root is invalid.")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        files = sorted(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
        for path in files:
            relative = path.relative_to(root.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    payload = stream.getvalue()
    return RemoteRuntimeBundle(payload, hashlib.sha256(payload).hexdigest())


def runtime_remote_paths(home: str, digest: str) -> RemoteRuntimePaths:
    normalized = validate_remote_path(home, allow_root=False).rstrip("/")
    if normalized != home.rstrip("/") or "\\" in home or not _DIGEST.fullmatch(digest):
        raise ValueError("Remote runtime home or digest is unsafe.")
    root = str(PurePosixPath(normalized) / ".cache" / "lyra")
    runtime = str(PurePosixPath(root) / "runtime" / digest)
    return RemoteRuntimePaths(
        cache_root=root,
        runtime_root=runtime,
        runtime_zip=str(PurePosixPath(runtime) / "lyra-runtime.zip"),
        agent_path=str(PurePosixPath(root) / "agent.py"),
        preflight_path=str(PurePosixPath(root) / "preflight.py"),
    )


def _version(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _VERSION.match(value)
    if match is None:
        return None
    parts = tuple(int(part or 0) for part in match.groups())
    return parts


def decode_preflight_report(
    raw: bytes,
    requirements: RemoteRuntimeRequirements | None = None,
) -> RemotePreflightReport:
    if not isinstance(raw, bytes) or len(raw) > MAX_PREFLIGHT_BYTES:
        raise ValueError("Remote preflight report exceeds its size limit.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Remote preflight report is invalid JSON.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Remote preflight report schema is unsupported.")
    packages = value.get("packages")
    if not isinstance(packages, dict) or set(packages) != {"numpy", "matplotlib", "PIL"}:
        raise ValueError("Remote preflight package report is invalid.")
    python = value.get("python")
    writable = value.get("cache_writable")
    symlink = value.get("cache_is_symlink")
    free_bytes = value.get("free_bytes")
    if (
        not isinstance(python, str)
        or not isinstance(writable, bool)
        or not isinstance(symlink, bool)
        or isinstance(free_bytes, bool)
        or not isinstance(free_bytes, int)
        or free_bytes < 0
        or any(item is not None and not isinstance(item, str) for item in packages.values())
    ):
        raise ValueError("Remote preflight report fields are invalid.")
    required = requirements or RemoteRuntimeRequirements()
    reasons: list[str] = []
    checks = (
        ("Python", python, required.python),
        ("NumPy", packages["numpy"], required.numpy),
        ("Matplotlib", packages["matplotlib"], required.matplotlib),
        ("Pillow", packages["PIL"], required.pillow),
    )
    for label, actual, minimum in checks:
        parsed = _version(actual)
        if parsed is None:
            reasons.append(f"未检测到可用的 {label}。")
        elif parsed < minimum:
            reasons.append(
                f"{label} {actual} 过旧；需要 {'.'.join(map(str, minimum))} 或更高版本。"
            )
    if symlink:
        reasons.append("远端 Lyra 缓存目录不能是符号链接。")
    if not writable:
        reasons.append("远端 Lyra 缓存目录不可写。")
    return RemotePreflightReport(
        python=python,
        packages=dict(packages),
        cache_writable=writable,
        cache_is_symlink=symlink,
        free_bytes=free_bytes,
        compatible=not reasons,
        reasons=tuple(reasons),
    )

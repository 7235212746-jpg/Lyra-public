from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from lyra.remote.runtime import (
    RemoteRuntimeRequirements,
    build_runtime_bundle,
    decode_preflight_report,
    preflight_script_bytes,
    runtime_remote_paths,
)


def test_runtime_bundle_is_deterministic_and_excludes_generated_files(tmp_path: Path) -> None:
    package = tmp_path / "lyra"
    package.mkdir()
    (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (package / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "worker.pyc").write_bytes(b"generated")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "worker.pyc").write_bytes(b"generated")

    first = build_runtime_bundle(package)
    second = build_runtime_bundle(package)

    assert first.digest == second.digest
    assert first.payload == second.payload
    with zipfile.ZipFile(first.as_file(tmp_path / "runtime.zip")) as archive:
        assert archive.namelist() == ["lyra/__init__.py", "lyra/worker.py"]


def test_runtime_paths_are_scoped_to_remote_home_and_digest() -> None:
    paths = runtime_remote_paths("/home/user", "a" * 64)

    assert paths.cache_root == "/home/user/.cache/lyra"
    assert paths.runtime_zip.endswith(f"/{'a' * 64}/lyra-runtime.zip")
    assert paths.agent_path == "/home/user/.cache/lyra/agent.py"


@pytest.mark.parametrize("home", ["relative", "/", "/home/../etc", "/home/a\\b"])
def test_runtime_paths_reject_unsafe_home(home: str) -> None:
    with pytest.raises(ValueError):
        runtime_remote_paths(home, "b" * 64)


def test_preflight_report_accepts_compatible_runtime() -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "python": "3.12.4",
            "packages": {"numpy": "2.1.0", "matplotlib": "3.9.1", "PIL": "10.4.0"},
            "cache_writable": True,
            "cache_is_symlink": False,
            "free_bytes": 10_000_000,
        }
    ).encode()

    report = decode_preflight_report(raw, RemoteRuntimeRequirements())

    assert report.compatible is True
    assert report.reasons == ()


def test_preflight_report_explains_incompatible_dependencies() -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "python": "3.11.9",
            "packages": {"numpy": "1.26.0", "matplotlib": None, "PIL": "10.0.0"},
            "cache_writable": False,
            "cache_is_symlink": True,
            "free_bytes": 0,
        }
    ).encode()

    report = decode_preflight_report(raw, RemoteRuntimeRequirements())

    assert report.compatible is False
    assert any("Python" in reason for reason in report.reasons)
    assert any("NumPy" in reason for reason in report.reasons)
    assert any("Matplotlib" in reason for reason in report.reasons)
    assert any("符号链接" in reason for reason in report.reasons)


def test_preflight_report_rejects_unbounded_or_unknown_schema() -> None:
    with pytest.raises(ValueError):
        decode_preflight_report(b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError):
        decode_preflight_report(b'{"schema_version":2}')


def test_preflight_payload_is_standard_library_and_reports_required_fields() -> None:
    script = preflight_script_bytes().decode("utf-8")

    assert "import numpy" not in script
    assert "import matplotlib" not in script
    assert "import PIL" not in script
    assert "importlib.metadata.version" in script
    assert '"schema_version"' in script
    assert "cache_is_symlink" in script

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyra.remote.protocol import Artifact, RemoteResult
from lyra.remote.runner import RemoteRunner
from tests.test_remote_support import FakeSession


class RuntimeSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.remote_agent_path = "/wrong/agent.py"
        self.uploads = []

    def remote_home(self) -> str:
        return "/home/user"

    def runtime_preflight(self, script_path: str, cache_root: str) -> bytes:
        assert script_path == "/home/user/.cache/lyra/preflight.py"
        return json.dumps(
            {
                "schema_version": 1,
                "python": "3.12.5",
                "packages": {"numpy": "2.1", "matplotlib": "3.9", "PIL": "10.4"},
                "cache_writable": True,
                "cache_is_symlink": False,
                "free_bytes": 10**9,
            }
        ).encode()

    def write_bytes_atomic(self, path: str, data: bytes, *, mode: int = 0o600) -> None:
        self.uploads.append((path, mode))
        self.files[path] = data


def test_prepare_deploys_content_addressed_runtime_and_sets_real_agent_path() -> None:
    session = RuntimeSession()
    runner = RemoteRunner(session, Path(__file__).parents[1] / "lyra")

    report = runner.prepare()

    assert report.compatible
    assert session.remote_agent_path == "/home/user/.cache/lyra/agent.py"
    paths = [path for path, _mode in session.uploads]
    assert "/home/user/.cache/lyra/preflight.py" in paths
    assert "/home/user/.cache/lyra/agent.py" in paths
    assert any(path.endswith("/lyra-runtime.zip") for path in paths)


def test_start_adds_runtime_identity_without_putting_source_in_command() -> None:
    session = RuntimeSession()
    runner = RemoteRunner(session, Path(__file__).parents[1] / "lyra")
    runner.prepare()

    job_id = runner.start("projected_band", "/work/calc", "a" * 64, {"mode": "standard"})

    request = json.loads(
        session.files[f"/home/user/.cache/lyra/jobs/{job_id}/request.json"]
    )
    assert request["source_paths"] == ["/work/calc"]
    assert request["options"]["workflow"] == "projected_band"
    assert request["options"]["runtime_zip"].endswith("/lyra-runtime.zip")
    assert session.started == [job_id]


def test_fetch_preview_verifies_hash_and_returns_bytes_without_destination(tmp_path: Path) -> None:
    session = RuntimeSession()
    runner = RemoteRunner(session, Path(__file__).parents[1] / "lyra")
    payload = b"png-data"
    import hashlib

    remote = "/home/user/.cache/lyra/jobs/x/artifacts/preview.png"
    session.files[remote] = payload
    result = RemoteResult(
        __import__("uuid").uuid4(),
        True,
        {},
        (Artifact("preview", remote, "preview.png", len(payload), hashlib.sha256(payload).hexdigest(), "image/png"),),
    )

    assert runner.fetch_preview_bytes(result) == payload
    bad = RemoteResult(result.job_id, True, {}, (Artifact("preview", remote, "preview.png", len(payload), "0" * 64, "image/png"),))
    with pytest.raises(ValueError, match="hash"):
        runner.fetch_preview_bytes(bad)

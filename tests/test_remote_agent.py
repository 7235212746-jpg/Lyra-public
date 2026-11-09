from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from lyra.remote.agent import resolve_runtime_zip, validate_job_id


def test_agent_accepts_only_canonical_job_uuid() -> None:
    job_id = uuid4()
    assert validate_job_id(str(job_id)) == job_id
    for value in (str(job_id).upper(), "../job", "", "not-a-uuid"):
        with pytest.raises(ValueError):
            validate_job_id(value)


def test_agent_runtime_zip_must_be_content_addressed_below_cache(tmp_path: Path) -> None:
    cache = tmp_path / ".cache" / "lyra"
    digest = "b" * 64
    runtime = cache / "runtime" / digest / "lyra-runtime.zip"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"zip")

    assert resolve_runtime_zip(cache, str(runtime)) == runtime.resolve()
    with pytest.raises(ValueError):
        resolve_runtime_zip(cache, str(tmp_path / "outside.zip"))

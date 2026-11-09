from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from lyra.remote.protocol import (
    Artifact,
    JobProgress,
    RemoteRequest,
    RemoteResult,
    RemoteStatus,
    RemoteTaskState,
    StatusCursor,
    decode_request,
    decode_result,
    decode_status,
    encode_message,
)


def test_protocol_round_trips_closed_request_status_and_result():
    job_id = uuid4()
    request = RemoteRequest(
        job_id,
        "scan",
        ("/work/materials",),
        {"max_depth": 12},
        "opaque-cancel-token-123456",
    )
    status = RemoteStatus(job_id, 2, RemoteTaskState.RUNNING, 100.5, JobProgress("scan", "reading", 2, 5), None)
    result = RemoteResult(
        job_id,
        True,
        {"verdict": "ok"},
        (Artifact("report", "/cache/result/report.json", "report.json", 2, "a" * 64, "application/json"),),
    )

    assert decode_request(encode_message(request)) == request
    assert decode_status(encode_message(status)) == status
    assert decode_result(encode_message(result)) == result


def test_protocol_rejects_unknown_fields_truncation_and_noncanonical_uuid():
    job_id = uuid4()
    payload = encode_message(
        RemoteRequest(job_id, "scan", ("/work",), {}, "opaque-cancel-token-123456")
    )
    with pytest.raises(ValueError):
        decode_request(payload[:-1])
    with pytest.raises(ValueError, match="unknown"):
        decode_request(payload[:-1] + b',"extra":1}')
    with pytest.raises(ValueError, match="UUID"):
        decode_request(payload.replace(str(job_id).encode(), str(job_id).upper().encode()))


def test_status_cursor_accepts_duplicates_but_rejects_regression_and_bad_transition():
    job_id = uuid4()
    cursor = StatusCursor(job_id)
    queued = RemoteStatus(job_id, 0, RemoteTaskState.QUEUED, 1.0, None, None)
    running = RemoteStatus(job_id, 1, RemoteTaskState.RUNNING, 2.0, None, None)
    assert cursor.accept(queued)
    assert not cursor.accept(queued)
    assert cursor.accept(running)
    with pytest.raises(ValueError, match="sequence"):
        cursor.accept(queued)
    with pytest.raises(ValueError, match="transition"):
        cursor.accept(replace(running, sequence=2, state=RemoteTaskState.QUEUED))


def test_artifact_rejects_non_string_relative_path_as_protocol_error():
    with pytest.raises(ValueError, match="relative"):
        Artifact(
            "preview",
            "/cache/plot.png",
            None,
            1,
            "a" * 64,
            "image/png",
        )

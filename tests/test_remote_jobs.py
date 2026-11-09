from __future__ import annotations

from uuid import uuid4

from lyra.remote.jobs import RemoteJobClient
from lyra.remote.protocol import (
    RemoteResult,
    RemoteStatus,
    RemoteTaskState,
    decode_request,
    encode_message,
)
from lyra.remote.transport import RemoteDisconnected
from tests.test_remote_support import FakeSession


def test_submit_writes_request_then_starts_only_the_uuid_entrypoint():
    session = FakeSession()
    client = RemoteJobClient(session, "/home/user/.cache/lyra")

    job_id = client.submit("scan", ("/work/materials with spaces",), {"max_depth": 4})

    request_path = f"/home/user/.cache/lyra/jobs/{job_id}/request.json"
    request = decode_request(session.files[request_path])
    assert request.job_id == job_id
    assert request.source_paths == ("/work/materials with spaces",)
    assert session.started == [job_id]


def test_poll_ignores_duplicate_status_and_fetches_matching_terminal_result():
    session = FakeSession()
    client = RemoteJobClient(session, "/cache")
    job_id = client.submit("inspect", ("/calc",), {})
    status_path = f"/cache/jobs/{job_id}/status.json"
    result_path = f"/cache/jobs/{job_id}/result.json"
    running = RemoteStatus(job_id, 1, RemoteTaskState.RUNNING, 1.0, None, None)
    session.files[status_path] = encode_message(running)

    event = client.poll(job_id)
    assert event is not None and event.state is RemoteTaskState.RUNNING
    assert client.poll(job_id) is None

    result = RemoteResult(job_id, True, {"answer": 42}, ())
    session.files[status_path] = encode_message(
        RemoteStatus(job_id, 2, RemoteTaskState.SUCCEEDED, 2.0, None, 0)
    )
    session.files[result_path] = encode_message(result)
    event = client.poll(job_id)
    assert event is not None and event.result == result


def test_success_status_waits_for_atomic_result_publication_without_losing_terminal_event():
    session = FakeSession()
    client = RemoteJobClient(session, "/cache")
    job_id = client.submit("inspect", ("/calc",), {})
    status_path = f"/cache/jobs/{job_id}/status.json"
    result_path = f"/cache/jobs/{job_id}/result.json"
    session.files[status_path] = encode_message(
        RemoteStatus(job_id, 1, RemoteTaskState.SUCCEEDED, 2.0, None, 0)
    )

    assert client.poll(job_id) is None

    result = RemoteResult(job_id, True, {"published": True}, ())
    session.files[result_path] = encode_message(result)
    event = client.poll(job_id)
    assert event is not None
    assert event.state is RemoteTaskState.SUCCEEDED
    assert event.result == result


def test_disconnect_yields_unknown_without_fabricating_failure():
    session = FakeSession()
    client = RemoteJobClient(session, "/cache")
    job_id = client.submit("scan", ("/work",), {})
    session.read_error = RemoteDisconnected("network lost")

    event = client.poll(job_id)

    assert event is not None
    assert event.state is RemoteTaskState.UNKNOWN
    assert event.result is None


def test_cancel_uses_independent_control_session_and_token():
    data_session = FakeSession()
    control_session = FakeSession()
    client = RemoteJobClient(data_session, "/cache", control_session_factory=lambda: control_session)
    job_id = client.submit("scan", ("/work",), {})

    assert client.cancel(job_id)

    cancel_path = f"/cache/jobs/{job_id}/cancel.json"
    assert cancel_path in control_session.files
    assert str(job_id).encode() in control_session.files[cancel_path]
    assert b"cancel_token" in control_session.files[cancel_path]


def test_poll_discards_a_status_for_another_job():
    session = FakeSession()
    client = RemoteJobClient(session, "/cache")
    job_id = client.submit("scan", ("/work",), {})
    session.files[f"/cache/jobs/{job_id}/status.json"] = encode_message(
        RemoteStatus(uuid4(), 1, RemoteTaskState.RUNNING, 1.0, None, None)
    )
    assert client.poll(job_id) is None

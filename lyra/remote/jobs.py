from __future__ import annotations

import json
from pathlib import PurePosixPath
import secrets
from typing import Callable
from uuid import UUID, uuid4

from .protocol import (
    MAX_MESSAGE_BYTES,
    RemoteEvent,
    RemoteRequest,
    RemoteTaskState,
    StatusCursor,
    decode_result,
    decode_status,
    encode_message,
)
from .transport import RemoteDisconnected, RemoteSession, validate_remote_path


class RemoteJobClient:
    def __init__(
        self,
        session: RemoteSession,
        remote_cache_root: str,
        *,
        control_session_factory: Callable[[], RemoteSession] | None = None,
    ) -> None:
        self._session = session
        self._root = validate_remote_path(remote_cache_root, allow_root=False).rstrip("/")
        self._control_session_factory = control_session_factory
        self._cursors: dict[UUID, StatusCursor] = {}
        self._tokens: dict[UUID, str] = {}
        self._cancelled: set[UUID] = set()

    def _job_path(self, job_id: UUID, filename: str) -> str:
        if not isinstance(job_id, UUID):
            raise ValueError("Remote job ID must be a UUID.")
        return str(PurePosixPath(self._root) / "jobs" / str(job_id) / filename)

    def submit(self, task: str, source_paths: tuple[str, ...], options: dict) -> UUID:
        job_id = uuid4()
        token = secrets.token_urlsafe(24)
        request = RemoteRequest(job_id, task, tuple(source_paths), options, token)
        self._session.write_bytes_atomic(
            self._job_path(job_id, "request.json"),
            encode_message(request),
            mode=0o600,
        )
        self._cursors[job_id] = StatusCursor(job_id)
        self._tokens[job_id] = token
        try:
            self._session.start_job(job_id)
        except BaseException:
            self._cursors.pop(job_id, None)
            self._tokens.pop(job_id, None)
            raise
        return job_id

    def resume(self, job_id: UUID, *, cancel_token: str | None = None) -> None:
        self._cursors.setdefault(job_id, StatusCursor(job_id))
        if cancel_token is not None:
            self._tokens[job_id] = cancel_token

    def poll(self, job_id: UUID) -> RemoteEvent | None:
        cursor = self._cursors.get(job_id)
        if cursor is None:
            raise KeyError(f"Remote job is not active: {job_id}")
        try:
            raw = self._session.read_bytes(
                self._job_path(job_id, "status.json"),
                max_bytes=MAX_MESSAGE_BYTES,
            )
        except (FileNotFoundError, KeyError):
            return None
        except RemoteDisconnected as exc:
            return RemoteEvent(job_id, RemoteTaskState.UNKNOWN, message=str(exc))
        try:
            status = decode_status(raw)
        except ValueError:
            return None
        if status.job_id != job_id:
            return None
        if cursor.last == status:
            return None
        result = None
        if status.state is RemoteTaskState.SUCCEEDED:
            try:
                result = decode_result(
                    self._session.read_bytes(
                        self._job_path(job_id, "result.json"),
                        max_bytes=MAX_MESSAGE_BYTES,
                    )
                )
            except (FileNotFoundError, KeyError, ValueError):
                return None
            except RemoteDisconnected as exc:
                return RemoteEvent(job_id, RemoteTaskState.UNKNOWN, status=status, message=str(exc))
            if result.job_id != job_id:
                return None
        if not cursor.accept(status):
            return None
        return RemoteEvent(job_id, status.state, status=status, result=result)

    def cancel(self, job_id: UUID) -> bool:
        if job_id not in self._cursors or job_id in self._cancelled:
            return False
        token = self._tokens.get(job_id)
        if token is None:
            raise RuntimeError("Cancellation token is unavailable for the resumed job.")
        document = {
            "schema_version": 1,
            "job_id": str(job_id),
            "cancel_token": token,
        }
        data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        control = self._control_session_factory() if self._control_session_factory else self._session
        try:
            control.write_bytes_atomic(
                self._job_path(job_id, "cancel.json"),
                data,
                mode=0o600,
            )
        finally:
            if control is not self._session:
                control.close()
        self._cancelled.add(job_id)
        return True

    def close(self) -> None:
        self._session.close()

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping
from uuid import UUID

from .transport import validate_remote_path


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_ARTIFACTS = 4096
MAX_OPTIONS_DEPTH = 8
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TASKS = {
    "scan",
    "inspect",
    "plot",
    "band",
    "dos",
    "combined",
    "cache-build",
    "wavecar-evidence",
}


class RemoteTaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {self.CANCELLED, self.SUCCEEDED, self.FAILED}


def _canonical_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        raise ValueError("Job UUID must be a lowercase canonical UUID string.")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Job UUID is invalid.") from exc
    if value != str(parsed):
        raise ValueError("Job UUID must use lowercase canonical form.")
    return parsed


def _bounded_string(value: Any, field: str, maximum: int = 4096, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value) or len(value) > maximum:
        raise ValueError(f"{field} must be a bounded string.")
    if any(ord(character) < 32 and character not in "\t" for character in value):
        raise ValueError(f"{field} contains control characters.")
    return value


def _validate_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_OPTIONS_DEPTH:
        raise ValueError("JSON options exceed the nesting limit.")
    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str):
            return _bounded_string(value, "JSON string", 65536, empty=True)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        return value
    if isinstance(value, list | tuple):
        if len(value) > 10_000:
            raise ValueError("JSON array exceeds its item limit.")
        return [_validate_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 10_000:
            raise ValueError("JSON object exceeds its field limit.")
        result = {}
        for key, item in value.items():
            key = _bounded_string(key, "JSON field name", 256)
            result[key] = _validate_json_value(item, depth=depth + 1)
        return result
    raise ValueError("Value is not supported by the remote JSON protocol.")


@dataclass(frozen=True)
class RemoteRequest:
    job_id: UUID
    task: str
    source_paths: tuple[str, ...]
    options: Mapping[str, Any]
    cancel_token: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _canonical_uuid(self.job_id))
        if self.task not in _ALLOWED_TASKS:
            raise ValueError("Remote task is not in the fixed allowlist.")
        if not isinstance(self.source_paths, tuple) or not 1 <= len(self.source_paths) <= 32:
            raise ValueError("Remote request requires one through 32 source paths.")
        object.__setattr__(
            self,
            "source_paths",
            tuple(validate_remote_path(path, allow_root=False) for path in self.source_paths),
        )
        if not isinstance(self.options, Mapping):
            raise ValueError("Remote request options must be an object.")
        object.__setattr__(self, "options", _validate_json_value(dict(self.options)))
        if not isinstance(self.cancel_token, str) or not _TOKEN_PATTERN.fullmatch(self.cancel_token):
            raise ValueError("cancel_token must be an opaque URL-safe token.")


@dataclass(frozen=True)
class JobProgress:
    phase: str
    detail: str
    current: int
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _bounded_string(self.phase, "phase", 128))
        object.__setattr__(self, "detail", _bounded_string(self.detail, "detail", 4096, empty=True))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.current, self.total)):
            raise ValueError("Progress counts must be non-negative integers.")
        if self.total and self.current > self.total:
            raise ValueError("Progress current may not exceed total.")


@dataclass(frozen=True)
class RemoteStatus:
    job_id: UUID
    sequence: int
    state: RemoteTaskState
    timestamp: float
    progress: JobProgress | None
    exit_code: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _canonical_uuid(self.job_id))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("Status sequence must be a non-negative integer.")
        if not isinstance(self.state, RemoteTaskState):
            object.__setattr__(self, "state", RemoteTaskState(self.state))
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int | float) or not math.isfinite(self.timestamp):
            raise ValueError("Status timestamp must be finite.")
        if self.progress is not None and not isinstance(self.progress, JobProgress):
            raise ValueError("Status progress is invalid.")
        if self.exit_code is not None and (isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)):
            raise ValueError("Status exit_code must be an integer or null.")


@dataclass(frozen=True)
class Artifact:
    role: str
    remote_path: str
    relative_path: str
    size: int
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _bounded_string(self.role, "artifact role", 64))
        object.__setattr__(self, "remote_path", validate_remote_path(self.remote_path, allow_root=False))
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("Artifact relative path is unsafe.")
        relative = PurePosixPath(self.relative_path)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in self.relative_path
        ):
            raise ValueError("Artifact relative path is unsafe.")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("Artifact size must be a non-negative integer.")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("Artifact sha256 must be lowercase hexadecimal.")
        object.__setattr__(self, "media_type", _bounded_string(self.media_type, "media_type", 128))


@dataclass(frozen=True)
class RemoteResult:
    job_id: UUID
    success: bool
    payload: Mapping[str, Any]
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _canonical_uuid(self.job_id))
        if not isinstance(self.success, bool):
            raise ValueError("Result success must be boolean.")
        if not isinstance(self.payload, Mapping):
            raise ValueError("Result payload must be an object.")
        object.__setattr__(self, "payload", _validate_json_value(dict(self.payload)))
        if not isinstance(self.artifacts, tuple) or len(self.artifacts) > MAX_ARTIFACTS:
            raise ValueError("Result artifacts must be a bounded tuple.")
        if any(not isinstance(item, Artifact) for item in self.artifacts):
            raise ValueError("Result contains an invalid artifact.")
        if len({item.relative_path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("Result artifact relative paths must be unique.")


@dataclass(frozen=True)
class RemoteEvent:
    job_id: UUID
    state: RemoteTaskState
    status: RemoteStatus | None = None
    result: RemoteResult | None = None
    message: str | None = None


def _message_mapping(message: RemoteRequest | RemoteStatus | RemoteResult) -> dict[str, Any]:
    if isinstance(message, RemoteRequest):
        return {
            "schema_version": PROTOCOL_VERSION,
            "job_id": str(message.job_id),
            "task": message.task,
            "source_paths": list(message.source_paths),
            "options": message.options,
            "cancel_token": message.cancel_token,
        }
    if isinstance(message, RemoteStatus):
        progress = None
        if message.progress is not None:
            progress = {
                "phase": message.progress.phase,
                "detail": message.progress.detail,
                "current": message.progress.current,
                "total": message.progress.total,
            }
        return {
            "schema_version": PROTOCOL_VERSION,
            "job_id": str(message.job_id),
            "sequence": message.sequence,
            "state": message.state.value,
            "timestamp": message.timestamp,
            "progress": progress,
            "exit_code": message.exit_code,
        }
    if isinstance(message, RemoteResult):
        return {
            "schema_version": PROTOCOL_VERSION,
            "job_id": str(message.job_id),
            "success": message.success,
            "payload": message.payload,
            "artifacts": [
                {
                    "role": item.role,
                    "remote_path": item.remote_path,
                    "relative_path": item.relative_path,
                    "size": item.size,
                    "sha256": item.sha256,
                    "media_type": item.media_type,
                }
                for item in message.artifacts
            ],
        }
    raise TypeError("Unsupported remote protocol message.")


def encode_message(message: RemoteRequest | RemoteStatus | RemoteResult) -> bytes:
    data = json.dumps(
        _message_mapping(message),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Remote protocol message exceeds its size limit.")
    return data


def _decode_document(data: bytes, expected: set[str]) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Remote protocol payload is empty or exceeds its size limit.")
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Remote protocol payload is not complete UTF-8 JSON.") from exc
    if not isinstance(document, dict):
        raise ValueError("Remote protocol payload must be an object.")
    unknown = set(document) - expected
    missing = expected - set(document)
    if unknown or missing:
        raise ValueError(f"Remote protocol payload has unknown or missing fields: {sorted(unknown | missing)}")
    if document["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("Remote protocol schema version is unsupported.")
    return document


def decode_request(data: bytes) -> RemoteRequest:
    fields = {"schema_version", "job_id", "task", "source_paths", "options", "cancel_token"}
    document = _decode_document(data, fields)
    if not isinstance(document["source_paths"], list):
        raise ValueError("source_paths must be an array.")
    return RemoteRequest(
        _canonical_uuid(document["job_id"]),
        document["task"],
        tuple(document["source_paths"]),
        document["options"],
        document["cancel_token"],
    )


def decode_status(data: bytes) -> RemoteStatus:
    fields = {"schema_version", "job_id", "sequence", "state", "timestamp", "progress", "exit_code"}
    document = _decode_document(data, fields)
    raw_progress = document["progress"]
    progress = None
    if raw_progress is not None:
        if not isinstance(raw_progress, dict) or set(raw_progress) != {"phase", "detail", "current", "total"}:
            raise ValueError("Status progress has unknown or missing fields.")
        progress = JobProgress(**raw_progress)
    try:
        state = RemoteTaskState(document["state"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Remote task state is invalid.") from exc
    return RemoteStatus(
        _canonical_uuid(document["job_id"]),
        document["sequence"],
        state,
        document["timestamp"],
        progress,
        document["exit_code"],
    )


def decode_result(data: bytes) -> RemoteResult:
    fields = {"schema_version", "job_id", "success", "payload", "artifacts"}
    document = _decode_document(data, fields)
    raw_artifacts = document["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ValueError("Result artifacts must be an array.")
    expected = {"role", "remote_path", "relative_path", "size", "sha256", "media_type"}
    artifacts = []
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Result artifact has unknown or missing fields.")
        artifacts.append(Artifact(**raw))
    return RemoteResult(
        _canonical_uuid(document["job_id"]),
        document["success"],
        document["payload"],
        tuple(artifacts),
    )


_TRANSITIONS = {
    RemoteTaskState.QUEUED: {
        RemoteTaskState.RUNNING,
        RemoteTaskState.CANCELLING,
        RemoteTaskState.CANCELLED,
        RemoteTaskState.FAILED,
        RemoteTaskState.UNKNOWN,
    },
    RemoteTaskState.RUNNING: {
        RemoteTaskState.CANCELLING,
        RemoteTaskState.CANCELLED,
        RemoteTaskState.SUCCEEDED,
        RemoteTaskState.FAILED,
        RemoteTaskState.UNKNOWN,
    },
    RemoteTaskState.CANCELLING: {
        RemoteTaskState.CANCELLED,
        RemoteTaskState.SUCCEEDED,
        RemoteTaskState.FAILED,
        RemoteTaskState.UNKNOWN,
    },
    RemoteTaskState.UNKNOWN: {
        RemoteTaskState.RUNNING,
        RemoteTaskState.CANCELLING,
        RemoteTaskState.CANCELLED,
        RemoteTaskState.SUCCEEDED,
        RemoteTaskState.FAILED,
    },
    RemoteTaskState.CANCELLED: set(),
    RemoteTaskState.SUCCEEDED: set(),
    RemoteTaskState.FAILED: set(),
}


class StatusCursor:
    def __init__(self, job_id: UUID) -> None:
        self.job_id = _canonical_uuid(job_id)
        self.last: RemoteStatus | None = None

    def accept(self, status: RemoteStatus) -> bool:
        if status.job_id != self.job_id:
            raise ValueError("Status UUID does not match the active remote job.")
        if self.last is None:
            self.last = status
            return True
        if status.sequence < self.last.sequence:
            raise ValueError("Remote status sequence regressed.")
        if status.sequence == self.last.sequence:
            if status == self.last:
                return False
            raise ValueError("Remote status changed without advancing its sequence.")
        if status.state != self.last.state and status.state not in _TRANSITIONS[self.last.state]:
            raise ValueError(f"Invalid remote status transition: {self.last.state.value} -> {status.state.value}.")
        if self.last.state.terminal:
            raise ValueError("A terminal remote status cannot transition.")
        self.last = status
        return True

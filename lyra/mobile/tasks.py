from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import threading
import time
from typing import Any, Mapping, Protocol
import uuid


class TaskError(ValueError):
    pass


FORBIDDEN_FIELDS = {
    "path", "filepath", "file_path", "folder", "directory", "password",
    "passphrase", "private_key", "linux_key", "ssh_key", "secret", "credential",
}
VALID_STATES = {"queued", "running", "completed", "failed", "cancelling", "cancelled"}
KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")
PUBLIC_RESOURCE_ID = re.compile(r"^[A-Za-z0-9_-]{20,80}$")
PUBLIC_PARAMETER_FIELDS = {
    "directory_id",
    "band_directory_id",
    "dos_directory_id",
    "remote_profile_id",
    "energy_range",
    "density_range",
    "read_mode",
}


def _validate_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        raise TaskError("task parameters are too deeply nested")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > 4096:
            raise TaskError("task parameter is too long")
        return value
    if isinstance(value, list):
        if len(value) > 200:
            raise TaskError("task parameter list is too long")
        return [_validate_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TaskError("task parameter names must be strings")
            normalized = key.strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_FIELDS or normalized.endswith("_path"):
                raise TaskError(f"task field '{normalized}' is not permitted")
            clean[key] = _validate_value(item, depth=depth + 1)
        return clean
    raise TaskError("task parameters must contain JSON values only")


def validate_task_request(payload: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) - {"kind", "parameters"}:
        raise TaskError("task request must contain only kind and parameters")
    kind = payload.get("kind")
    if not isinstance(kind, str) or not KIND_PATTERN.fullmatch(kind):
        raise TaskError("invalid task kind")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise TaskError("task parameters must be an object")
    return kind, _validate_value(parameters)


@dataclass(frozen=True)
class MobileTask:
    task_id: str
    kind: str
    parameters: Mapping[str, Any]
    state: str = "queued"
    progress: float = 0.0
    stage: str = "Queued"
    message: str = ""
    resources: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_public_dict(self) -> dict[str, Any]:
        public_parameters = {
            key: value
            for key, value in self.parameters.items()
            if key.strip().lower().replace("-", "_") in PUBLIC_PARAMETER_FIELDS
        }
        public_resources = []
        for item in self.resources:
            resource_id = str(item.get("id", ""))
            if not PUBLIC_RESOURCE_ID.fullmatch(resource_id):
                continue
            label = re.split(r"[\\/]", str(item.get("label", "Lyra result")))[-1][:120]
            previewable = bool(item.get("previewable", False))
            resource = {
                "id": resource_id,
                "label": label or "Lyra result",
                "previewable": previewable,
                "preview_url": f"/api/resources/{resource_id}/preview" if previewable else None,
                "download_url": f"/api/resources/{resource_id}/download",
            }
            media_type = item.get("media_type")
            if isinstance(media_type, str) and re.fullmatch(r"[a-z]+/[a-z0-9.+-]+", media_type):
                resource["media_type"] = media_type
            public_resources.append(resource)
        return {
            "id": self.task_id,
            "kind": self.kind,
            "parameters": public_parameters,
            "state": self.state,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "resources": public_resources,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskBackend(Protocol):
    def list_tasks(self) -> list[MobileTask]: ...
    def get(self, task_id: str) -> MobileTask: ...
    def create(self, kind: str, parameters: Mapping[str, Any]) -> MobileTask: ...
    def cancel(self, task_id: str) -> MobileTask: ...


class InMemoryTaskBackend:
    """Reference backend and test double for the pluggable mobile API."""

    def __init__(self) -> None:
        self._tasks: dict[str, MobileTask] = {}
        self._lock = threading.RLock()

    def list_tasks(self) -> list[MobileTask]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, task_id: str) -> MobileTask:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise TaskError("task not found") from exc

    def create(self, kind: str, parameters: Mapping[str, Any]) -> MobileTask:
        kind, clean = validate_task_request({"kind": kind, "parameters": dict(parameters)})
        task = MobileTask(task_id=uuid.uuid4().hex, kind=kind, parameters=clean)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def update(self, task_id: str, **changes: Any) -> MobileTask:
        with self._lock:
            current = self.get(task_id)
            state = changes.get("state", current.state)
            if state not in VALID_STATES:
                raise TaskError("invalid task state")
            progress = float(changes.get("progress", current.progress))
            if not 0.0 <= progress <= 1.0:
                raise TaskError("task progress must be between zero and one")
            normalized = dict(changes)
            normalized["progress"] = progress
            normalized["updated_at"] = time.time()
            task = replace(current, **normalized)
            self._tasks[task_id] = task
            return task

    def cancel(self, task_id: str) -> MobileTask:
        current = self.get(task_id)
        if current.state in {"completed", "failed", "cancelled"}:
            return current
        return self.update(task_id, state="cancelled", stage="Cancelled")

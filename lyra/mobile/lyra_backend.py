from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Callable, Mapping

from lyra.models import (
    BandPlotSettings,
    CombinedPlotSettings,
    DosPlotSettings,
    GeneratedResult,
    PlotRange,
)
from lyra.workers import SpawnWorker

from .resources import ResourceRegistry
from .tasks import MobileTask, TaskError, validate_task_request


@dataclass(frozen=True)
class PublicDirectory:
    directory_id: str
    label: str

    def to_public_dict(self) -> dict[str, str]:
        return {"id": self.directory_id, "label": self.label}


@dataclass(frozen=True, repr=False)
class _AuthorizedDirectory:
    public: PublicDirectory
    path: Path


class AuthorizedDirectoryRegistry:
    """Maps desktop-approved directories to opaque mobile identifiers."""

    def __init__(self, *, state_file: Path | None = None) -> None:
        self._entries: dict[str, _AuthorizedDirectory] = {}
        self._lock = threading.RLock()
        self.state_file = Path(state_file).resolve() if state_file else None
        self._load()

    def authorize(self, path: Path, *, label: str | None = None) -> PublicDirectory:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_dir():
            raise TaskError("authorized target must be a directory")
        raw_label = " ".join((label or resolved.name).split())
        safe_label = raw_label.replace("\\", "/").rsplit("/", 1)[-1][:120] or "Lyra calculation"
        with self._lock:
            for entry in self._entries.values():
                if entry.path == resolved:
                    return entry.public
            directory_id = secrets.token_urlsafe(24)
            public = PublicDirectory(directory_id, safe_label)
            self._entries[directory_id] = _AuthorizedDirectory(public, resolved)
            self._save()
            return public

    def list_public(self) -> tuple[PublicDirectory, ...]:
        with self._lock:
            return tuple(entry.public for entry in self._entries.values())

    def resolve(self, directory_id: str) -> Path:
        if not isinstance(directory_id, str) or len(directory_id) > 80:
            raise TaskError("calculation directory is not authorized")
        with self._lock:
            entry = self._entries.get(directory_id)
        if entry is None or not entry.path.is_dir():
            raise TaskError("calculation directory is not authorized")
        return entry.path

    def revoke(self, directory_id: str) -> bool:
        with self._lock:
            removed = self._entries.pop(directory_id, None) is not None
            if removed:
                self._save()
            return removed

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            if self.state_file.stat().st_size > 1024 * 1024:
                raise ValueError("authorized directory state is too large")
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(payload.get("directories"), list):
                raise ValueError("unsupported authorized directory state")
            for item in payload["directories"]:
                directory_id = str(item["id"])
                if not re.fullmatch(r"[A-Za-z0-9_-]{20,80}", directory_id):
                    raise ValueError("invalid authorized directory identifier")
                path = Path(str(item["path"])).resolve(strict=True)
                if not path.is_dir():
                    continue
                label = " ".join(str(item["label"]).split())[:120]
                public = PublicDirectory(directory_id, label or path.name)
                self._entries[directory_id] = _AuthorizedDirectory(public, path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._entries.clear()

    def _save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "directories": [
                {
                    "id": entry.public.directory_id,
                    "label": entry.public.label,
                    "path": str(entry.path),
                }
                for entry in self._entries.values()
            ],
        }
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_file)


class LyraTaskBackend:
    """Queues mobile preview jobs on Lyra's existing SpawnWorker."""

    _ALLOWED_PARAMETERS = {
        "directory_id",
        "band_directory_id",
        "dos_directory_id",
        "energy_range",
        "density_range",
        "read_mode",
        "projection",
        "projection_channels",
        "band_projection_channels",
        "dos_projection_channels",
    }
    _PROJECTION_PARAMETERS = {
        "projection",
        "projection_channels",
        "band_projection_channels",
        "dos_projection_channels",
    }
    _PHASE_PROGRESS = {
        "metadata": 0.08,
        "source-parse": 0.12,
        "parse": 0.30,
        "projection-parse": 0.34,
        "cache-read": 0.52,
        "memory-hit": 0.58,
        "cache-write": 0.68,
        "complete": 0.78,
        "preview-render": 0.90,
        "export": 0.90,
    }
    _MAX_ACTIVE_TASKS = 10

    def __init__(
        self,
        *,
        directories: AuthorizedDirectoryRegistry,
        resources: ResourceRegistry,
        worker_factory: Callable[[], Any] = SpawnWorker,
        poll_interval: float = 0.10,
        autostart: bool = True,
        state_file: Path | None = None,
    ) -> None:
        self.directories = directories
        self.resources = resources
        self._worker = worker_factory()
        self._poll_interval = max(0.02, float(poll_interval))
        self._tasks: dict[str, MobileTask] = {}
        self._private_payloads: dict[str, tuple[str, dict[str, Any]]] = {}
        self._active_task_id: str | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.state_file = Path(state_file).resolve() if state_file else None
        self._thread: threading.Thread | None = None
        self._load_tasks()
        if autostart:
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="LyraMobileTasks",
                daemon=True,
            )
            self._thread.start()

    def capabilities(self) -> dict[str, Any]:
        return {
            "task_kinds": ["band", "dos", "combined"],
            "read_modes": ["lite", "full"],
            "projection": {
                "supported": False,
                "reason": "Mobile preview currently supports standard non-projected plots; configure projected plots in the desktop Lyra interface.",
            },
            "directories": [item.to_public_dict() for item in self.directories.list_public()],
        }

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
        if kind not in {"band", "dos", "combined"}:
            raise TaskError("mobile Lyra backend supports band, dos, and combined previews")
        unknown = set(clean) - self._ALLOWED_PARAMETERS
        if unknown:
            raise TaskError("task contains unsupported parameters")
        if any(clean.get(name) not in (None, False, [], (), {}) for name in self._PROJECTION_PARAMETERS):
            raise TaskError("projected mobile previews are not available; use desktop Lyra")

        operation, private_payload, public_parameters = self._build_payload(kind, clean)
        task = MobileTask(
            task_id=secrets.token_hex(16),
            kind=kind,
            parameters=public_parameters,
            state="queued",
            progress=0.0,
            stage="Queued",
        )
        with self._lock:
            active_count = sum(
                item.state in {"queued", "running", "cancelling"}
                for item in self._tasks.values()
            )
            if active_count >= self._MAX_ACTIVE_TASKS:
                raise TaskError("mobile task queue is full")
            terminal = sorted(
                (
                    item
                    for item in self._tasks.values()
                    if item.state in {"completed", "failed", "cancelled"}
                ),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            for stale in terminal[200:]:
                self._tasks.pop(stale.task_id, None)
                self._private_payloads.pop(stale.task_id, None)
            self._tasks[task.task_id] = task
            self._private_payloads[task.task_id] = (operation, private_payload)
            self._save_tasks()
        return task

    def cancel(self, task_id: str) -> MobileTask:
        with self._lock:
            task = self.get(task_id)
            if task.state == "queued":
                return self._replace_task(task_id, state="cancelled", stage="Cancelled")
            if task.state != "running":
                return task
            if task_id != self._active_task_id or not self._worker.cancel():
                return task
            return self._replace_task(task_id, state="cancelling", stage="Cancelling")

    def poll_once(self) -> None:
        with self._lock:
            if self._active_task_id is None:
                queued = sorted(
                    (task for task in self._tasks.values() if task.state == "queued"),
                    key=lambda item: item.created_at,
                )
                if queued:
                    self._start_task(queued[0].task_id)
            active_id = self._active_task_id
        if active_id is None:
            return
        message = self._worker.poll()
        if message is None:
            return
        kind = message.get("kind") if isinstance(message, dict) else None
        payload = message.get("payload") if isinstance(message, dict) else None
        with self._lock:
            if active_id != self._active_task_id:
                return
            if kind == "progress" and isinstance(payload, dict):
                self._apply_progress(active_id, payload)
            elif kind == "success":
                self._apply_success(active_id, payload)
                self._finish_active()
            elif kind == "cancelled":
                self._replace_task(active_id, state="cancelled", stage="Cancelled")
                self._finish_active()
            elif kind == "error":
                self._replace_task(
                    active_id,
                    state="failed",
                    stage="Failed",
                    message="Lyra task failed. Check the desktop log for details.",
                )
                self._finish_active()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._active_task_id is not None:
                self._worker.cancel()
        self._worker.close()

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                self.poll_once()
            except Exception:
                with self._lock:
                    if self._active_task_id:
                        self._replace_task(
                            self._active_task_id,
                            state="failed",
                            stage="Failed",
                            message="Lyra task failed. Check the desktop log for details.",
                        )
                        self._finish_active()

    def _start_task(self, task_id: str) -> None:
        operation, payload = self._private_payloads[task_id]
        try:
            self._worker.start(operation, payload)
        except Exception:
            self._replace_task(
                task_id,
                state="failed",
                stage="Failed to start",
                message="Lyra could not start this task. Check the desktop log for details.",
            )
            self._private_payloads.pop(task_id, None)
            return
        self._active_task_id = task_id
        self._replace_task(task_id, state="running", progress=0.01, stage="Starting Lyra worker")

    def _apply_progress(self, task_id: str, event: Mapping[str, Any]) -> None:
        current_task = self.get(task_id)
        phase = str(event.get("phase") or "")
        base = self._PHASE_PROGRESS.get(phase, current_task.progress)
        try:
            current = float(event.get("current", 0))
            total = float(event.get("total", 0))
            fraction = max(0.0, min(1.0, current / total)) if total > 0 else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            fraction = 0.0
        next_phase_values = sorted(value for value in self._PHASE_PROGRESS.values() if value > base)
        ceiling = next_phase_values[0] if next_phase_values else 0.97
        progress = min(0.97, max(current_task.progress, base + (ceiling - base) * fraction))
        detail = str(event.get("detail") or phase or "Working")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        self._replace_task(task_id, progress=progress, stage=detail)

    def _apply_success(self, task_id: str, payload: Any) -> None:
        if not isinstance(payload, GeneratedResult):
            self._replace_task(
                task_id,
                state="failed",
                stage="Failed",
                message="Lyra returned an invalid result.",
            )
            return
        try:
            public = self.resources.register(
                payload.preview_file,
                label=Path(payload.preview_file).name,
            )
        except (OSError, ValueError):
            self._replace_task(
                task_id,
                state="failed",
                stage="Preview unavailable",
                message="The preview could not be registered for mobile access.",
            )
            return
        self._replace_task(
            task_id,
            state="completed",
            progress=1.0,
            stage="Completed",
            message="Preview ready",
            resources=(public.to_public_dict(),),
        )

    def _finish_active(self) -> None:
        if self._active_task_id is not None:
            self._private_payloads.pop(self._active_task_id, None)
        self._active_task_id = None

    def _replace_task(self, task_id: str, **changes: Any) -> MobileTask:
        task = replace(self._tasks[task_id], **changes, updated_at=time.time())
        self._tasks[task_id] = task
        self._save_tasks()
        return task

    def _load_tasks(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            if self.state_file.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("mobile task state is too large")
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(payload.get("tasks"), list):
                raise ValueError("unsupported mobile task state")
            for item in payload["tasks"][-200:]:
                task_id = str(item["id"])
                if not re.fullmatch(r"[a-f0-9]{32}", task_id):
                    raise ValueError("invalid mobile task identifier")
                kind, parameters = validate_task_request({
                    "kind": str(item["kind"]),
                    "parameters": dict(item.get("parameters", {})),
                })
                if kind not in {"band", "dos", "combined"}:
                    raise ValueError("invalid persisted mobile task kind")
                if set(parameters) - self._ALLOWED_PARAMETERS:
                    raise ValueError("invalid persisted mobile task parameters")
                state = str(item["state"])
                message = str(item.get("message", ""))[:300]
                stage = str(item.get("stage", ""))[:180]
                if state in {"queued", "running", "cancelling"}:
                    state = "failed"
                    stage = "Interrupted"
                    message = "Task was interrupted by a Lyra service restart."
                progress = float(item.get("progress", 0.0))
                created_at = float(item["created_at"])
                updated_at = float(item["updated_at"])
                if (
                    not math.isfinite(progress)
                    or not 0.0 <= progress <= 1.0
                    or not math.isfinite(created_at)
                    or not math.isfinite(updated_at)
                    or created_at < 0
                    or updated_at < 0
                ):
                    raise ValueError("invalid persisted mobile task values")
                raw_resources = item.get("resources", ())
                if not isinstance(raw_resources, (list, tuple)) or len(raw_resources) > 20:
                    raise ValueError("invalid persisted mobile task resources")
                task = MobileTask(
                    task_id=task_id,
                    kind=kind,
                    parameters=parameters,
                    state=state,
                    progress=progress,
                    stage=stage,
                    message=message,
                    resources=tuple(raw_resources),
                    created_at=created_at,
                    updated_at=updated_at,
                )
                public = task.to_public_dict()
                if public["id"] != task.task_id or state not in {
                    "completed", "failed", "cancelled"
                }:
                    raise ValueError("invalid persisted mobile task")
                self._tasks[task.task_id] = task
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._tasks.clear()
        self._save_tasks()

    def _save_tasks(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tasks = sorted(self._tasks.values(), key=lambda item: item.created_at)[-200:]
        payload = {
            "version": 1,
            "tasks": [task.to_public_dict() for task in tasks],
        }
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_file)

    def _build_payload(
        self, kind: str, parameters: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        energy_range = self._plot_range(parameters.get("energy_range"), default=(-5.0, 5.0))
        read_mode = str(parameters.get("read_mode", "lite")).lower()
        if read_mode not in {"lite", "full"}:
            raise TaskError("read_mode must be lite or full")

        if kind in {"band", "dos"}:
            directory_id = parameters.get("directory_id")
            folder = self.directories.resolve(directory_id)
            public = {
                "directory_id": directory_id,
                "energy_range": [energy_range.minimum, energy_range.maximum],
                "read_mode": read_mode,
            }
            if kind == "band":
                settings = BandPlotSettings(energy_range=energy_range, mode="standard")
            else:
                density_range = self._optional_plot_range(parameters.get("density_range"))
                settings = DosPlotSettings(
                    energy_range=energy_range,
                    density_range=density_range,
                    projection_channels=(),
                )
                if density_range is not None:
                    public["density_range"] = [density_range.minimum, density_range.maximum]
            return f"preview-{kind}", {
                "mode": read_mode,
                "folder": str(folder),
                "settings": settings,
            }, public

        common_id = parameters.get("directory_id")
        band_id = parameters.get("band_directory_id", common_id)
        dos_id = parameters.get("dos_directory_id", common_id)
        band_folder = self.directories.resolve(band_id)
        dos_folder = self.directories.resolve(dos_id)
        density_range = self._optional_plot_range(parameters.get("density_range"))
        settings = CombinedPlotSettings(
            energy_range=energy_range,
            band_mode="standard",
            band_projection_channels=(),
            dos_projection_channels=(),
            dos_density_range=density_range,
        )
        public = {
            "band_directory_id": band_id,
            "dos_directory_id": dos_id,
            "energy_range": [energy_range.minimum, energy_range.maximum],
            "read_mode": read_mode,
        }
        if density_range is not None:
            public["density_range"] = [density_range.minimum, density_range.maximum]
        return "preview-combined", {
            "mode": read_mode,
            "band_mode": read_mode,
            "dos_mode": read_mode,
            "band_folder": str(band_folder),
            "dos_folder": str(dos_folder),
            "settings": settings,
        }, public

    @staticmethod
    def _plot_range(value: Any, *, default: tuple[float, float]) -> PlotRange:
        if value is None:
            minimum, maximum = default
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                minimum, maximum = float(value[0]), float(value[1])
            except (TypeError, ValueError) as exc:
                raise TaskError("plot range must contain two numbers") from exc
        else:
            raise TaskError("plot range must contain two numbers")
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise TaskError("plot range minimum must be smaller than maximum")
        return PlotRange(minimum, maximum)

    @classmethod
    def _optional_plot_range(cls, value: Any) -> PlotRange | None:
        return None if value is None else cls._plot_range(value, default=(0.0, 1.0))

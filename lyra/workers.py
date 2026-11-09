from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.reduction import ForkingPickler
from queue import Empty
import os
from pathlib import Path
import shutil
import time
import uuid

from lyra.job_handlers import JobCancelled, execute_job as _execute_job
from lyra.memory_policy import MemoryLeasePool


_DEFAULT_MEMORY_POOL = MemoryLeasePool()
_TERMINAL_KINDS = {"success", "cache-cleared", "cancelled", "error"}


def _envelope(job_id: uuid.UUID, kind: str, payload) -> dict:
    return {"job_id": job_id, "kind": kind, "payload": payload}


def _worker_loop(command_queue, result_queue, cancel_event) -> None:
    from lyra.data_service import ProjectionDataService

    service = ProjectionDataService()
    while True:
        command = command_queue.get()
        job_id = command["job_id"]
        kind = command["kind"]
        payload = command["payload"]
        if kind == "shutdown":
            return
        operation = kind
        try:
            result = _execute_job(
                service,
                result_queue,
                operation,
                payload,
                job_id=job_id,
                cancel_event=cancel_event,
            )
            message_kind = "cache-cleared" if operation == "clear-cache" else "success"
            result_queue.put(_envelope(job_id, message_kind, result))
        except JobCancelled:
            result_queue.put(_envelope(job_id, "cancelled", None))
        except Exception as exc:
            result_queue.put(
                _envelope(job_id, "error", f"{type(exc).__name__}: {exc}")
            )


class SpawnWorker:
    def __init__(
        self,
        *,
        memory_pool: MemoryLeasePool | None = None,
        clock=time.monotonic,
        cancel_timeout: float = 5.0,
    ) -> None:
        self.context = get_context("spawn")
        self._memory_pool = memory_pool or _DEFAULT_MEMORY_POOL
        self._clock = clock
        self._cancel_timeout = cancel_timeout
        self.command_queue, self.queue, self.cancel_event = self._create_resources()
        self._resources_need_replacement = False
        self.process = None
        self._state = "idle"
        self._active_job_id: uuid.UUID | None = None
        self._active_operation: str | None = None
        self._active_payload: dict | None = None
        self._active_lease = None
        self._cancel_deadline: float | None = None
        self._abandoned_publication: tuple[Path, Path] | None = None
        self._abandoned_publication_ready = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def running(self) -> bool:
        return self._state in {"running", "cancelling"}

    def start(self, operation: str, payload: dict) -> uuid.UUID:
        if self.running:
            raise RuntimeError("A Lyra plotting task is already running.")
        job_id = uuid.uuid4()
        command = _envelope(job_id, operation, payload)
        ForkingPickler.dumps(command)
        slots = 2 if "combined" in operation or "paired-band" in operation else 1
        lease = self._memory_pool.acquire(operation, slots)
        process_ready = False
        try:
            if self._resources_need_replacement:
                self._replace_resources()
            self._ensure_process()
            process_ready = True
            self.cancel_event.clear()
            self._active_job_id = job_id
            self._active_operation = operation
            self._active_payload = payload
            self._active_lease = lease
            self._state = "running"
            self.command_queue.put(command)
        except BaseException:
            if process_ready or self.process is not None:
                try:
                    self._retire_process(force=True, replace_resources=True)
                except BaseException:
                    self._resources_need_replacement = True
            if self._active_lease is lease:
                self._finish_active_job()
            else:
                lease.close()
            raise
        return job_id

    def cancel(self) -> bool:
        if self._state != "running" or self._active_operation == "clear-cache":
            return False
        self._state = "cancelling"
        self.cancel_event.set()
        self._cancel_deadline = self._clock() + self._cancel_timeout
        return True

    def poll(self):
        if self.running and self.process is not None and not self._process_is_alive(
            self.process
        ):
            kind = "cancelled" if self._state == "cancelling" else "error"
            payload = (
                None
                if kind == "cancelled"
                else "RuntimeError: Lyra plotting worker exited unexpectedly."
            )
            return self._finalize_exited_process(kind, payload)
        if (
            self._state == "cancelling"
            and self._cancel_deadline is not None
            and self._clock() >= self._cancel_deadline
        ):
            return self._finalize_exited_process("cancelled", None, force=True)

        while True:
            try:
                message = self.queue.get_nowait()
            except Empty:
                break
            if not isinstance(message, dict) or set(message) != {
                "job_id",
                "kind",
                "payload",
            }:
                continue
            if message["job_id"] != self._active_job_id:
                continue
            if message["kind"] in _TERMINAL_KINDS:
                self._finish_active_job()
            return message

        return None

    def _finalize_exited_process(self, kind: str, payload, *, force: bool = False):
        job_id = self._active_job_id
        try:
            self._retire_process(force=force, replace_resources=True)
        except BaseException:
            self._resources_need_replacement = True
        finally:
            if self.process is None:
                self._finish_active_job()
        if self.process is not None:
            return None
        return _envelope(job_id, kind, payload)

    def _finish_active_job(self) -> None:
        lease = self._active_lease
        self._active_lease = None
        self._active_job_id = None
        self._active_operation = None
        self._active_payload = None
        self._cancel_deadline = None
        self._state = "idle"
        if lease is not None:
            lease.close()

    def _ensure_process(self) -> None:
        if self._resources_need_replacement:
            self._replace_resources()
        if self.process is not None and self._process_is_alive(self.process):
            return
        if self.process is not None:
            self._retire_process(replace_resources=True)
        process = self.context.Process(
            target=_worker_loop,
            args=(self.command_queue, self.queue, self.cancel_event),
            daemon=True,
        )
        self.process = process
        try:
            process.start()
        except BaseException:
            try:
                self._stop_process(process, force=True)
            except BaseException:
                raise
            else:
                self.process = None
            raise

    def _retire_process(
        self,
        *,
        force: bool = False,
        replace_resources: bool = False,
    ) -> None:
        process = self.process
        errors: list[BaseException] = []
        if process is not None:
            self._stop_process(process, force=force)
            self.process = None
            for cleanup in (
                self._cleanup_abandoned_render,
                self._cleanup_incomplete_rebuild,
                self._finalize_abandoned_publication,
            ):
                try:
                    cleanup()
                except BaseException as exc:
                    errors.append(exc)
        if replace_resources:
            try:
                self._replace_resources()
            except BaseException as exc:
                self._resources_need_replacement = True
                errors.append(exc)
        if errors:
            raise errors[0]

    def _stop_process(self, process, *, force: bool) -> None:
        if force and self._process_is_alive(process):
            process.terminate()
        try:
            process.join(timeout=1.0 if force else 0.2)
        except (AttributeError, AssertionError, ValueError):
            pass
        if self._process_is_alive(process):
            process.terminate()
            process.join(timeout=1.0)
        if self._process_is_alive(process):
            raise RuntimeError("Lyra plotting worker could not be stopped.")

    @staticmethod
    def _process_is_alive(process) -> bool:
        try:
            return bool(process.is_alive())
        except (AttributeError, AssertionError, ValueError):
            return False

    def _create_resources(self):
        command_queue = None
        result_queue = None
        try:
            command_queue = self.context.SimpleQueue()
            result_queue = self.context.Queue()
            cancel_event = self.context.Event()
            return command_queue, result_queue, cancel_event
        except BaseException:
            if command_queue is not None:
                self._close_queue(command_queue)
            if result_queue is not None:
                self._close_queue(result_queue)
            raise

    def _replace_resources(self) -> None:
        old_command_queue = self.command_queue
        old_result_queue = self.queue
        command_queue, result_queue, cancel_event = self._create_resources()
        self.command_queue = command_queue
        self.queue = result_queue
        self.cancel_event = cancel_event
        self._resources_need_replacement = False
        self._close_queue(old_command_queue)
        self._close_queue(old_result_queue)

    def _cleanup_incomplete_rebuild(self) -> None:
        operation = self._active_operation or ""
        payload = self._active_payload or {}
        if not operation.startswith("rebuild-preview-"):
            return
        from lyra.cache import ProjectionCache

        plot_kind = operation.removeprefix("rebuild-preview-")
        if plot_kind == "combined":
            requested = payload.get("rebuild_data_kind")
            entries = (
                ((payload.get("band_folder"), "band"), (payload.get("dos_folder"), "dos"))
                if requested is None
                else ((payload.get(f"{requested}_folder"), requested),)
            )
        else:
            entries = ((payload.get("folder"), plot_kind),)
        cache = ProjectionCache()
        commit = self._abandoned_publication is not None
        try:
            for folder, data_kind in entries:
                if folder and data_kind in {"band", "dos", "fermi"}:
                    if commit:
                        cache.finish_rebuild(folder, data_kind, None, success=True)
                    else:
                        cache.cleanup_incomplete(folder, data_kind)
        except BaseException:
            if commit:
                self._abandoned_publication_ready = False
            raise

    def _cleanup_abandoned_render(self) -> None:
        self._abandoned_publication = None
        self._abandoned_publication_ready = False
        operation = self._active_operation or ""
        payload = self._active_payload or {}
        job_id = self._active_job_id
        if job_id is None:
            return
        plot_kind = operation
        intent = "export"
        for prefix, candidate_intent in (
            ("rebuild-preview-", "preview"),
            ("preview-", "preview"),
            ("export-", "export"),
        ):
            if operation.startswith(prefix):
                plot_kind = operation.removeprefix(prefix)
                intent = candidate_intent
                break
        if plot_kind not in {"band", "dos", "fermi", "combined"}:
            return
        if intent == "preview":
            folder = payload.get("band_folder") if plot_kind == "combined" else payload.get("folder")
            if not folder:
                return
            destination = Path(folder).resolve() / ".lyra-cache" / "previews"
        else:
            output = payload.get("output")
            if not output:
                return
            destination = Path(output)
        staging = destination / f".lyra-job-{job_id}"
        from lyra.job_handlers import (
            _publication_committed,
            _rollback_staged_publication,
        )

        if _publication_committed(staging):
            self._abandoned_publication = (staging, destination)
            self._abandoned_publication_ready = True
            return

        if _rollback_staged_publication(
            staging,
            destination,
            replace_file=os.replace,
        ):
            shutil.rmtree(staging, ignore_errors=True)

    def _finalize_abandoned_publication(self) -> None:
        if (
            self._abandoned_publication is None
            or not self._abandoned_publication_ready
        ):
            return
        staging, _destination = self._abandoned_publication
        from lyra.job_handlers import _discard_publication_artifacts

        _discard_publication_artifacts(staging)
        shutil.rmtree(staging, ignore_errors=True)
        self._abandoned_publication = None
        self._abandoned_publication_ready = False

    def close(self) -> None:
        process = self.process
        if process is not None and self._process_is_alive(process) and not self.running:
            try:
                self.command_queue.put(_envelope(uuid.uuid4(), "shutdown", None))
                process.join(timeout=2.0)
            except (OSError, ValueError):
                pass
        if process is not None:
            try:
                self._retire_process(
                    force=self._process_is_alive(process)
                )
            except BaseException:
                pass
        if self.process is None:
            self._finish_active_job()
        self._close_queue(self.command_queue)
        self._close_queue(self.queue)

    @staticmethod
    def _close_queue(queue) -> None:
        try:
            queue.close()
            queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass

import tempfile
import time
import unittest
import uuid
import json
import pickle
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import Mock, patch

from lyra.cache import CacheIdentity, CacheSource
from lyra.models import (
    BandPlotSettings,
    CombinedPlotSettings,
    DosPlotSettings,
    ExportSettings,
    FermiPlotSettings,
    GeneratedResult,
    BandStateSelection,
    PlotRange,
    ProjectionChannel,
)
from lyra.memory_policy import MemoryBudgetExceeded, MemoryLeasePool
from lyra.workers import SpawnWorker, _execute_job
from tests.fixtures import write_band_fixture, write_fermi_fixture, write_procar


class WorkerTests(unittest.TestCase):
    def test_command_envelope_is_exact_and_queue_publication_failure_releases_lease(self):
        class RecordingQueue:
            def __init__(self, error=None):
                self.error = error
                self.messages = []

            def put(self, message):
                if self.error is not None:
                    raise self.error
                self.messages.append(message)

            def close(self):
                pass

            def cancel_join_thread(self):
                pass

        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        original_queue = worker.command_queue
        original_queue.close()
        worker.command_queue = RecordingQueue()
        worker._ensure_process = lambda: None
        try:
            job_id = worker.start("preview-band", {"value": 1})
            command = worker.command_queue.messages[0]
            self.assertEqual(set(command), {"job_id", "kind", "payload"})
            self.assertEqual(command, {
                "job_id": job_id,
                "kind": "preview-band",
                "payload": {"value": 1},
            })
            worker._finish_active_job()

            worker.command_queue = RecordingQueue(OSError("publication failed"))
            with self.assertRaisesRegex(OSError, "publication failed"):
                worker.start("preview-band", {})
            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
        finally:
            worker.close()

    def test_process_start_failure_releases_lease_and_leaves_worker_idle(self):
        class UnstartableProcess:
            def start(self):
                raise OSError("spawn failed")

        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker.context = Mock()
        worker.context.Process.return_value = UnstartableProcess()
        try:
            with self.assertRaisesRegex(OSError, "spawn failed"):
                worker.start("preview-band", {})

            self.assertEqual(pool.used_slots, 0)
            self.assertEqual(worker.state, "idle")
            self.assertIsNone(worker.process)
        finally:
            worker.close()

    def test_late_process_start_failure_retires_partial_child(self):
        class PartialProcess:
            def __init__(self):
                self.alive = False
                self.terminated = False
                self.joined = False

            def start(self):
                self.alive = True
                raise OSError("late spawn failure")

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def join(self, timeout=None):
                self.joined = True

        process = PartialProcess()
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker.context = Mock()
        worker.context.Process.return_value = process
        try:
            with self.assertRaisesRegex(OSError, "late spawn failure"):
                worker.start("preview-band", {})

            self.assertTrue(process.terminated)
            self.assertTrue(process.joined)
            self.assertIsNone(worker.process)
            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
        finally:
            worker.close()

    def test_event_reset_failure_retires_ready_child_and_releases_lease(self):
        class ReadyProcess:
            def __init__(self):
                self.alive = True
                self.terminated = False

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def join(self, timeout=None):
                pass

        process = ReadyProcess()
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: setattr(worker, "process", process)
        worker.cancel_event = Mock()
        worker.cancel_event.clear.side_effect = OSError("event reset failed")
        try:
            with self.assertRaisesRegex(OSError, "event reset failed"):
                worker.start("preview-band", {})

            self.assertTrue(process.terminated)
            self.assertIsNone(worker.process)
            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
        finally:
            worker.close()

    def test_real_unpickleable_payload_is_rejected_synchronously(self):
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        try:
            with self.assertRaises((AttributeError, TypeError, pickle.PicklingError)):
                worker.start("preview-band", {"callback": lambda: None})

            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
            self.assertIsNone(worker.process)
        finally:
            worker.close()

    def test_poll_ignores_stale_envelopes_before_state_or_lease_changes(self):
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.queue.close()
        worker.command_queue = Queue()
        worker.queue = Queue()
        try:
            active_id = worker.start("preview-band", {})
            for kind in ("progress", "success", "error", "cancelled"):
                worker.queue.put({"job_id": uuid.uuid4(), "kind": kind, "payload": "stale"})
            worker.queue.put({"job_id": active_id, "kind": "progress", "payload": {"phase": "parse"}})

            message = worker.poll()

            self.assertEqual(message["job_id"], active_id)
            self.assertEqual(message["kind"], "progress")
            self.assertEqual(worker.state, "running")
            self.assertEqual(pool.used_slots, 1)
        finally:
            worker.close()

    def test_child_crash_replaces_resources_and_releases_lease_after_join(self):
        class CrashedProcess:
            def __init__(self):
                self.joined = False

            def is_alive(self):
                return False

            def join(self, timeout=None):
                self.joined = True

        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.queue.close()
        worker.command_queue = Queue()
        worker.queue = Queue()
        job_id = worker.start("preview-band", {})
        crashed = CrashedProcess()
        worker.process = crashed

        message = worker.poll()

        self.assertEqual(message["job_id"], job_id)
        self.assertEqual(message["kind"], "error")
        self.assertTrue(crashed.joined)
        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(worker.state, "idle")
        worker.close()

    def test_close_while_cancelling_terminates_child_before_releasing_lease(self):
        class BusyProcess:
            def __init__(self):
                self.alive = True
                self.terminated = False
                self.joined = False

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def join(self, timeout=None):
                self.joined = True

        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.command_queue = Queue()
        worker.start("preview-band", {})
        process = BusyProcess()
        worker.process = process
        worker.cancel()

        worker.close()

        self.assertTrue(process.terminated)
        self.assertTrue(process.joined)
        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(worker.state, "idle")

    def test_forced_timeout_replaces_queues_and_event_without_waiting_wall_clock(self):
        now = [10.0]
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool, clock=lambda: now[0], cancel_timeout=5.0)
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.queue.close()
        worker.command_queue = Queue()
        worker.queue = Queue()
        old_command = worker.command_queue
        old_results = worker.queue
        old_event = worker.cancel_event
        try:
            job_id = worker.start("preview-band", {})
            self.assertTrue(worker.cancel())
            now[0] = 15.0

            message = worker.poll()

            self.assertEqual(message, {"job_id": job_id, "kind": "cancelled", "payload": None})
            self.assertIsNot(worker.command_queue, old_command)
            self.assertIsNot(worker.queue, old_results)
            self.assertIsNot(worker.cancel_event, old_event)
            self.assertEqual(pool.used_slots, 0)
            self.assertEqual(worker.state, "idle")
        finally:
            worker.close()

    def test_expired_deadline_preempts_matching_progress_backlog(self):
        class BusyProcess:
            def __init__(self):
                self.alive = True
                self.terminate_calls = 0

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminate_calls += 1
                self.alive = False

            def join(self, timeout=None):
                pass

        now = [0.0]
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool, clock=lambda: now[0])
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.queue.close()
        worker.command_queue = Queue()
        worker.queue = Queue()
        try:
            job_id = worker.start("preview-band", {})
            process = BusyProcess()
            worker.process = process
            worker.cancel()
            worker.queue.put(
                {"job_id": job_id, "kind": "progress", "payload": {"phase": "parse"}}
            )
            now[0] = 5.0

            message = worker.poll()

            self.assertEqual(message, {"job_id": job_id, "kind": "cancelled", "payload": None})
            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
        finally:
            worker.close()

    def test_dead_child_preempts_matching_progress_backlog(self):
        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: None
        worker.command_queue.close()
        worker.queue.close()
        worker.command_queue = Queue()
        worker.queue = Queue()
        try:
            job_id = worker.start("preview-band", {})
            process = Mock()
            process.is_alive.return_value = False
            worker.process = process
            worker.queue.put(
                {"job_id": job_id, "kind": "progress", "payload": {"phase": "parse"}}
            )

            message = worker.poll()

            self.assertEqual(message["job_id"], job_id)
            self.assertEqual(message["kind"], "error")
            self.assertEqual(worker.state, "idle")
            self.assertEqual(pool.used_slots, 0)
        finally:
            worker.close()

    def test_replacement_construction_failure_keeps_old_resources_and_finalizes_job(self):
        class Resource:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

            def cancel_join_thread(self):
                pass

        class FailingContext:
            def __init__(self):
                self.new_command = Resource()

            def SimpleQueue(self):
                return self.new_command

            def Queue(self):
                raise OSError("result queue allocation failed")

            def Event(self):
                raise AssertionError("Event must not be constructed after queue failure")

        pool = MemoryLeasePool()
        worker = SpawnWorker(memory_pool=pool)
        worker._ensure_process = lambda: None
        old_command = worker.command_queue
        old_result = worker.queue
        old_event = worker.cancel_event
        job_id = worker.start("preview-band", {})
        process = Mock()
        process.is_alive.return_value = False
        worker.process = process
        context = FailingContext()
        worker.context = context

        message = worker.poll()

        self.assertEqual(message["job_id"], job_id)
        self.assertEqual(message["kind"], "error")
        self.assertIs(worker.command_queue, old_command)
        self.assertIs(worker.queue, old_result)
        self.assertIs(worker.cancel_event, old_event)
        self.assertTrue(context.new_command.closed)
        self.assertEqual(worker.state, "idle")
        self.assertEqual(pool.used_slots, 0)
        worker.close()

    def test_forced_cleanup_rolls_back_interrupted_output_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            job_id = uuid.uuid4()
            staging = output / f".lyra-job-{job_id}"
            backup = staging / ".prior"
            backup.mkdir(parents=True)
            final = output / "stable.png"
            final.write_bytes(b"partial-new")
            (backup / "stable.png").write_bytes(b"prior-bytes")
            (staging / ".publication.json").write_text(
                json.dumps(
                    {"files": [{"relative": "stable.png", "had_prior": True}]}
                ),
                encoding="utf-8",
            )
            worker = SpawnWorker()
            try:
                worker._active_job_id = job_id
                worker._active_operation = "export-band"
                worker._active_payload = {"output": str(output), "folder": str(Path(tmp))}

                worker._cleanup_abandoned_render()

                self.assertEqual(final.read_bytes(), b"prior-bytes")
                self.assertFalse(staging.exists())
            finally:
                worker.close()

    def test_forced_cleanup_finishes_marked_output_and_cache_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / ".lyra-cache" / "previews"
            output.mkdir(parents=True)
            job_id = uuid.uuid4()
            staging = output / f".lyra-job-{job_id}"
            backup = staging / ".prior"
            backup.mkdir(parents=True)
            final = output / "stable.png"
            final.write_bytes(b"published-new")
            (backup / "stable.png").write_bytes(b"prior-bytes")
            (staging / ".publication.json").write_text(
                json.dumps(
                    {"files": [{"relative": "stable.png", "had_prior": True}]}
                ),
                encoding="utf-8",
            )
            (staging / ".publication-committed").write_text(
                "committed", encoding="ascii"
            )
            cache = Mock()
            worker = SpawnWorker()
            try:
                process = Mock()
                process.is_alive.return_value = False
                worker.process = process
                worker._active_job_id = job_id
                worker._active_operation = "rebuild-preview-band"
                worker._active_payload = {"folder": str(folder)}

                with patch("lyra.cache.ProjectionCache", return_value=cache):
                    worker._retire_process()

                self.assertEqual(final.read_bytes(), b"published-new")
                self.assertFalse(staging.exists())
                cache.finish_rebuild.assert_called_once_with(
                    str(folder), "band", None, success=True
                )
                cache.cleanup_incomplete.assert_not_called()
            finally:
                worker.close()

    def test_failed_parent_restore_retains_manifest_and_only_prior_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            job_id = uuid.uuid4()
            staging = output / f".lyra-job-{job_id}"
            backup = staging / ".prior"
            backup.mkdir(parents=True)
            final = output / "stable.png"
            final.write_bytes(b"partial-new")
            prior = backup / "stable.png"
            prior.write_bytes(b"prior-bytes")
            manifest = staging / ".publication.json"
            manifest.write_text(
                json.dumps(
                    {"files": [{"relative": "stable.png", "had_prior": True}]}
                ),
                encoding="utf-8",
            )
            worker = SpawnWorker()
            try:
                worker._active_job_id = job_id
                worker._active_operation = "export-band"
                worker._active_payload = {"output": str(output), "folder": str(Path(tmp))}
                with patch("lyra.workers.os.replace", side_effect=OSError("restore failed")):
                    worker._cleanup_abandoned_render()

                self.assertTrue(manifest.is_file())
                self.assertEqual(prior.read_bytes(), b"prior-bytes")
            finally:
                worker.close()

    def test_real_spawn_cooperative_cancellation_returns_cancelled_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            pool = MemoryLeasePool()
            worker = SpawnWorker(memory_pool=pool)
            try:
                job_id = worker.start(
                    "preview-band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(folder / "output"),
                        "settings": BandPlotSettings(
                            energy_range=PlotRange(-2.0, 2.0),
                            export=ExportSettings("cancelled", ("png",), 80),
                        ),
                    },
                )
                self.assertTrue(worker.cancel())
                message, _progress = self.wait_for_job(worker)
            finally:
                worker.close()

        self.assertEqual(message, {"job_id": job_id, "kind": "cancelled", "payload": None})
        self.assertEqual(pool.used_slots, 0)

    def test_start_returns_unique_uuid_and_emits_exact_matching_envelopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            worker = SpawnWorker()
            try:
                first_id = worker.start("clear-cache", {"folder": str(folder)})
                first_message, _progress = self.wait_for_job(worker)
                second_id = worker.start("clear-cache", {"folder": str(folder)})
                second_message, _progress = self.wait_for_job(worker)
            finally:
                worker.close()

        self.assertIsInstance(first_id, uuid.UUID)
        self.assertIsInstance(second_id, uuid.UUID)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(set(first_message), {"job_id", "kind", "payload"})
        self.assertEqual(set(second_message), {"job_id", "kind", "payload"})
        self.assertEqual(first_message["job_id"], first_id)
        self.assertEqual(second_message["job_id"], second_id)
        self.assertEqual(first_message["kind"], "cache-cleared")
        self.assertEqual(worker.state, "idle")

    def test_state_rejects_start_while_running_or_cancelling_and_cancel_is_idempotent(self):
        worker = SpawnWorker()
        try:
            worker._state = "running"
            with self.assertRaisesRegex(RuntimeError, "already running"):
                worker.start("band", {})
            worker._active_operation = "band"
            worker._active_job_id = uuid.uuid4()
            self.assertTrue(worker.cancel())
            self.assertEqual(worker.state, "cancelling")
            self.assertTrue(worker.running)
            self.assertFalse(worker.cancel())
            with self.assertRaisesRegex(RuntimeError, "already running"):
                worker.start("band", {})
        finally:
            worker.close()

    def test_clear_cache_refuses_cancellation(self):
        worker = SpawnWorker()
        try:
            worker._state = "running"
            worker._active_operation = "clear-cache"
            worker._active_job_id = uuid.uuid4()
            self.assertFalse(worker.cancel())
            self.assertEqual(worker.state, "running")
        finally:
            worker.close()

    def test_memory_admission_uses_one_slot_for_band_and_two_for_combined(self):
        pool = MemoryLeasePool(capacity_slots=2, slot_bytes=4 * 1024**3)
        worker = SpawnWorker(memory_pool=pool)
        try:
            worker._ensure_process = lambda: None
            worker.command_queue.put = lambda _message: None
            job_id = worker.start("preview-band", {})
            self.assertIsInstance(job_id, uuid.UUID)
            self.assertEqual(pool.used_slots, 1)
            worker._finish_active_job()
            self.assertEqual(pool.used_slots, 0)

            worker.start("preview-combined", {})
            self.assertEqual(pool.used_slots, 2)
            worker._finish_active_job()

            held = pool.acquire("external", 1)
            with self.assertRaises(MemoryBudgetExceeded):
                worker.start("preview-combined", {})
            self.assertEqual(worker.state, "idle")
            held.close()
        finally:
            worker.close()

    def test_worker_prepares_inspector_and_queries_one_band_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            result_queue = __import__("queue").Queue()
            from lyra.data_service import ProjectionDataService

            service = ProjectionDataService()
            prepared = _execute_job(
                service,
                result_queue,
                "prepare-band-inspector",
                {
                    "mode": "lite",
                    "folder": str(folder),
                },
            )
            detail = _execute_job(
                service,
                result_queue,
                "query-band-state",
                {
                    "mode": "lite",
                    "folder": str(folder),
                    "identity": prepared["identity"],
                    "selection": BandStateSelection("total", 0, 0),
                    "whole_band": False,
                },
            )

        self.assertEqual(prepared["index"].elements, ("X",))
        self.assertEqual(prepared["index"].raw_element_totals.shape, (1, 1, 9, 2))
        self.assertEqual(detail.selection, BandStateSelection("total", 0, 0))
        self.assertEqual(detail.raw_orbital_weights.shape, (1, 9))
        phases = []
        while not result_queue.empty():
            kind, event = result_queue.get_nowait()
            if kind == "progress":
                phases.append(event["phase"])
        self.assertIn("memory-hit", phases)

    def test_worker_prepares_data_driven_band_zoom(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            result_queue = Queue()
            from lyra.data_service import ProjectionDataService

            service = ProjectionDataService()
            prepared = _execute_job(
                service,
                result_queue,
                "prepare-band-inspector",
                {"mode": "lite", "folder": str(folder)},
            )
            settings = BandPlotSettings(
                PlotRange(-2, 2),
                mode="standard",
            )
            zoom = _execute_job(
                service,
                result_queue,
                "prepare-band-zoom",
                {
                    "mode": "lite",
                    "folder": str(folder),
                    "identity": prepared["identity"],
                    "settings": settings,
                },
            )

        self.assertEqual(zoom["identity"], prepared["identity"])
        self.assertEqual(zoom["data"].energies["total"].shape, (9, 2))
        self.assertIs(zoom["settings"], settings)

    def test_worker_prepares_two_independent_band_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            for folder in (left, right):
                write_band_fixture(folder)
                write_procar(folder)
            result_queue = __import__("queue").Queue()
            from lyra.data_service import ProjectionDataService

            session = _execute_job(
                ProjectionDataService(),
                result_queue,
                "prepare-paired-band",
                {
                    "left_folder": str(left),
                    "left_mode": "lite",
                    "right_folder": str(right),
                    "right_mode": "lite",
                },
            )

        self.assertEqual(session.left_folder, left.resolve())
        self.assertEqual(session.right_folder, right.resolve())
        self.assertTrue(session.compatibility.sync_enabled)

    def test_band_inspector_rejects_stale_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            result_queue = __import__("queue").Queue()
            from lyra.data_service import ProjectionDataService

            service = ProjectionDataService()
            prepared = _execute_job(
                service,
                result_queue,
                "prepare-band-inspector",
                {"mode": "lite", "folder": str(folder)},
            )
            with (folder / "PROCAR").open("a", encoding="utf-8") as handle:
                handle.write("\nsource changed")

            with self.assertRaisesRegex(ValueError, "changed|stale"):
                _execute_job(
                    service,
                    result_queue,
                    "query-band-state",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "identity": prepared["identity"],
                        "selection": BandStateSelection("total", 0, 0),
                    },
                )

    def test_fermi_worker_loads_uniform_grid_and_renders_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_fermi_fixture(folder)
            result_queue = __import__("queue").Queue()
            from lyra.data_service import ProjectionDataService

            with patch(
                "lyra.plotting.fermi.render_fermi",
                return_value=GeneratedResult(
                    (folder / "preview.png",),
                    folder / "preview.png",
                    "done",
                ),
            ) as renderer:
                result = _execute_job(
                    ProjectionDataService(),
                    result_queue,
                    "preview-fermi",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(folder / "output"),
                        "settings": FermiPlotSettings(),
                    },
                )

        renderer.assert_called_once()
        self.assertEqual(result.source_identities[0][1].data_kind, "fermi")
        self.assertIsNotNone(result.quality_report)
        self.assertIn("linear interpolation", result.quality_report.method)
        self.assertEqual(
            renderer.call_args.args[2],
            folder / ".lyra-cache" / "previews",
        )

    @staticmethod
    def wait_for_job(worker, timeout=30):
        deadline = time.monotonic() + timeout
        progress = []
        while time.monotonic() < deadline:
            message = worker.poll()
            if message is None:
                time.sleep(0.05)
                continue
            if message["kind"] == "progress":
                progress.append(message["payload"])
                if not worker.running:
                    raise AssertionError("Progress released the active worker job.")
                continue
            return message, progress
        raise AssertionError("Timed out waiting for worker job.")

    def test_spawned_worker_renders_without_blocking_main_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / "output"
            write_band_fixture(folder)
            worker = SpawnWorker()
            try:
                worker.start(
                    "band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": BandPlotSettings(
                            energy_range=PlotRange(-2.0, 2.0),
                            export=ExportSettings(
                                basename="worker_band", formats=("png",), dpi=80
                            ),
                        ),
                    },
                )
                message, _progress = self.wait_for_job(worker)
            finally:
                worker.close()

            self.assertEqual(message["kind"], "success")
            self.assertTrue((output / "worker_band.png").is_file())

    def test_projected_jobs_share_child_service_emit_progress_and_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / "output"
            write_band_fixture(folder)
            write_procar(folder)
            worker = SpawnWorker()
            try:
                first_settings = BandPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    mode="fat",
                    projection_channels=(ProjectionChannel("X", "s"),),
                    export=ExportSettings(
                        basename="first_projected", formats=("png",), dpi=80
                    ),
                )
                worker.start(
                    "band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": first_settings,
                    },
                )
                first_pid = worker.process.pid
                first_message, first_progress = self.wait_for_job(worker)

                self.assertEqual(first_message["kind"], "success")
                self.assertFalse(worker.running)
                self.assertTrue(worker.process.is_alive())
                self.assertTrue(first_progress)
                self.assertIn("cache-write", {event["phase"] for event in first_progress})

                second_settings = BandPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    mode="fat",
                    projection_channels=(ProjectionChannel("X", "d"),),
                    export=ExportSettings(
                        basename="second_projected", formats=("png",), dpi=80
                    ),
                )
                worker.start(
                    "band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": second_settings,
                    },
                )
                second_pid = worker.process.pid
                second_message, second_progress = self.wait_for_job(worker)

                self.assertEqual(second_message["kind"], "success")
                self.assertEqual(second_pid, first_pid)
                self.assertIn("memory-hit", {event["phase"] for event in second_progress})
                self.assertTrue((output / "first_projected.png").is_file())
                self.assertTrue((output / "second_projected.png").is_file())
                self.assertTrue((folder / ".lyra-cache").is_dir())

                worker.start("clear-cache", {"folder": str(folder)})
                clear_message, _clear_progress = self.wait_for_job(worker)
                self.assertEqual(clear_message["kind"], "cache-cleared")
                self.assertEqual(clear_message["payload"], {
                        "folder": str(folder.resolve()),
                        "message": "Projection cache cleared.",
                    })
                self.assertEqual(worker.process.pid, first_pid)
                self.assertFalse((folder / ".lyra-cache").exists())

                process = worker.process
            finally:
                worker.close()

            self.assertFalse(process.is_alive())
            self.assertIsNone(worker.process)

    def test_preview_uses_a_contained_fixed_basename_and_export_reuses_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / "output"
            write_band_fixture(folder)
            write_procar(folder)
            settings = BandPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                mode="fat",
                projection_channels=(ProjectionChannel("X", "s"),),
                export=ExportSettings(
                    basename="../../escaped", formats=("png", "pdf"), dpi=240
                ),
            )
            worker = SpawnWorker()
            try:
                worker.start(
                    "preview-band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": settings,
                    },
                )
                preview_message, preview_progress = self.wait_for_job(worker)
                self.assertEqual(preview_message["kind"], "success")
                preview = preview_message["payload"].preview_file
                self.assertEqual(preview.parent, folder / ".lyra-cache" / "previews")
                self.assertEqual(preview.name, "lyra_preview_band.png")
                self.assertEqual(preview_message["payload"].output_files, (preview,))
                self.assertFalse((output / "escaped.png").exists())
                self.assertTrue(preview.is_file())

                worker.start(
                    "export-band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                    "settings": BandPlotSettings(
                        energy_range=settings.energy_range,
                        mode=settings.mode,
                        projection_channels=settings.projection_channels,
                        export=ExportSettings(
                            basename="saved_band", formats=("png", "pdf"), dpi=240
                        ),
                    ),
                    "preview_sources": preview_message["payload"].source_identities,
                    },
                )
                export_message, export_progress = self.wait_for_job(worker)
            finally:
                worker.close()

            self.assertEqual(export_message["kind"], "success")
            self.assertTrue((output / "saved_band.png").is_file())
            self.assertTrue((output / "saved_band.pdf").is_file())
            self.assertTrue((output / "saved_band.lyra-report.json").is_file())
            self.assertTrue((output / "saved_band.lyra-report.html").is_file())
            self.assertEqual(len(export_message["payload"].report_files), 2)
            self.assertIn("memory-hit", {event["phase"] for event in export_progress})
            self.assertIn("cache-write", {event["phase"] for event in preview_progress})
            self.assertIn("metadata", {event["phase"] for event in preview_progress})
            self.assertIn("parse", {event["phase"] for event in preview_progress})
            self.assertIn("complete", {event["phase"] for event in preview_progress})
            self.assertIn("preview-render", {event["phase"] for event in preview_progress})
            self.assertIn("export", {event["phase"] for event in export_progress})
            parse_details = [event["detail"] for event in preview_progress if event["phase"] == "parse"]
            self.assertTrue(any("k-point" in detail and "table" in detail for detail in parse_details))
            self.assertTrue(all("elapsed_seconds" in event for event in preview_progress))

    def test_combined_rebuild_bypasses_only_the_requested_namespace(self):
        class RecordingService:
            def __init__(self):
                self.rebuilds = []

            def clear(self, folder, data_kind):
                raise AssertionError("rebuild must not clear an indexed generation")

            def load_band(self, *args, **kwargs):
                self.rebuilds.append(("band", kwargs.get("rebuild")))
                from tests.test_projection_store import parsed_band_data

                return parsed_band_data()

            @staticmethod
            def band_identity(folder, mode, channels, progress):
                return CacheIdentity(
                    schema_version=1,
                    parser_version="test",
                    read_mode=mode.value,
                    data_kind="band",
                    sources=(CacheSource("EIGENVAL", 1, 1),),
                )

            def load_dos(self, *args, **kwargs):
                self.rebuilds.append(("dos", kwargs.get("rebuild")))
                from tests.test_projection_store import parsed_dos_data

                return parsed_dos_data()

            @staticmethod
            def dos_identity(folder, mode, channels, progress):
                return CacheIdentity(
                    schema_version=1,
                    parser_version="test",
                    read_mode=mode.value,
                    data_kind="dos",
                    sources=(CacheSource("DOSCAR", 1, 1),),
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band = root / "band"
            dos = root / "dos"
            settings = CombinedPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=ExportSettings(basename="formal", formats=("png",), dpi=80),
            )
            for data_kind, expected in (
                ("band", [("band", True), ("dos", False)]),
                ("dos", [("band", False), ("dos", True)]),
            ):
                with self.subTest(data_kind=data_kind):
                    service = RecordingService()
                    _execute_job(
                        service,
                        __import__("queue").Queue(),
                        "rebuild-preview-combined",
                        {
                            "mode": "lite",
                            "band_folder": str(band),
                            "dos_folder": str(dos),
                            "output": str(root / "output"),
                            "settings": settings,
                            "rebuild_data_kind": data_kind,
                        },
                    )
                    self.assertEqual(service.rebuilds, expected)

    def test_combined_uses_independent_band_and_dos_read_modes(self):
        class RecordingService:
            def __init__(self):
                self.modes = []

            def band_identity(self, folder, mode, channels, progress):
                self.modes.append(("band-identity", mode.value))
                return CacheIdentity(1, "test", mode.value, "band", (CacheSource("vasprun.xml", 1, 1),))

            def dos_identity(self, folder, mode, channels, progress):
                self.modes.append(("dos-identity", mode.value))
                return CacheIdentity(1, "test", mode.value, "dos", (CacheSource("DOSCAR", 1, 1),))

            def load_band(self, folder, mode, channels, progress, **kwargs):
                self.modes.append(("band-load", mode.value))
                from tests.test_projection_store import parsed_band_data

                return parsed_band_data()

            def load_dos(self, folder, mode, channels, progress, **kwargs):
                self.modes.append(("dos-load", mode.value))
                from tests.test_projection_store import parsed_dos_data

                return parsed_dos_data()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RecordingService()
            settings = CombinedPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=ExportSettings(basename="formal", formats=("png",), dpi=80),
            )
            with patch(
                "lyra.plotting.combined.render_combined",
                return_value=GeneratedResult((), root / "preview.png", "done"),
            ):
                _execute_job(
                    service,
                    __import__("queue").Queue(),
                    "preview-combined",
                    {
                        "mode": "lite",
                        "band_mode": "full",
                        "dos_mode": "lite",
                        "band_folder": str(root / "band"),
                        "dos_folder": str(root / "dos"),
                        "output": str(root / "output"),
                        "settings": settings,
                    },
                )

        self.assertEqual(
            service.modes,
            [
                ("band-identity", "full"),
                ("dos-identity", "lite"),
                ("band-load", "full"),
                ("dos-load", "lite"),
            ],
        )

    def test_combined_export_rejects_preview_identity_from_another_read_mode(self):
        class GuardedService:
            @staticmethod
            def identity_matches(folder, identity):
                return True

            def __getattr__(self, name):
                raise AssertionError(f"Worker reached {name} after a mode mismatch.")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band = root / "band"
            dos = root / "dos"
            settings = CombinedPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=ExportSettings(basename="formal", formats=("png",), dpi=80),
            )
            preview_sources = (
                (
                    band,
                    CacheIdentity(1, "test", "lite", "band", (CacheSource("EIGENVAL", 1, 1),)),
                ),
                (
                    dos,
                    CacheIdentity(1, "test", "lite", "dos", (CacheSource("DOSCAR", 1, 1),)),
                ),
            )

            with self.assertRaisesRegex(ValueError, "read mode|new preview"):
                _execute_job(
                    GuardedService(),
                    __import__("queue").Queue(),
                    "export-combined",
                    {
                        "mode": "lite",
                        "band_mode": "full",
                        "dos_mode": "lite",
                        "band_folder": str(band),
                        "dos_folder": str(dos),
                        "output": str(root / "output"),
                        "settings": settings,
                        "preview_sources": preview_sources,
                    },
                )

    def test_combined_phase_timing_scopes_repeated_phases_and_reports_render_completion(self):
        class Clock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                return self.value

        class TimedService:
            @staticmethod
            def band_identity(folder, mode, channels, progress):
                return CacheIdentity(1, "test", mode.value, "band", (CacheSource("EIGENVAL", 1, 1),))

            @staticmethod
            def dos_identity(folder, mode, channels, progress):
                return CacheIdentity(1, "test", mode.value, "dos", (CacheSource("DOSCAR", 1, 1),))

            def load_band(self, folder, mode, channels, progress, **kwargs):
                clock.value = 2.0
                progress({"phase": "parse", "current": 1, "total": 2, "detail": "Band table"})
                clock.value = 5.0
                progress({"phase": "parse", "current": 2, "total": 2, "detail": "Band k-point"})
                from tests.test_projection_store import parsed_band_data

                return parsed_band_data()

            def load_dos(self, folder, mode, channels, progress, **kwargs):
                clock.value = 9.0
                progress({"phase": "parse", "current": 1, "total": 1, "detail": "DOS table"})
                clock.value = 12.0
                from tests.test_projection_store import parsed_dos_data

                return parsed_dos_data()

        def events(result_queue):
            received = []
            while True:
                try:
                    kind, event = result_queue.get_nowait()
                except Empty:
                    return received
                if kind == "progress":
                    received.append(event)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = CombinedPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=ExportSettings(basename="formal", formats=("png",), dpi=80),
            )
            for operation, render_phase in (("preview-combined", "preview-render"), ("combined", "export")):
                with self.subTest(operation=operation):
                    clock = Clock()
                    result_queue = __import__("queue").Queue()

                    def render(*args):
                        clock.value = 17.0
                        return GeneratedResult((), root / "preview.png", "done")

                    with (
                        patch("lyra.workers.time.monotonic", clock),
                        patch("lyra.plotting.combined.render_combined", side_effect=render),
                    ):
                        _execute_job(
                            TimedService(),
                            result_queue,
                            operation,
                            {
                                "mode": "lite",
                                "band_folder": str(root / "band"),
                                "dos_folder": str(root / "dos"),
                                "output": str(root / "output"),
                                "settings": settings,
                            },
                        )

                    progress = events(result_queue)
                    self.assertEqual(
                        [event["elapsed_seconds"] for event in progress if event["phase"] == "parse"],
                        [0.0, 3.0, 0.0],
                    )
                    render_events = [event for event in progress if event["phase"] == render_phase]
                    self.assertEqual(len(render_events), 2)
                    self.assertEqual(render_events[0]["elapsed_seconds"], 0.0)
                    self.assertEqual(render_events[1]["elapsed_seconds"], 5.0)
                    self.assertIn("Finished", render_events[1]["detail"])

    def test_band_and_dos_render_events_keep_source_context(self):
        class Service:
            @staticmethod
            def band_identity(folder, mode, channels, progress):
                return CacheIdentity(1, "test", mode.value, "band", (CacheSource("EIGENVAL", 1, 1),))

            @staticmethod
            def dos_identity(folder, mode, channels, progress):
                return CacheIdentity(1, "test", mode.value, "dos", (CacheSource("DOSCAR", 1, 1),))

            @staticmethod
            def load_band(*args, **kwargs):
                from tests.test_projection_store import parsed_band_data

                return parsed_band_data()

            @staticmethod
            def load_dos(*args, **kwargs):
                from tests.test_projection_store import parsed_dos_data

                return parsed_dos_data()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "calculation"
            cases = (
                (
                    "preview-band",
                    "preview-render",
                    BandPlotSettings(PlotRange(-2.0, 2.0), export=ExportSettings("band", ("png",), 80)),
                    "lyra.plotting.band.render_band",
                ),
                (
                    "band",
                    "export",
                    BandPlotSettings(PlotRange(-2.0, 2.0), export=ExportSettings("band", ("png",), 80)),
                    "lyra.plotting.band.render_band",
                ),
                (
                    "preview-dos",
                    "preview-render",
                    DosPlotSettings(PlotRange(-2.0, 2.0), export=ExportSettings("dos", ("png",), 80)),
                    "lyra.plotting.dos.render_dos",
                ),
                (
                    "dos",
                    "export",
                    DosPlotSettings(PlotRange(-2.0, 2.0), export=ExportSettings("dos", ("png",), 80)),
                    "lyra.plotting.dos.render_dos",
                ),
            )
            for operation, render_phase, settings, renderer_path in cases:
                with self.subTest(operation=operation):
                    result_queue = __import__("queue").Queue()
                    with patch(
                        renderer_path,
                        return_value=GeneratedResult((), root / "preview.png", "done"),
                    ):
                        _execute_job(
                            Service(),
                            result_queue,
                            operation,
                            {
                                "mode": "lite",
                                "folder": str(folder),
                                "output": str(root / "output"),
                                "settings": settings,
                            },
                        )

                    render_events = []
                    while True:
                        try:
                            kind, event = result_queue.get_nowait()
                        except Empty:
                            break
                        if kind == "progress" and event["phase"] == render_phase:
                            render_events.append(event)
                    self.assertEqual(len(render_events), 2)
                    expected_kind = "band" if operation.endswith("band") else "dos"
                    self.assertEqual({event["data_kind"] for event in render_events}, {expected_kind})
                    self.assertEqual({event["folder"] for event in render_events}, {str(folder.resolve())})

    def test_dead_child_discards_stale_queues_before_next_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "band"
            stale_folder = root / "must-not-clear"
            output = root / "output"
            folder.mkdir()
            (stale_folder / ".lyra-cache").mkdir(parents=True)
            (stale_folder / ".lyra-cache" / "marker").write_text(
                "keep", encoding="utf-8"
            )
            write_band_fixture(folder)
            worker = SpawnWorker()
            try:
                first_settings = BandPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    export=ExportSettings(
                        basename="before_death", formats=("png",), dpi=80
                    ),
                )
                worker.start(
                    "band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": first_settings,
                    },
                )
                first_message, _first_progress = self.wait_for_job(worker)
                self.assertEqual(first_message["kind"], "success")

                old_command_queue = worker.command_queue
                old_result_queue = worker.queue
                worker.process.terminate()
                worker.process.join(timeout=5.0)
                self.assertFalse(worker.process.is_alive())
                old_command_queue.put(
                    ("job", "clear-cache", {"folder": str(stale_folder)})
                )

                next_settings = BandPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    export=ExportSettings(
                        basename="after_restart", formats=("png",), dpi=80
                    ),
                )
                worker.start(
                    "band",
                    {
                        "mode": "lite",
                        "folder": str(folder),
                        "output": str(output),
                        "settings": next_settings,
                    },
                )

                self.assertIsNot(worker.command_queue, old_command_queue)
                self.assertIsNot(worker.queue, old_result_queue)
                next_message, next_progress = self.wait_for_job(worker)

                self.assertEqual(next_message["kind"], "success")
                self.assertEqual(next_message["payload"].preview_file.name, "after_restart.png")
                self.assertTrue(next_progress)
                self.assertTrue((stale_folder / ".lyra-cache" / "marker").is_file())
                self.assertTrue((output / "after_restart.png").is_file())
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()

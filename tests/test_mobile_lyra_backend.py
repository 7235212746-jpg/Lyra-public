from __future__ import annotations

from pathlib import Path
import json
import time

import pytest

from lyra.mobile.lyra_backend import AuthorizedDirectoryRegistry, LyraTaskBackend
from lyra.mobile.resources import ResourceRegistry
from lyra.mobile.tasks import TaskError
from lyra.models import GeneratedResult
from tests.fixtures import write_band_fixture, write_dos_fixture


class FakeWorker:
    def __init__(self) -> None:
        self.started = []
        self.messages = []
        self.cancelled = False
        self.closed = False

    def start(self, operation, payload):
        self.started.append((operation, payload))
        return "worker-job-id"

    def poll(self):
        return self.messages.pop(0) if self.messages else None

    def cancel(self):
        self.cancelled = True
        return True

    def close(self):
        self.closed = True


def authorize(tmp_path: Path):
    folder = tmp_path / "calculation"
    folder.mkdir()
    directories = AuthorizedDirectoryRegistry()
    public = directories.authorize(folder, label="SOC band")
    return directories, public, folder


def test_authorized_directory_registry_exposes_only_opaque_identity(tmp_path: Path) -> None:
    directories, public, folder = authorize(tmp_path)

    assert "/" not in public.directory_id and "\\" not in public.directory_id
    assert str(folder) not in repr(public)
    assert public.to_public_dict() == {"id": public.directory_id, "label": "SOC band"}
    assert directories.resolve(public.directory_id) == folder.resolve()

    directories.revoke(public.directory_id)
    with pytest.raises(TaskError, match="not authorized"):
        directories.resolve(public.directory_id)


def test_authorized_directories_keep_stable_opaque_ids_across_restarts(tmp_path: Path) -> None:
    folder = tmp_path / "calculation"
    folder.mkdir()
    state_file = tmp_path / "directories.json"
    first = AuthorizedDirectoryRegistry(state_file=state_file)
    public = first.authorize(folder, label="SOC band")

    second = AuthorizedDirectoryRegistry(state_file=state_file)

    assert second.list_public() == (public,)
    assert second.resolve(public.directory_id) == folder.resolve()
    saved = state_file.read_text(encoding="utf-8")
    assert public.directory_id in saved
    assert "password" not in saved.casefold()


def test_band_task_resolves_directory_privately_and_starts_real_operation_shape(tmp_path: Path) -> None:
    directories, public, folder = authorize(tmp_path)
    worker = FakeWorker()
    backend = LyraTaskBackend(
        directories=directories,
        resources=ResourceRegistry(),
        worker_factory=lambda: worker,
        autostart=False,
    )
    try:
        task = backend.create("band", {"directory_id": public.directory_id, "energy_range": [-3, 2]})
        backend.poll_once()

        assert task.to_public_dict()["parameters"]["directory_id"] == public.directory_id
        operation, payload = worker.started[0]
        assert operation == "preview-band"
        assert payload["folder"] == str(folder.resolve())
        assert payload["mode"] == "lite"
        assert payload["settings"].mode == "standard"
        assert payload["settings"].energy_range.minimum == -3
    finally:
        backend.close()


def test_dos_and_combined_build_standard_non_projected_preview_payloads(tmp_path: Path) -> None:
    band = tmp_path / "band"
    dos = tmp_path / "dos"
    band.mkdir()
    dos.mkdir()
    directories = AuthorizedDirectoryRegistry()
    band_id = directories.authorize(band, label="Band").directory_id
    dos_id = directories.authorize(dos, label="DOS").directory_id

    for kind, parameters, operation in (
        ("dos", {"directory_id": dos_id}, "preview-dos"),
        ("combined", {"band_directory_id": band_id, "dos_directory_id": dos_id}, "preview-combined"),
    ):
        worker = FakeWorker()
        backend = LyraTaskBackend(
            directories=directories,
            resources=ResourceRegistry(),
            worker_factory=lambda worker=worker: worker,
            autostart=False,
        )
        try:
            backend.create(kind, parameters)
            backend.poll_once()
            actual_operation, payload = worker.started[0]
            assert actual_operation == operation
            settings = payload["settings"]
            if kind == "dos":
                assert settings.projection_channels == ()
            else:
                assert settings.band_mode == "standard"
                assert settings.band_projection_channels == ()
                assert settings.dos_projection_channels == ()
                assert payload["band_folder"] == str(band.resolve())
                assert payload["dos_folder"] == str(dos.resolve())
        finally:
            backend.close()


def test_progress_success_and_preview_registration(tmp_path: Path) -> None:
    directories, public, folder = authorize(tmp_path)
    preview = folder / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"\x89PNG\r\n\x1a\nplot")
    worker = FakeWorker()
    registry = ResourceRegistry()
    backend = LyraTaskBackend(
        directories=directories,
        resources=registry,
        worker_factory=lambda: worker,
        autostart=False,
    )
    try:
        task = backend.create("band", {"directory_id": public.directory_id})
        backend.poll_once()
        worker.messages.extend([
            {"job_id": "x", "kind": "progress", "payload": {"phase": "parse", "current": 4, "total": 10, "detail": "Reading k-points"}},
            {"job_id": "x", "kind": "success", "payload": GeneratedResult((preview,), preview, "done")},
        ])

        backend.poll_once()
        running = backend.get(task.task_id)
        assert running.state == "running"
        assert 0 < running.progress < 1
        assert running.stage == "Reading k-points"

        backend.poll_once()
        complete = backend.get(task.task_id)
        assert complete.state == "completed"
        assert complete.progress == 1
        resource = complete.resources[0]
        assert "path" not in resource
        assert registry.resolve(resource["id"], purpose="preview").path == preview.resolve()
    finally:
        backend.close()


def test_active_cancel_delegates_to_spawn_worker(tmp_path: Path) -> None:
    directories, public, _folder = authorize(tmp_path)
    worker = FakeWorker()
    backend = LyraTaskBackend(
        directories=directories,
        resources=ResourceRegistry(),
        worker_factory=lambda: worker,
        autostart=False,
    )
    try:
        task = backend.create("band", {"directory_id": public.directory_id})
        backend.poll_once()
        cancelling = backend.cancel(task.task_id)
        assert worker.cancelled
        assert cancelling.state == "cancelling"
        worker.messages.append({"job_id": "x", "kind": "cancelled", "payload": None})
        backend.poll_once()
        assert backend.get(task.task_id).state == "cancelled"
    finally:
        backend.close()


@pytest.mark.parametrize(
    "parameters",
    [
        {"path": "C:/private"},
        {"folder": "/srv/vasp"},
        {"password": "secret"},
        {"directory_id": "unknown", "projection_channels": [{"element": "Ta"}]},
    ],
)
def test_backend_rejects_paths_credentials_and_projection(parameters: dict) -> None:
    backend = LyraTaskBackend(
        directories=AuthorizedDirectoryRegistry(),
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
    )
    try:
        with pytest.raises(TaskError):
            backend.create("band", parameters)
    finally:
        backend.close()


def test_backend_limits_active_mobile_queue(tmp_path: Path) -> None:
    directories, public, _folder = authorize(tmp_path)
    backend = LyraTaskBackend(
        directories=directories,
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
    )
    try:
        for _ in range(10):
            backend.create("band", {"directory_id": public.directory_id})
        with pytest.raises(TaskError, match="queue"):
            backend.create("band", {"directory_id": public.directory_id})
    finally:
        backend.close()


def test_capabilities_clearly_report_projection_limit() -> None:
    backend = LyraTaskBackend(
        directories=AuthorizedDirectoryRegistry(),
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
    )
    try:
        capabilities = backend.capabilities()
        assert capabilities["task_kinds"] == ["band", "dos", "combined"]
        assert capabilities["projection"]["supported"] is False
        assert "desktop" in capabilities["projection"]["reason"].lower()
    finally:
        backend.close()


def test_task_history_survives_restart_and_marks_active_work_interrupted(tmp_path: Path) -> None:
    directories, public, _folder = authorize(tmp_path)
    state = tmp_path / "tasks.json"
    first = LyraTaskBackend(
        directories=directories,
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
        state_file=state,
    )
    task = first.create("band", {"directory_id": public.directory_id})
    first.poll_once()
    assert first.get(task.task_id).state == "running"
    first.close()

    second = LyraTaskBackend(
        directories=directories,
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
        state_file=state,
    )
    try:
        restored = second.get(task.task_id)
        assert restored.state == "failed"
        assert "restart" in restored.message.lower()
        assert "folder" not in repr(restored.to_public_dict()).lower()
    finally:
        second.close()


def test_invalid_persisted_task_history_is_discarded_without_exposing_paths(tmp_path: Path) -> None:
    state = tmp_path / "tasks.json"
    state.write_text(json.dumps({
        "version": 1,
        "tasks": [{
            "id": "../../bad",
            "kind": "band",
            "parameters": {"path": "C:/private/secret"},
            "state": "completed",
            "progress": float("nan"),
            "stage": "Done",
            "message": "",
            "resources": [],
            "created_at": 1,
            "updated_at": 1,
        }],
    }), encoding="utf-8")
    backend = LyraTaskBackend(
        directories=AuthorizedDirectoryRegistry(),
        resources=ResourceRegistry(),
        worker_factory=FakeWorker,
        autostart=False,
        state_file=state,
    )
    try:
        assert backend.list_tasks() == []
        assert "private" not in state.read_text(encoding="utf-8").lower()
    finally:
        backend.close()


def test_real_spawn_worker_generates_and_registers_band_preview(tmp_path: Path) -> None:
    folder = tmp_path / "band"
    folder.mkdir()
    write_band_fixture(folder)
    directories = AuthorizedDirectoryRegistry()
    directory_id = directories.authorize(folder, label="Band fixture").directory_id
    resources = ResourceRegistry()
    backend = LyraTaskBackend(directories=directories, resources=resources)
    try:
        task = backend.create("band", {"directory_id": directory_id, "energy_range": [-2, 2]})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = backend.get(task.task_id)
            if current.state in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        else:
            pytest.fail("real SpawnWorker preview did not finish within 30 seconds")

        assert current.state == "completed", current.message
        assert current.resources and current.resources[0]["previewable"] is True
        registered = resources.resolve(current.resources[0]["id"], purpose="preview")
        assert registered.path.is_file()
        assert registered.path.name == "lyra_preview_band.png"
    finally:
        backend.close()


def test_real_spawn_worker_generates_dos_and_combined_previews(tmp_path: Path) -> None:
    band_folder = tmp_path / "band"
    dos_folder = tmp_path / "dos"
    band_folder.mkdir()
    dos_folder.mkdir()
    write_band_fixture(band_folder)
    write_dos_fixture(dos_folder, projected=False)
    directories = AuthorizedDirectoryRegistry()
    band_id = directories.authorize(band_folder, label="Band fixture").directory_id
    dos_id = directories.authorize(dos_folder, label="DOS fixture").directory_id
    resources = ResourceRegistry()
    backend = LyraTaskBackend(directories=directories, resources=resources)

    def wait_for(task_id: str):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = backend.get(task_id)
            if current.state in {"completed", "failed", "cancelled"}:
                return current
            time.sleep(0.05)
        pytest.fail("real SpawnWorker preview did not finish within 30 seconds")

    try:
        dos_task = backend.create("dos", {"directory_id": dos_id, "energy_range": [-2, 2]})
        dos_result = wait_for(dos_task.task_id)
        assert dos_result.state == "completed", dos_result.message
        assert resources.resolve(dos_result.resources[0]["id"], purpose="preview").path.name == "lyra_preview_dos.png"

        combined_task = backend.create(
            "combined",
            {"band_directory_id": band_id, "dos_directory_id": dos_id, "energy_range": [-2, 2]},
        )
        combined_result = wait_for(combined_task.task_id)
        assert combined_result.state == "completed", combined_result.message
        assert resources.resolve(combined_result.resources[0]["id"], purpose="preview").path.name == "lyra_preview_combined.png"
    finally:
        backend.close()

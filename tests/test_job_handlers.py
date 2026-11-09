from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from queue import Queue
import threading
import uuid
from unittest.mock import patch

from lyra.cache import CacheIdentity, CacheSource
from lyra.models import BandPlotSettings, ExportSettings, GeneratedResult, PlotRange
from tests.test_projection_store import parsed_band_data
from tests.fixtures import write_band_fixture, write_procar


def _handlers():
    assert importlib.util.find_spec("lyra.job_handlers") is not None
    return importlib.import_module("lyra.job_handlers")


def test_workers_execute_job_is_the_handler_compatibility_alias():
    from lyra import workers

    handlers = _handlers()

    assert workers._execute_job is handlers.execute_job


def test_job_handlers_owns_job_dispatch_and_preview_helpers():
    from lyra import workers

    handlers = _handlers()

    worker_source = inspect.getsource(workers)
    handler_source = inspect.getsource(handlers)
    for name in ("SCREEN_PREVIEW_DPI", "PREVIEW_BASENAMES"):
        assert name not in worker_source
        assert name in handler_source
    for name in (
        "_job_parts",
        "_preview_settings",
        "_preview_directory",
        "_execute_job",
    ):
        assert f"def {name}" not in worker_source
    for name in ("_job_parts", "_preview_settings", "_preview_directory", "execute_job"):
        assert f"def {name}" in handler_source


def test_job_handlers_has_no_worker_process_management_dependencies():
    handlers = _handlers()

    module_source = inspect.getsource(handlers)
    tree = ast.parse(module_source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "multiprocessing" not in imports
    assert "queue" not in imports
    assert "lyra.workers" not in imports
    assert not hasattr(handlers, "SpawnWorker")


def test_job_handler_helpers_preserve_preview_operations(tmp_path):
    handlers = _handlers()
    settings = BandPlotSettings(
        energy_range=PlotRange(-2.0, 2.0),
        export=ExportSettings("exported_band", ("pdf",), 300),
    )

    assert handlers._job_parts("rebuild-preview-band") == ("rebuild-preview", "band")
    assert handlers._job_parts("preview-dos") == ("preview", "dos")
    assert handlers._job_parts("fermi") == ("export", "fermi")

    preview_settings = handlers._preview_settings(settings, "band")
    assert preview_settings.export == ExportSettings(
        "lyra_preview_band", ("png",), handlers.SCREEN_PREVIEW_DPI
    )
    assert handlers._preview_directory(
        {"folder": tmp_path / "fermi", "band_folder": tmp_path / "band"}, "fermi"
    ) == (tmp_path / "fermi").resolve() / ".lyra-cache" / "previews"
    assert handlers._preview_directory(
        {"folder": tmp_path / "combined", "band_folder": tmp_path / "band"}, "combined"
    ) == (tmp_path / "band").resolve() / ".lyra-cache" / "previews"


class _RebuildService:
    def __init__(self):
        self.rebuild_flags = []

    def clear(self, *_args, **_kwargs):
        raise AssertionError("rebuild must not destructively clear the cache")

    @staticmethod
    def band_identity(folder, mode, channels, progress):
        progress({"phase": "validate", "current": 0, "total": 0, "detail": "identity"})
        return CacheIdentity(
            1,
            "test",
            mode.value,
            "band",
            (CacheSource("EIGENVAL", 1, 1),),
        )

    def load_band(self, folder, mode, channels, progress, **kwargs):
        self.rebuild_flags.append(kwargs.get("rebuild"))
        return parsed_band_data()


def _band_payload(tmp_path):
    return {
        "folder": str(tmp_path),
        "output": str(tmp_path / "output"),
        "mode": "lite",
        "settings": BandPlotSettings(
            energy_range=PlotRange(-2.0, 2.0),
            export=ExportSettings("stable", ("png",), 80),
        ),
    }


def test_rebuild_bypasses_cache_without_destructive_clear_and_publishes_from_staging(tmp_path):
    handlers = _handlers()
    service = _RebuildService()
    job_id = uuid.uuid4()

    def render(_data, settings, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        image = output / f"{settings.export.basename}.png"
        image.write_bytes(b"new-preview")
        return GeneratedResult((image,), image, "done")

    with patch("lyra.plotting.band.render_band", side_effect=render):
        result = handlers.execute_job(
            service,
            Queue(),
            "rebuild-preview-band",
            _band_payload(tmp_path),
            job_id=job_id,
            cancel_event=threading.Event(),
        )

    expected = tmp_path / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    assert service.rebuild_flags == [True]
    assert result.output_files == (expected,)
    assert result.preview_file == expected
    assert expected.read_bytes() == b"new-preview"
    assert not list(expected.parent.glob(".lyra-job-*"))


def test_render_error_preserves_prior_bytes_and_removes_job_staging(tmp_path):
    handlers = _handlers()
    final = tmp_path / "output" / "stable.png"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"prior-bytes")

    def fail_render(_data, settings, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{settings.export.basename}.png").write_bytes(b"partial")
        raise RuntimeError("render failed")

    with patch("lyra.plotting.band.render_band", side_effect=fail_render):
        try:
            handlers.execute_job(
                _RebuildService(),
                Queue(),
                "band",
                _band_payload(tmp_path),
                job_id=uuid.uuid4(),
                cancel_event=threading.Event(),
            )
        except RuntimeError as exc:
            assert "render failed" in str(exc)
        else:
            raise AssertionError("render failure was not propagated")

    assert final.read_bytes() == b"prior-bytes"
    assert not list(final.parent.glob(".lyra-job-*"))


def test_cancelled_rebuild_restores_prior_index_and_source_bytes(tmp_path):
    from lyra.data_service import ProjectionDataService
    from lyra.models import ProjectionChannel

    write_band_fixture(tmp_path)
    write_procar(tmp_path)
    service = ProjectionDataService()
    channel = (ProjectionChannel("X", "s"),)
    service.load_band(tmp_path, "lite", channel, lambda _event: None)
    index = tmp_path / ".lyra-cache" / "band-projection-v1" / "index.json"
    prior_index = index.read_bytes()
    sources = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    event = threading.Event()

    class CancellingQueue(Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            if item["kind"] == "progress" and item["payload"].get("phase") == "preview-render":
                event.set()

    payload = _band_payload(tmp_path)
    payload["settings"] = BandPlotSettings(
        energy_range=PlotRange(-2.0, 2.0),
        mode="fat",
        projection_channels=channel,
        export=ExportSettings("stable", ("png",), 80),
    )

    try:
        _handlers().execute_job(
            service,
            CancellingQueue(),
            "rebuild-preview-band",
            payload,
            job_id=uuid.uuid4(),
            cancel_event=event,
        )
    except _handlers().JobCancelled:
        pass
    else:
        raise AssertionError("rebuild did not observe cancellation")

    assert index.read_bytes() == prior_index
    assert {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    } == sources


def test_cancelled_export_preserves_prior_bytes_and_removes_staging(tmp_path):
    final = tmp_path / "output" / "stable.png"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"prior-export")
    event = threading.Event()

    class CancellingQueue(Queue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            if item["kind"] == "progress" and item["payload"].get("phase") == "export":
                event.set()

    try:
        _handlers().execute_job(
            _RebuildService(),
            CancellingQueue(),
            "band",
            _band_payload(tmp_path),
            job_id=uuid.uuid4(),
            cancel_event=event,
        )
    except _handlers().JobCancelled:
        pass
    else:
        raise AssertionError("export did not observe cancellation")

    assert final.read_bytes() == b"prior-export"
    assert not list(final.parent.glob(".lyra-job-*"))


def _prime_projected_band_cache(tmp_path):
    from lyra.data_service import ProjectionDataService
    from lyra.models import ProjectionChannel

    write_band_fixture(tmp_path)
    write_procar(tmp_path)
    service = ProjectionDataService()
    channel = (ProjectionChannel("X", "s"),)
    service.load_band(tmp_path, "lite", channel, lambda _event: None)
    index = tmp_path / ".lyra-cache" / "band-projection-v1" / "index.json"
    generation = json.loads(index.read_text(encoding="utf-8"))["generation"]
    payload = _band_payload(tmp_path)
    payload["settings"] = BandPlotSettings(
        energy_range=PlotRange(-2.0, 2.0),
        mode="fat",
        projection_channels=channel,
        export=ExportSettings("stable", ("png",), 80),
    )
    return service, payload, index, generation


def _staged_band_render(_data, settings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = output / f"{settings.export.basename}.png"
    image.write_bytes(b"new-preview")
    return GeneratedResult((image,), image, "done")


def test_rebuild_publication_failure_retains_prior_cache_generation(tmp_path):
    handlers = _handlers()
    service, payload, index, prior_generation = _prime_projected_band_cache(tmp_path)
    prior_index = index.read_bytes()
    prior_directory = index.parent / "generations" / prior_generation
    preview = tmp_path / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"prior-preview")

    with (
        patch("lyra.plotting.band.render_band", side_effect=_staged_band_render),
        patch(
            "lyra.job_handlers._publish_generated_result",
            side_effect=OSError("publication failed"),
        ),
    ):
        try:
            handlers.execute_job(
                service,
                Queue(),
                "rebuild-preview-band",
                payload,
                job_id=uuid.uuid4(),
                cancel_event=threading.Event(),
            )
        except OSError as exc:
            assert "publication failed" in str(exc)
        else:
            raise AssertionError("publication failure was not propagated")

    assert index.read_bytes() == prior_index
    assert prior_directory.is_dir()
    assert preview.read_bytes() == b"prior-preview"


def test_cancellation_at_publication_entry_aborts_rebuild_before_commit(tmp_path):
    handlers = _handlers()
    service, payload, index, prior_generation = _prime_projected_band_cache(tmp_path)
    prior_index = index.read_bytes()
    prior_directory = index.parent / "generations" / prior_generation
    preview = tmp_path / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"prior-preview")
    event = threading.Event()
    publish = handlers._publish_generated_result

    def cancel_at_entry(*args, **kwargs):
        event.set()
        return publish(*args, **kwargs)

    with (
        patch("lyra.plotting.band.render_band", side_effect=_staged_band_render),
        patch("lyra.job_handlers._publish_generated_result", side_effect=cancel_at_entry),
    ):
        try:
            handlers.execute_job(
                service,
                Queue(),
                "rebuild-preview-band",
                payload,
                job_id=uuid.uuid4(),
                cancel_event=event,
            )
        except handlers.JobCancelled:
            pass
        else:
            raise AssertionError("publication-entry cancellation was not observed")

    assert index.read_bytes() == prior_index
    assert prior_directory.is_dir()
    assert preview.read_bytes() == b"prior-preview"


def test_cancellation_after_publication_rolls_back_before_cache_commit(tmp_path):
    handlers = _handlers()
    service, payload, index, prior_generation = _prime_projected_band_cache(tmp_path)
    prior_index = index.read_bytes()
    prior_directory = index.parent / "generations" / prior_generation
    preview = tmp_path / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"prior-preview")
    event = threading.Event()
    publish = handlers._publish_generated_result

    def cancel_after_publication(*args, **kwargs):
        published = publish(*args, **kwargs)
        event.set()
        return published

    with (
        patch("lyra.plotting.band.render_band", side_effect=_staged_band_render),
        patch(
            "lyra.job_handlers._publish_generated_result",
            side_effect=cancel_after_publication,
        ),
    ):
        try:
            handlers.execute_job(
                service,
                Queue(),
                "rebuild-preview-band",
                payload,
                job_id=uuid.uuid4(),
                cancel_event=event,
            )
        except handlers.JobCancelled:
            pass
        else:
            raise AssertionError("pre-commit cancellation was not observed")

    assert index.read_bytes() == prior_index
    assert prior_directory.is_dir()
    assert preview.read_bytes() == b"prior-preview"


def test_failed_cooperative_restore_retains_manifest_and_only_prior_backup(tmp_path):
    handlers = _handlers()
    destination = tmp_path / "output"
    staging = destination / ".lyra-job-test"
    staging.mkdir(parents=True)
    final = destination / "stable.png"
    final.write_bytes(b"prior-bytes")
    staged = staging / "stable.png"
    staged.write_bytes(b"new-bytes")
    result = GeneratedResult((staged,), staged, "done")
    real_replace = os.replace

    def fail_publish_then_restore(source, target):
        source = Path(source)
        target = Path(target)
        if source == staged:
            raise OSError("publish failed")
        if source == staging / ".prior" / "stable.png":
            raise OSError("restore failed")
        return real_replace(source, target)

    with patch("lyra.job_handlers.os.replace", side_effect=fail_publish_then_restore):
        try:
            handlers._publish_generated_result(result, staging, destination)
        except OSError as exc:
            assert "publish failed" in str(exc)
        else:
            raise AssertionError("publication failure was not propagated")

    assert not final.exists()
    assert (staging / ".publication.json").is_file()
    assert (staging / ".prior" / "stable.png").read_bytes() == b"prior-bytes"

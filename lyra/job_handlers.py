from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4


SCREEN_PREVIEW_DPI = 120
_PUBLICATION_COMMIT_NAME = ".publication-committed"
PREVIEW_BASENAMES = {
    "band": "lyra_preview_band",
    "dos": "lyra_preview_dos",
    "combined": "lyra_preview_combined",
    "fermi": "lyra_preview_fermi",
}


class JobCancelled(Exception):
    """Raised when a local job reaches a cooperative cancellation boundary."""


class _Emitter:
    def __init__(self, queue, job_id=None, cancel_event=None) -> None:
        self.queue = queue
        self.job_id = job_id
        self.cancel_event = cancel_event

    def checkpoint(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise JobCancelled("Lyra job cancelled.")

    def put(self, kind: str, payload) -> None:
        self.checkpoint()
        if self.job_id is None:
            self.queue.put((kind, payload))
        else:
            self.queue.put(
                {"job_id": self.job_id, "kind": kind, "payload": payload}
            )


def _job_parts(operation: str) -> tuple[str, str]:
    for prefix, intent in (
        ("rebuild-preview-", "rebuild-preview"),
        ("preview-", "preview"),
        ("export-", "export"),
    ):
        if operation.startswith(prefix):
            return intent, operation.removeprefix(prefix)
    return "export", operation


def _preview_settings(settings, operation: str):
    return replace(
        settings,
        export=replace(
            settings.export,
            basename=PREVIEW_BASENAMES[operation],
            formats=("png",),
            dpi=SCREEN_PREVIEW_DPI,
        ),
    )


def _preview_directory(payload: dict, operation: str) -> Path:
    if "_render_output" in payload:
        return Path(payload["_render_output"])
    folder = payload["band_folder"] if operation == "combined" else payload["folder"]
    return Path(folder).resolve() / ".lyra-cache" / "previews"


def _execute_job_impl(
    service,
    result_queue,
    operation: str,
    payload: dict,
    *,
    job_id=None,
    cancel_event=None,
):
    from lyra.models import ReadMode
    from lyra.plotting.band import render_band
    from lyra.plotting.combined import render_combined
    from lyra.plotting.dos import render_dos
    from lyra.plotting.fermi import render_fermi
    from lyra.quality import (
        assess_band,
        assess_combined,
        assess_dos,
        assess_fermi,
        write_quality_reports,
    )

    emitter = _Emitter(result_queue, job_id, cancel_event)
    emitter.checkpoint()
    active_phase: tuple[str, str | None, str | None] | None = None
    phase_started = 0.0

    def progress(event: dict, *, data_kind: str | None = None, folder=None) -> None:
        nonlocal active_phase, phase_started
        enriched = dict(event)
        phase = str(enriched.get("phase") or "unknown")
        now = time.monotonic()
        resolved_folder = str(Path(folder).resolve()) if folder is not None else None
        phase_context = (phase, data_kind, resolved_folder)
        if phase_context != active_phase:
            active_phase = phase_context
            phase_started = now
        enriched["elapsed_seconds"] = now - phase_started
        if data_kind is not None:
            enriched["data_kind"] = data_kind
        if resolved_folder is not None:
            enriched["folder"] = resolved_folder
        emitter.put("progress", enriched)

    def render_plot(
        renderer,
        args,
        phase: str,
        start_detail: str,
        complete_detail: str,
        *,
        data_kind: str | None = None,
        folder=None,
    ):
        progress(
            {"phase": phase, "current": 0, "total": 0, "detail": start_detail},
            data_kind=data_kind,
            folder=folder,
        )
        emitter.checkpoint()
        result = renderer(*args)
        emitter.checkpoint()
        progress(
            {"phase": phase, "current": 0, "total": 0, "detail": complete_detail},
            data_kind=data_kind,
            folder=folder,
        )
        return result

    def attach_quality(result, report, settings, intent):
        report_files = ()
        if intent == "export":
            report_files = write_quality_reports(
                report,
                payload["output"],
                settings.export.basename,
            )
        return replace(
            result,
            quality_report=report,
            report_files=report_files,
        )

    if operation == "clear-cache":
        service.clear(payload["folder"], payload.get("data_kind"))
        return {
            "folder": str(Path(payload["folder"]).resolve()),
            "message": "Projection cache cleared.",
        }

    if operation == "export-magnetic":
        from lyra.magnetism import read_magnetic_moments, write_vesta_magnetic_files

        progress(
            {
                "phase": "source-parse",
                "current": 0,
                "total": 0,
                "detail": "Reading final POSCAR structure and OUTCAR magnetic moments.",
            },
            data_kind="magnetic",
            folder=payload["folder"],
        )
        data = read_magnetic_moments(payload["folder"])
        progress(
            {
                "phase": "export",
                "current": 0,
                "total": 0,
                "detail": "Writing VESTA magnetic structure and evidence file.",
            },
            data_kind="magnetic",
            folder=payload["folder"],
        )
        outputs = write_vesta_magnetic_files(data, payload["output"])
        return {
            "output_files": outputs,
            "message": "VESTA magnetic files generated successfully.",
        }

    if operation == "prepare-paired-band":
        def paired_progress(event: dict) -> None:
            side = event.get("side", "left")
            folder = payload[f"{side}_folder"]
            progress(event, data_kind="band", folder=folder)

        return service.prepare_paired_band(
            payload["left_folder"],
            ReadMode(payload["left_mode"]),
            payload["right_folder"],
            ReadMode(payload["right_mode"]),
            paired_progress,
        )

    if operation == "prepare-band-zoom":
        mode = ReadMode(payload["mode"])
        folder = payload["folder"]
        report_progress = lambda event: progress(
            event,
            data_kind="band",
            folder=folder,
        )
        return service.prepare_band_zoom(
            folder,
            mode,
            payload["settings"],
            payload["identity"],
            report_progress,
        )

    if operation == "prepare-combined-zoom":
        settings = payload["settings"]
        band_folder = payload["band_folder"]
        dos_folder = payload["dos_folder"]
        band_progress = lambda event: progress(
            event, data_kind="band", folder=band_folder
        )
        dos_progress = lambda event: progress(
            event, data_kind="dos", folder=dos_folder
        )
        band_data = service.load_band(
            band_folder,
            ReadMode(payload["band_mode"]),
            settings.band_projection_channels,
            band_progress,
            identity=payload["band_identity"],
        )
        dos_data = service.load_dos(
            dos_folder,
            ReadMode(payload["dos_mode"]),
            settings.dos_projection_channels,
            dos_progress,
            identity=payload["dos_identity"],
        )
        return {
            "band_data": band_data,
            "dos_data": dos_data,
            "settings": settings,
        }

    if operation in {"prepare-band-inspector", "query-band-state"}:
        mode = ReadMode(payload["mode"])
        folder = payload["folder"]
        report_progress = lambda event: progress(
            event,
            data_kind="band",
            folder=folder,
        )
        if operation == "prepare-band-inspector":
            index, identity, display_data = service.prepare_band_inspector(
                folder,
                mode,
                report_progress,
                identity=payload.get("identity"),
                reference_energy=payload.get("reference_energy"),
                display_channels=tuple(payload.get("display_channels", ())),
            )
            return {
                "index": index,
                "identity": identity,
                "display_data": display_data,
                "settings": payload.get("settings"),
            }
        return service.query_band_state(
            folder,
            mode,
            payload["selection"],
            report_progress,
            identity=payload.get("identity"),
            whole_band=bool(payload.get("whole_band", False)),
            band_indices=payload.get("band_indices"),
        )

    original_operation = operation
    intent, operation = _job_parts(operation)
    requires_preview_identity = original_operation.startswith("export-")
    rebuild_kinds: set[str] = set()
    if intent == "rebuild-preview":
        if operation == "combined":
            requested_kind = payload.get("rebuild_data_kind")
            entries = (
                ((payload["band_folder"], "band"), (payload["dos_folder"], "dos"))
                if requested_kind is None
                else ((payload[f"{requested_kind}_folder"], requested_kind),)
            )
        else:
            entries = ((payload["folder"], operation),)
        rebuild_kinds = {data_kind for _folder, data_kind in entries}
        intent = "preview"

    mode = ReadMode(payload["mode"])
    band_read_mode = (
        ReadMode(payload.get("band_mode", mode.value))
        if operation == "combined"
        else mode
    )
    dos_read_mode = (
        ReadMode(payload.get("dos_mode", mode.value))
        if operation == "combined"
        else mode
    )
    output = _preview_directory(payload, operation) if intent == "preview" else payload["output"]

    def expected_identity(folder, data_kind, read_mode):
        resolved = Path(folder).resolve()
        for source_folder, identity in preview_sources:
            if (
                Path(source_folder).resolve() == resolved
                and identity.data_kind == data_kind
                and identity.read_mode == read_mode.value
            ):
                return identity
        return None

    preview_sources = tuple(payload.get("preview_sources", ()))
    required_sources = (
        ((payload["folder"], operation, mode),)
        if operation in {"band", "dos", "fermi"}
        else (
            (payload["band_folder"], "band", band_read_mode),
            (payload["dos_folder"], "dos", dos_read_mode),
        )
    )
    if requires_preview_identity and (
        not preview_sources
        or not all(
            (identity := expected_identity(folder, data_kind, read_mode)) is not None
            and service.identity_matches(folder, identity)
            for folder, data_kind, read_mode in required_sources
        )
    ):
        raise ValueError("Preview sources changed. Generate a new preview before exporting.")

    if operation == "band":
        settings = payload["settings"]
        channels = settings.projection_channels if settings.mode != "standard" else ()
        identity = expected_identity(payload["folder"], "band", mode) or service.band_identity(
            payload["folder"], mode, channels,
            lambda event: progress(event, data_kind="band", folder=payload["folder"]),
        )
        data = service.load_band(
            payload["folder"],
            mode,
            channels,
            lambda event: progress(event, data_kind="band", folder=payload["folder"]),
            identity=identity,
            rebuild="band" in rebuild_kinds,
        )
        render_phase = "preview-render" if intent == "preview" else "export"
        result = render_plot(
            render_band,
            (data, _preview_settings(settings, operation) if intent == "preview" else settings, output),
            render_phase,
            "Rendering Band preview." if intent == "preview" else "Writing Band export.",
            "Finished rendering Band preview." if intent == "preview" else "Finished writing Band export.",
            data_kind="band",
            folder=payload["folder"],
        )
        result = attach_quality(
            result,
            assess_band(data, payload["folder"]),
            settings,
            intent,
        )
        return replace(result, source_identities=((Path(payload["folder"]).resolve(), identity),))
    if operation == "dos":
        settings = payload["settings"]
        identity = expected_identity(payload["folder"], "dos", mode) or service.dos_identity(
            payload["folder"], mode, settings.projection_channels,
            lambda event: progress(event, data_kind="dos", folder=payload["folder"]),
        )
        data = service.load_dos(
            payload["folder"],
            mode,
            settings.projection_channels,
            lambda event: progress(event, data_kind="dos", folder=payload["folder"]),
            identity=identity,
            rebuild="dos" in rebuild_kinds,
        )
        render_phase = "preview-render" if intent == "preview" else "export"
        result = render_plot(
            render_dos,
            (data, _preview_settings(settings, operation) if intent == "preview" else settings, output),
            render_phase,
            "Rendering DOS preview." if intent == "preview" else "Writing DOS export.",
            "Finished rendering DOS preview." if intent == "preview" else "Finished writing DOS export.",
            data_kind="dos",
            folder=payload["folder"],
        )
        result = attach_quality(
            result,
            assess_dos(data, payload["folder"]),
            settings,
            intent,
        )
        return replace(result, source_identities=((Path(payload["folder"]).resolve(), identity),))
    if operation == "fermi":
        settings = payload["settings"]
        identity = expected_identity(
            payload["folder"],
            "fermi",
            mode,
        ) or service.fermi_identity(
            payload["folder"],
            mode,
            lambda event: progress(
                event,
                data_kind="fermi",
                folder=payload["folder"],
            ),
        )
        data = service.load_fermi(
            payload["folder"],
            mode,
            lambda event: progress(
                event,
                data_kind="fermi",
                folder=payload["folder"],
            ),
            identity=identity,
            rebuild="fermi" in rebuild_kinds,
        )
        render_phase = "preview-render" if intent == "preview" else "export"
        result = render_plot(
            render_fermi,
            (
                data,
                _preview_settings(settings, operation)
                if intent == "preview"
                else settings,
                output,
            ),
            render_phase,
            (
                "Rendering Fermi-surface preview."
                if intent == "preview"
                else "Writing Fermi-surface export."
            ),
            (
                "Finished rendering Fermi-surface preview."
                if intent == "preview"
                else "Finished writing Fermi-surface export."
            ),
            data_kind="fermi",
            folder=payload["folder"],
        )
        result = attach_quality(
            result,
            assess_fermi(data),
            settings,
            intent,
        )
        return replace(
            result,
            source_identities=((Path(payload["folder"]).resolve(), identity),),
        )
    if operation == "combined":
        settings = payload["settings"]
        band_channels = (
            settings.band_projection_channels
            if settings.band_mode != "standard"
            else ()
        )
        band_identity = expected_identity(payload["band_folder"], "band", band_read_mode) or service.band_identity(
            payload["band_folder"], band_read_mode, band_channels,
            lambda event: progress(event, data_kind="band", folder=payload["band_folder"]),
        )
        dos_identity = expected_identity(payload["dos_folder"], "dos", dos_read_mode) or service.dos_identity(
            payload["dos_folder"], dos_read_mode, settings.dos_projection_channels,
            lambda event: progress(event, data_kind="dos", folder=payload["dos_folder"]),
        )
        band = service.load_band(
            payload["band_folder"],
            band_read_mode,
            band_channels,
            lambda event: progress(event, data_kind="band", folder=payload["band_folder"]),
            identity=band_identity,
            rebuild="band" in rebuild_kinds,
        )
        dos = service.load_dos(
            payload["dos_folder"],
            dos_read_mode,
            settings.dos_projection_channels,
            lambda event: progress(event, data_kind="dos", folder=payload["dos_folder"]),
            identity=dos_identity,
            rebuild="dos" in rebuild_kinds,
        )
        render_phase = "preview-render" if intent == "preview" else "export"
        result = render_plot(
            render_combined,
            (band, dos, _preview_settings(settings, operation) if intent == "preview" else settings, output),
            render_phase,
            "Rendering combined preview." if intent == "preview" else "Writing combined export.",
            "Finished rendering combined preview." if intent == "preview" else "Finished writing combined export.",
        )
        result = attach_quality(
            result,
            assess_combined(band, dos),
            settings,
            intent,
        )
        return replace(
            result,
            source_identities=(
                (Path(payload["band_folder"]).resolve(), band_identity),
                (Path(payload["dos_folder"]).resolve(), dos_identity),
            ),
        )
    raise ValueError(f"Unknown worker operation: {operation}")


def _publication_marker(staging: Path, relative: Path) -> Path:
    return staging / ".publishing" / relative.parent / f"{relative.name}.marker"


def _publication_committed(staging: Path) -> bool:
    return (staging / _PUBLICATION_COMMIT_NAME).is_file()


def _mark_publication_committed(staging: Path) -> None:
    temporary = staging / f"{_PUBLICATION_COMMIT_NAME}.tmp"
    try:
        temporary.write_text("committed", encoding="ascii")
        os.replace(temporary, staging / _PUBLICATION_COMMIT_NAME)
    finally:
        temporary.unlink(missing_ok=True)


def _discard_publication_artifacts(staging: Path) -> None:
    shutil.rmtree(staging / ".prior", ignore_errors=True)
    shutil.rmtree(staging / ".publishing", ignore_errors=True)
    (staging / ".publication.json").unlink(missing_ok=True)
    (staging / _PUBLICATION_COMMIT_NAME).unlink(missing_ok=True)


def _rollback_publication_records(
    staging: Path,
    destination: Path,
    records,
    *,
    replace_file=None,
) -> bool:
    replace_file = replace_file or os.replace
    restored = True
    for record in reversed(tuple(records)):
        try:
            relative = Path(record["relative"])
            if relative.is_absolute() or ".." in relative.parts:
                restored = False
                continue
            final = destination / relative
            backup = staging / ".prior" / relative
            if (
                not _publication_marker(staging, relative).exists()
                and not backup.exists()
                and (bool(record["had_prior"]) or not final.exists())
            ):
                continue
            if backup.is_file():
                try:
                    final.unlink(missing_ok=True)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    replace_file(backup, final)
                except OSError:
                    restored = False
            elif not bool(record["had_prior"]):
                try:
                    final.unlink(missing_ok=True)
                except OSError:
                    restored = False
        except (KeyError, TypeError, ValueError):
            restored = False
    if restored:
        _discard_publication_artifacts(staging)
    return restored


def _rollback_staged_publication(
    staging: Path,
    destination: Path,
    *,
    replace_file=None,
) -> bool:
    manifest = staging / ".publication.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload["files"]
        if not isinstance(records, list):
            raise TypeError("publication records must be a list")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        backup_root = staging / ".prior"
        return not backup_root.exists() or not any(backup_root.rglob("*"))
    return _rollback_publication_records(
        staging,
        destination,
        records,
        replace_file=replace_file,
    )


def _publish_generated_result(
    result,
    staging: Path,
    destination: Path,
    *,
    checkpoint=None,
    retain_backups: bool = False,
):
    sources = tuple(
        dict.fromkeys((*result.output_files, *result.report_files, result.preview_file))
    )
    replacements: dict[Path, Path] = {}
    published_records: list[dict] = []
    backup_root = staging / ".prior"
    manifest = staging / ".publication.json"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        publication_files = []
        for source in sources:
            source = Path(source).resolve()
            try:
                relative = source.relative_to(staging.resolve())
            except ValueError as exc:
                raise RuntimeError("Renderer returned a file outside its job staging directory.") from exc
            publication_files.append(
                {"relative": relative.as_posix(), "had_prior": (destination / relative).exists()}
            )
        manifest.write_text(
            json.dumps({"files": publication_files}, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        if checkpoint is not None:
            checkpoint()
        for source, publication in zip(sources, publication_files):
            source = Path(source).resolve()
            relative = Path(publication["relative"])
            final = destination / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            marker = _publication_marker(staging, relative)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("publishing", encoding="ascii")
            published_records.append(publication)
            if final.exists():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final, backup)
            os.replace(source, final)
            replacements[source] = final
    except BaseException:
        _rollback_publication_records(
            staging,
            destination,
            published_records,
        )
        raise
    if not retain_backups:
        _discard_publication_artifacts(staging)
    return replace(
        result,
        output_files=tuple(replacements[Path(path).resolve()] for path in result.output_files),
        preview_file=replacements[Path(result.preview_file).resolve()],
        report_files=tuple(replacements[Path(path).resolve()] for path in result.report_files),
    )


def execute_job(
    service,
    result_queue,
    operation: str,
    payload: dict,
    *,
    job_id=None,
    cancel_event=None,
):
    intent, plot_kind = _job_parts(operation)
    is_plot_job = plot_kind in PREVIEW_BASENAMES
    rebuild_entries = ()
    if intent == "rebuild-preview":
        if plot_kind == "combined":
            requested = payload.get("rebuild_data_kind")
            rebuild_entries = (
                ((payload["band_folder"], "band"), (payload["dos_folder"], "dos"))
                if requested is None
                else ((payload[f"{requested}_folder"], requested),)
            )
        else:
            rebuild_entries = ((payload["folder"], plot_kind),)
    rebuild_tokens = tuple(
        (
            folder,
            data_kind,
            service.begin_rebuild(folder, data_kind),
        )
        for folder, data_kind in rebuild_entries
        if hasattr(service, "begin_rebuild")
    )

    def finish_rebuilds(success: bool) -> None:
        for folder, data_kind, previous in rebuild_tokens:
            service.finish_rebuild(
                folder, data_kind, previous, success=success
            )

    def promote_rebuilds() -> None:
        for folder, data_kind, _previous in rebuild_tokens:
            if hasattr(service, "promote_rebuild"):
                service.promote_rebuild(folder, data_kind)

    if job_id is None or not is_plot_job:
        try:
            result = _execute_job_impl(
                service,
                result_queue,
                operation,
                payload,
                job_id=job_id,
                cancel_event=cancel_event,
            )
        except BaseException:
            finish_rebuilds(False)
            raise
        finish_rebuilds(True)
        return result

    destination = (
        _preview_directory(payload, plot_kind)
        if intent in {"preview", "rebuild-preview"}
        else Path(payload["output"])
    )
    staging = destination / f".lyra-job-{job_id or uuid4()}"
    staged_payload = dict(payload)
    staged_payload["output"] = str(staging)
    staged_payload["_render_output"] = str(staging)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    preserve_staging = False
    try:
        result = _execute_job_impl(
            service,
            result_queue,
            operation,
            staged_payload,
            job_id=job_id,
            cancel_event=cancel_event,
        )
        _Emitter(result_queue, job_id, cancel_event).checkpoint()
        published = _publish_generated_result(
            result,
            staging,
            destination,
            checkpoint=_Emitter(result_queue, job_id, cancel_event).checkpoint,
            retain_backups=True,
        )
        _Emitter(result_queue, job_id, cancel_event).checkpoint()
        promote_rebuilds()
        _mark_publication_committed(staging)
        finish_rebuilds(True)
        _discard_publication_artifacts(staging)
        return published
    except BaseException:
        if _publication_committed(staging):
            try:
                finish_rebuilds(True)
                _discard_publication_artifacts(staging)
            except BaseException:
                preserve_staging = True
        else:
            try:
                finish_rebuilds(False)
            finally:
                preserve_staging = not _rollback_staged_publication(
                    staging, destination
                )
        raise
    finally:
        if not preserve_staging:
            shutil.rmtree(staging, ignore_errors=True)

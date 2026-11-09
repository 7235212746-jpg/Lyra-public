from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

from lyra.cache import ProjectionCache
from lyra.data_service import ProjectionDataService
from lyra.models import (
    BandPlotSettings,
    DosPlotSettings,
    ExportSettings,
    PlotRange,
    ProjectionChannel,
    ReadMode,
)
from lyra.remote.protocol import (
    Artifact,
    JobProgress,
    RemoteRequest,
    RemoteResult,
    RemoteStatus,
    RemoteTaskState,
    encode_message,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ELEMENT = re.compile(r"^[A-Z][a-z]?$|^__all_projection_data__$")
_ORBITAL = re.compile(r"^[A-Za-z0-9_()+-]{1,32}$")
MAX_PREVIEW_BYTES = 32 * 1024 * 1024


class RemoteTaskCancelled(Exception):
    pass


def cancellation_requested(job_directory: Path, token: str) -> bool:
    path = job_directory / "cancel.json"
    try:
        if path.stat().st_size > 64 * 1024:
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("cancel_token") == token
    )


def _number(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} is outside its supported range.")
    return result


def _range(options: dict) -> PlotRange:
    value = options.get("energy_range", (-5.0, 5.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("energy_range must contain two values.")
    result = PlotRange(
        _number(value[0], "energy minimum", -1000.0, 1000.0),
        _number(value[1], "energy maximum", -1000.0, 1000.0),
    )
    result.validate()
    return result


def _channels(options: dict) -> tuple[ProjectionChannel, ...]:
    values = options.get("projection_channels", ())
    if not isinstance(values, (list, tuple)) or len(values) > 12:
        raise ValueError("projection_channels must be a bounded list.")
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) - {"element", "orbital", "colour"}:
            raise ValueError("Projection channel fields are invalid.")
        element = value.get("element")
        orbital = value.get("orbital")
        colour = value.get("colour", "#3974d8")
        if not isinstance(element, str) or not _ELEMENT.fullmatch(element):
            raise ValueError("Projection element is invalid.")
        if orbital is not None and (not isinstance(orbital, str) or not _ORBITAL.fullmatch(orbital)):
            raise ValueError("Projection orbital is invalid.")
        if not isinstance(colour, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", colour):
            raise ValueError("Projection colour is invalid.")
        result.append(ProjectionChannel(element, orbital, colour))
    return tuple(result)


def _export(options: dict, basename: str) -> ExportSettings:
    return ExportSettings(
        basename=basename,
        formats=("png",),
        dpi=120,
        width_mm=_number(options.get("width_mm", 180.0), "width_mm", 50.0, 500.0),
        font_family="Arial",
        font_size=_number(options.get("font_size", 13.0), "font_size", 6.0, 72.0),
        line_width=_number(options.get("line_width", 1.0), "line_width", 0.1, 10.0),
    )


def build_band_settings(options: dict) -> BandPlotSettings:
    if not isinstance(options, dict):
        raise ValueError("Remote Band options must be an object.")
    channels = _channels(options)
    mode = options.get("mode", "standard")
    if mode not in {"standard", "fatband", "rgb"}:
        raise ValueError("Remote Band mode is unsupported.")
    if mode == "rgb" and len(channels) > 3:
        raise ValueError("RGB Band accepts at most three channels.")
    return BandPlotSettings(
        energy_range=_range(options),
        reference_energy=None,
        mode=mode,
        projection_channels=channels,
        complete_band_colour=options.get("complete_band_colour", "#000000"),
        fat_scale=_number(options.get("fat_scale", 120.0), "fat_scale", 1.0, 2000.0),
        export=_export(options, "lyra_remote_band"),
    )


def build_dos_settings(options: dict) -> DosPlotSettings:
    if not isinstance(options, dict):
        raise ValueError("Remote DOS options must be an object.")
    return DosPlotSettings(
        energy_range=_range(options),
        reference_energy=None,
        projection_channels=_channels(options),
        plot_total=bool(options.get("plot_total", True)),
        export=_export(options, "lyra_remote_dos"),
    )


def cache_namespace(cache_root: str | Path, fingerprint: str) -> Path:
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise ValueError("Remote manifest fingerprint is invalid.")
    return Path(cache_root).expanduser().resolve() / "data" / fingerprint


def _atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o600)
    temporary.replace(path)


def execute(request: RemoteRequest, job_directory: Path, cache_root: Path) -> RemoteResult:
    import matplotlib
    matplotlib.use("Agg")
    from lyra.plotting.band import render_band
    from lyra.plotting.dos import render_dos

    source = Path(request.source_paths[0]).resolve()
    fingerprint = request.options.get("manifest_fingerprint")
    namespace = cache_namespace(cache_root, fingerprint)
    namespace.mkdir(parents=True, exist_ok=True)
    service = ProjectionDataService(cache=ProjectionCache(root=namespace))
    artifact_dir = job_directory / "artifacts"
    artifact_dir.mkdir(mode=0o700, exist_ok=True)
    sequence = 0

    def progress(event: dict) -> None:
        nonlocal sequence
        if cancellation_requested(job_directory, request.cancel_token):
            raise RemoteTaskCancelled("Remote task cancelled by user.")
        sequence += 1
        _atomic(
            job_directory / "status.json",
            encode_message(
                RemoteStatus(
                    request.job_id,
                    sequence,
                    RemoteTaskState.RUNNING,
                    time.time(),
                    JobProgress(
                        str(event.get("phase", "analysis")),
                        str(event.get("detail", "")),
                        int(event.get("current", 0)),
                        int(event.get("total", 0)),
                    ),
                    None,
                )
            ),
        )

    workflow = request.options.get("workflow", request.task)
    progress({"phase": "source-identity", "detail": "Validated remote calculation source.", "current": 0, "total": 0})
    if workflow in {"band", "projected_band"}:
        settings = build_band_settings(dict(request.options.get("plot", {})))
        channels = settings.projection_channels
        if workflow == "projected_band" and not channels:
            _index, _identity, data = service.prepare_band_inspector(
                source, ReadMode.LITE, progress
            )
        else:
            data = service.load_band(source, ReadMode.LITE, channels, progress)
        generated = render_band(data, settings, artifact_dir)
    elif workflow == "dos":
        settings = build_dos_settings(dict(request.options.get("plot", {})))
        data = service.load_dos(
            source, ReadMode.LITE, settings.projection_channels, progress
        )
        generated = render_dos(data, settings, artifact_dir)
    else:
        raise ValueError("Remote scientific workflow is unsupported.")
    progress({"phase": "preview-rendered", "detail": "Remote preview rendering is complete.", "current": 0, "total": 0})
    preview = Path(generated.preview_file).resolve()
    size = preview.stat().st_size
    if size > MAX_PREVIEW_BYTES or not preview.is_relative_to(artifact_dir.resolve()):
        raise ValueError("Remote preview artifact is invalid or too large.")
    digest = hashlib.sha256(preview.read_bytes()).hexdigest()
    artifact = Artifact(
        "preview",
        preview.as_posix(),
        preview.name,
        size,
        digest,
        "image/png",
    )
    return RemoteResult(
        request.job_id,
        True,
        {"workflow": workflow, "cache_namespace": namespace.as_posix()},
        (artifact,),
    )

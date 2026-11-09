"""Smoke-test Full Mode DOS caching with a real vasprun.xml folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmark_projection_cache import (
    _MemorySampler,
    _directory_size,
    _forbidden_loader,
    _memory_bytes,
)
from lyra.data_service import ProjectionDataService
from lyra.models import DosPlotSettings, ExportSettings, PlotRange, ProjectionChannel
from lyra.plotting.dos import render_dos


def _run_case(service, folder: Path, name: str) -> dict[str, object]:
    channel = (ProjectionChannel("Ta", "d", "#3974d8"),)
    events: list[dict] = []
    started = time.perf_counter()
    data = service.load_dos(folder, "full", channel, events.append)
    loaded = time.perf_counter()
    result = render_dos(
        data,
        DosPlotSettings(
            energy_range=PlotRange(-5.0, 5.0),
            projection_channels=channel,
            export=ExportSettings(
                basename=name,
                formats=("png",),
                dpi=120,
                width_mm=90.0,
                font_family="Arial",
            ),
        ),
        folder / ".lyra-cache" / "previews",
    )
    rendered = time.perf_counter()
    return {
        "case": name,
        "data_seconds": round(loaded - started, 6),
        "preview_seconds": round(rendered - loaded, 6),
        "total_seconds": round(rendered - started, 6),
        "preview_file": str(result.preview_file),
        "preview_bytes": result.preview_file.stat().st_size,
        "phases": [event.get("phase") for event in events],
        "elements": data.elements,
        "orbitals": data.orbitals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    folder = args.folder.resolve()
    cache_root = folder / ".lyra-cache"
    if cache_root.exists():
        parser.error(f"Cold smoke test requires no existing cache: {cache_root}")

    source = next(
        path for path in (folder / "vasprun.xml", folder / "vasprun.xml.gz") if path.is_file()
    )
    source_before = source.stat()
    baseline_working_set, baseline_private = _memory_bytes()
    with _MemorySampler() as sampler:
        cold = _run_case(ProjectionDataService(), folder, "full_dos_cold")

    warm_service = ProjectionDataService(vasprun_dos_loader=_forbidden_loader)
    warm = _run_case(warm_service, folder, "full_dos_warm_restart")
    source_after = source.stat()
    print(
        json.dumps(
            {
                "folder": str(folder),
                "source": {
                    "path": str(source),
                    "size": source_before.st_size,
                    "mtime_ns_before": source_before.st_mtime_ns,
                    "mtime_ns_after": source_after.st_mtime_ns,
                    "unchanged": (
                        source_before.st_size == source_after.st_size
                        and source_before.st_mtime_ns == source_after.st_mtime_ns
                    ),
                },
                "cache_bytes": _directory_size(cache_root),
                "baseline_working_set_bytes": baseline_working_set,
                "baseline_private_bytes": baseline_private,
                "peak_working_set_bytes": sampler.peak_working_set,
                "peak_private_bytes": sampler.peak_private,
                "cases": [cold, warm],
                "warm_vasprun_parser_forbidden": True,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

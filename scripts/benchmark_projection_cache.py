"""Benchmark Lyra's real projected-band cold and warm cache paths."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import threading
import time

from lyra.data_service import ProjectionDataService
from lyra.models import BandPlotSettings, ExportSettings, PlotRange, ProjectionChannel
from lyra.plotting.band import render_band


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
_KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
_PSAPI.GetProcessMemoryInfo.argtypes = (
    wintypes.HANDLE,
    ctypes.c_void_p,
    wintypes.DWORD,
)
_PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    )


def _memory_bytes() -> tuple[int, int]:
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    process = _KERNEL32.GetCurrentProcess()
    ok = _PSAPI.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError()
    return counters.working_set_size, counters.private_usage


class _MemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak_working_set = 0
        self.peak_private = 0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            working_set, private = _memory_bytes()
            self.peak_working_set = max(self.peak_working_set, working_set)
            self.peak_private = max(self.peak_private, private)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        working_set, private = _memory_bytes()
        self.peak_working_set = max(self.peak_working_set, working_set)
        self.peak_private = max(self.peak_private, private)


def _forbidden_loader(*_args, **_kwargs):
    raise AssertionError("A warm cache path attempted to reopen a source parser.")


def _directory_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def _settings(name: str, mode: str, channels: tuple[ProjectionChannel, ...]):
    return BandPlotSettings(
        energy_range=PlotRange(-5.0, 5.0),
        mode=mode,
        projection_channels=channels,
        export=ExportSettings(
            basename=name,
            formats=("png",),
            dpi=120,
            width_mm=180.0,
            font_family="Arial",
        ),
    )


def _run_case(
    service: ProjectionDataService,
    folder: Path,
    preview_directory: Path,
    name: str,
    mode: str,
    channels: tuple[ProjectionChannel, ...],
) -> dict[str, object]:
    events: list[dict] = []
    case_started = time.perf_counter()
    data = service.load_band(folder, "lite", channels, events.append)
    data_ready = time.perf_counter()
    result = render_band(data, _settings(name, mode, channels), preview_directory)
    rendered = time.perf_counter()
    return {
        "case": name,
        "data_seconds": round(data_ready - case_started, 6),
        "preview_seconds": round(rendered - data_ready, 6),
        "total_seconds": round(rendered - case_started, 6),
        "preview_file": str(result.preview_file),
        "preview_bytes": result.preview_file.stat().st_size,
        "phases": [event.get("phase") for event in events],
        "last_parse_event": next(
            (event for event in reversed(events) if event.get("phase") == "parse"), None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    folder = args.folder.resolve()
    cache_root = folder / ".lyra-cache"
    if cache_root.exists():
        parser.error(
            f"Cold benchmark requires no existing cache: {cache_root}. "
            "Clear it explicitly before rerunning."
        )

    procar = folder / "PROCAR"
    procar_before = procar.stat()
    preview_directory = cache_root / "previews"
    cold_service = ProjectionDataService()

    ta_d = (ProjectionChannel("Ta", "d", "#3974d8"),)
    pt_px = (ProjectionChannel("Pt", "px", "#d94841"),)
    rgb = (
        ProjectionChannel("Pt", None, "#3974d8"),
        ProjectionChannel("Ga", None, "#d94841"),
        ProjectionChannel("Ta", None, "#1f9d72"),
    )

    baseline_working_set, baseline_private = _memory_bytes()
    with _MemorySampler() as sampler:
        cold = _run_case(
            cold_service, folder, preview_directory, "cold_ta_d", "fat", ta_d
        )

    warm_service = ProjectionDataService(
        eigenval_loader=_forbidden_loader,
        procar_loader=_forbidden_loader,
    )
    warm_restart = _run_case(
        warm_service, folder, preview_directory, "warm_restart_pt_px", "fat", pt_px
    )
    warm_memory = _run_case(
        warm_service, folder, preview_directory, "warm_memory_ta_d", "fat", ta_d
    )
    warm_rgb = _run_case(
        warm_service, folder, preview_directory, "warm_memory_rgb", "rgb", rgb
    )

    procar_after = procar.stat()
    summary = {
        "folder": str(folder),
        "procar": {
            "size": procar_before.st_size,
            "mtime_ns_before": procar_before.st_mtime_ns,
            "mtime_ns_after": procar_after.st_mtime_ns,
            "unchanged": (
                procar_before.st_size == procar_after.st_size
                and procar_before.st_mtime_ns == procar_after.st_mtime_ns
            ),
        },
        "cache_bytes": _directory_size(cache_root),
        "baseline_working_set_bytes": baseline_working_set,
        "baseline_private_bytes": baseline_private,
        "peak_working_set_bytes": sampler.peak_working_set,
        "peak_private_bytes": sampler.peak_private,
        "cases": [cold, warm_restart, warm_memory, warm_rgb],
        "warm_source_parsers_forbidden": True,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

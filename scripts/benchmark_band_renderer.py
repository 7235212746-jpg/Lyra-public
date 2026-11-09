"""Run the deterministic in-memory 900 x 344 projected-band renderer benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lyra.models import (
    BandPlotSettings,
    ExportSettings,
    ParsedBandData,
    PlotRange,
    ProjectionChannel,
    SpinKind,
)
from lyra.plotting.band import build_band_figure


def build_case(dpi: int) -> tuple[ParsedBandData, BandPlotSettings]:
    nkpoints = 900
    nbands = 344
    distances = np.linspace(0.0, 18.0, nkpoints, dtype=np.float64)
    kpoint_index = np.arange(nkpoints, dtype=np.float64)[:, None]
    band_index = np.arange(nbands, dtype=np.float64)[None, :]
    projections = {
        "X:p": (0.25 + 0.20 * np.sin(kpoint_index / 29.0 + band_index / 23.0)).astype(np.float32),
        "Y:d": (0.30 + 0.20 * np.cos(kpoint_index / 31.0 + band_index / 17.0)).astype(np.float32),
        "Z:s": (0.35 + 0.15 * np.sin(kpoint_index / 41.0 - band_index / 13.0)).astype(np.float32),
    }
    data = ParsedBandData(
        folder=Path("."),
        kpoints=np.zeros((nkpoints, 3), dtype=np.float64),
        distances=distances,
        segment_ranges=tuple((start, start + 100) for start in range(0, nkpoints, 100)),
        tick_positions=tuple(float(distances[index]) for index in range(0, nkpoints, 100)),
        tick_labels=tuple(f"K{index}" for index in range(9)),
        energies={
            "total": 4.75
            + 0.012 * band_index
            + 0.14 * np.sin(kpoint_index / 37.0 + band_index / 19.0)
        },
        efermi=5.0,
        spin_kind=SpinKind.NON_SPIN,
        elements=("X", "Y", "Z"),
        projections={key: {"total": value} for key, value in projections.items()},
    )
    settings = BandPlotSettings(
        energy_range=PlotRange(-2.0, 4.5),
        mode="rgb",
        projection_channels=(
            ProjectionChannel("X", "p", "#d94841"),
            ProjectionChannel("Y", "d", "#2457d6"),
            ProjectionChannel("Z", "s", "#3d9b68"),
        ),
        export=ExportSettings(
            basename="task5-benchmark",
            formats=("png",),
            dpi=dpi,
            width_mm=180.0,
            font_family="DejaVu Sans",
        ),
    )
    return data, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be at least one")

    data, settings = build_case(args.dpi)
    with tempfile.TemporaryDirectory() as temporary_directory:
        png_path = Path(temporary_directory) / "benchmark.png"
        for pass_number in range(1, args.passes + 1):
            started = time.perf_counter()
            figure = build_band_figure(data, settings)
            built = time.perf_counter()
            figure.savefig(png_path, dpi=settings.export.dpi, bbox_inches="tight")
            encoded = time.perf_counter()
            axes = figure.axes[0]
            print(
                json.dumps(
                    {
                        "pass": pass_number,
                        "build_seconds": round(built - started, 6),
                        "png_seconds": round(encoded - built, 6),
                        "png_bytes": png_path.stat().st_size,
                        "line_collections": sum(
                            type(collection).__name__ == "LineCollection"
                            for collection in axes.collections
                        ),
                        "segments": sum(
                            len(collection.get_segments())
                            for collection in axes.collections
                            if type(collection).__name__ == "LineCollection"
                        ),
                    },
                    sort_keys=True,
                )
            )
            plt.close(figure)


if __name__ == "__main__":
    main()

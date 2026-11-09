from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from lyra.models import ParsedBandData


@dataclass(frozen=True)
class BandPoint:
    spin: str
    kpoint_index: int
    band_index: int
    distance: float
    energy: float

    def __post_init__(self) -> None:
        if not isinstance(self.spin, str) or not self.spin:
            raise ValueError("Band point spin must be non-empty.")
        if self.kpoint_index < 0 or self.band_index < 0:
            raise ValueError("Band point indices must be non-negative.")
        if not math.isfinite(self.distance) or not math.isfinite(self.energy):
            raise ValueError("Band point coordinates must be finite.")


@dataclass(frozen=True)
class SelectedGapMeasurement:
    lower: BandPoint
    upper: BandPoint
    gap: float
    is_direct: bool


def nearest_band_point(
    data: ParsedBandData,
    reference_energy: float,
    *,
    click_display: tuple[float, float],
    transform: Callable[[np.ndarray], np.ndarray],
    max_distance_pixels: float = 18.0,
) -> BandPoint | None:
    if not math.isfinite(reference_energy):
        raise ValueError("Band reference energy must be finite.")
    if not math.isfinite(max_distance_pixels) or max_distance_pixels <= 0:
        raise ValueError("Snap radius must be positive and finite.")
    click = np.asarray(click_display, dtype=float)
    if click.shape != (2,) or not np.all(np.isfinite(click)):
        return None
    best: BandPoint | None = None
    best_distance = float(max_distance_pixels)
    distances = np.asarray(data.distances, dtype=float)
    for spin, raw_energies in data.energies.items():
        energies = np.asarray(raw_energies, dtype=float)
        if energies.ndim != 2 or energies.shape[0] != len(distances):
            continue
        shifted = energies - reference_energy
        for band_index in range(shifted.shape[1]):
            points = np.column_stack((distances, shifted[:, band_index]))
            finite = np.all(np.isfinite(points), axis=1)
            if not np.any(finite):
                continue
            transformed = np.asarray(transform(points[finite]), dtype=float)
            if transformed.shape != (int(np.count_nonzero(finite)), 2):
                raise ValueError("Band display transform returned an invalid shape.")
            pixel_distances = np.linalg.norm(transformed - click, axis=1)
            local_index = int(np.argmin(pixel_distances))
            pixel_distance = float(pixel_distances[local_index])
            if pixel_distance > best_distance:
                continue
            kpoint_index = int(np.flatnonzero(finite)[local_index])
            best_distance = pixel_distance
            best = BandPoint(
                str(spin),
                kpoint_index,
                band_index,
                float(distances[kpoint_index]),
                float(shifted[kpoint_index, band_index]),
            )
    return best


def measure_selected_gap(
    first: BandPoint,
    second: BandPoint,
    kpoints: np.ndarray,
    *,
    kpoint_tolerance: float = 1e-8,
) -> SelectedGapMeasurement:
    lower, upper = sorted((first, second), key=lambda point: point.energy)
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Band k-points must be an N x 3 array.")
    if max(lower.kpoint_index, upper.kpoint_index) >= len(points):
        raise ValueError("Selected Band point is outside the k-point array.")
    delta = points[upper.kpoint_index] - points[lower.kpoint_index]
    delta -= np.rint(delta)
    direct = bool(np.max(np.abs(delta)) <= kpoint_tolerance)
    return SelectedGapMeasurement(
        lower=lower,
        upper=upper,
        gap=float(upper.energy - lower.energy),
        is_direct=direct,
    )


def format_gap_measurement(measurement: SelectedGapMeasurement) -> str:
    kind = "直接" if measurement.is_direct else "间接"
    low = measurement.lower
    high = measurement.upper
    return (
        f"所选两点能量差：{measurement.gap:.6f} eV（{kind}） | "
        f"低点 {low.spin} / Band {low.band_index + 1} / k {low.kpoint_index + 1}: "
        f"{low.energy:.6f} eV | "
        f"高点 {high.spin} / Band {high.band_index + 1} / k {high.kpoint_index + 1}: "
        f"{high.energy:.6f} eV"
    )

from __future__ import annotations

import numpy as np

from lyra.band_gap_measurement import (
    BandPoint,
    format_gap_measurement,
    measure_selected_gap,
    nearest_band_point,
)
from tests.test_projection_store import parsed_band_data


def test_nearest_band_point_snaps_to_real_kpoint_band_and_spin() -> None:
    data = parsed_band_data()
    reference = data.efermi
    target_energy = float(data.energies["down"][1, 2] - reference)

    point = nearest_band_point(
        data,
        reference,
        click_display=(float(data.distances[1] * 100 + 2), target_energy * 100 + 1),
        transform=lambda values: np.asarray(values) * 100,
        max_distance_pixels=8,
    )

    assert point is not None
    assert point.spin == "down"
    assert point.kpoint_index == 1
    assert point.band_index == 2
    assert point.energy == target_energy


def test_nearest_band_point_returns_none_outside_visual_snap_radius() -> None:
    data = parsed_band_data()

    assert nearest_band_point(
        data,
        data.efermi,
        click_display=(10_000, 10_000),
        transform=lambda values: np.asarray(values) * 100,
        max_distance_pixels=12,
    ) is None


def test_measurement_orders_points_and_uses_fractional_kpoint_for_directness() -> None:
    data = parsed_band_data()
    low = BandPoint("up", 0, 1, float(data.distances[0]), -0.25)
    high = BandPoint("down", 0, 2, float(data.distances[0]), 0.40)

    result = measure_selected_gap(high, low, data.kpoints)

    assert result.lower == low
    assert result.upper == high
    assert result.gap == 0.65
    assert result.is_direct is True


def test_duplicate_path_indices_at_same_k_coordinate_are_still_direct() -> None:
    kpoints = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    first = BandPoint("total", 0, 0, 0.0, -0.1)
    second = BandPoint("total", 1, 1, 1.0, 0.2)

    result = measure_selected_gap(first, second, kpoints)

    assert result.is_direct is True


def test_measurement_text_states_selected_scope_and_one_based_indices() -> None:
    kpoints = np.array(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
    result = measure_selected_gap(
        BandPoint("up", 0, 4, 0.0, -0.3),
        BandPoint("down", 1, 7, 1.0, 0.6),
        kpoints,
    )

    text = format_gap_measurement(result)

    assert "所选两点能量差" in text
    assert "0.900000 eV" in text
    assert "间接" in text
    assert "Band 5" in text
    assert "Band 8" in text

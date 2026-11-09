from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lyra.band_edge_analysis import analyze_band_edges, format_band_edge_summary
from lyra.models import ParsedBandData, SpinKind


def band_data(
    energies: dict[str, np.ndarray],
    occupations: dict[str, np.ndarray],
    *,
    efermi: float = 0.5,
) -> ParsedBandData:
    point_count = next(iter(energies.values())).shape[0]
    return ParsedBandData(
        folder=Path("calculation"),
        kpoints=np.array([(index / max(1, point_count), 0.0, 0.0) for index in range(point_count)]),
        distances=np.arange(point_count, dtype=float),
        segment_ranges=((0, point_count),),
        tick_positions=(0.0, float(point_count - 1)),
        tick_labels=("G", "X"),
        energies=energies,
        occupations=occupations,
        efermi=efermi,
        spin_kind=SpinKind.COLLINEAR if len(energies) == 2 else SpinKind.NON_SPIN,
        elements=("X",),
    )


def test_occupation_analysis_finds_indirect_global_edges_and_real_indices():
    data = band_data(
        {"total": np.array([[-1.0, 1.0], [0.0, 2.0]])},
        {"total": np.array([[2.0, 0.0], [2.0, 0.0]])},
    )

    result = analyze_band_edges(data)

    assert result.classification == "semiconductor"
    assert result.method == "occupations"
    assert result.gap == 1.0
    assert result.is_direct is False
    assert "布里渊区" in result.warnings[0]
    assert (result.vbm.spin, result.vbm.kpoint_index, result.vbm.band_index) == ("total", 1, 0)
    assert (result.cbm.spin, result.cbm.kpoint_index, result.cbm.band_index) == ("total", 0, 1)
    assert "间接带隙" in format_band_edge_summary(result)


def test_non_spin_eigenval_with_unit_full_occupation_is_not_false_metal():
    data = band_data(
        {"total": np.array([[-1.0, 1.0], [0.0, 2.0]])},
        {"total": np.array([[1.0, 0.0], [1.0, 0.0]])},
    )

    result = analyze_band_edges(data)

    assert result.classification == "semiconductor"
    assert result.channels[0].full_occupation == 1.0


def test_partial_occupation_without_fermi_crossing_is_smearing_ambiguous():
    data = band_data(
        {"total": np.array([[-1.0, 0.2, 1.0], [-0.8, 0.4, 1.2]])},
        {"total": np.array([[2.0, 0.7, 0.0], [2.0, 0.2, 0.0]])},
    )

    result = analyze_band_edges(data)

    assert result.classification == "unavailable"
    assert result.gap is None
    assert result.channels[0].partial_state_count == 2
    assert result.channels[0].status == "ambiguous"
    assert "展宽" in result.channels[0].evidence


def test_partial_occupation_with_continuous_fermi_crossing_is_metallic():
    data = band_data(
        {"total": np.array([[-1.0, 0.2, 1.0], [-0.8, 0.8, 1.2]])},
        {"total": np.array([[1.0, 0.7, 0.0], [1.0, 0.2, 0.0]])},
    )

    result = analyze_band_edges(data)

    assert result.classification == "metal"
    assert result.channels[0].status == "metallic"
    assert "穿越 E-fermi" in result.channels[0].evidence


def test_one_metallic_spin_and_one_gapped_spin_is_half_metal():
    data = band_data(
        {
            "up": np.array([[-1.0, 0.2, 1.0], [-0.8, 0.8, 1.2]]),
            "down": np.array([[-1.2, 1.1, 2.0], [-0.2, 0.8, 2.2]]),
        },
        {
            "up": np.array([[1.0, 0.5, 0.0], [1.0, 0.2, 0.0]]),
            "down": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        },
    )

    result = analyze_band_edges(data)

    assert result.classification == "half_metal"
    assert {channel.spin: channel.status for channel in result.channels} == {
        "up": "metallic",
        "down": "gapped",
    }
    assert result.gap is None


def test_degenerate_edge_candidates_report_direct_when_any_kpoint_matches():
    data = band_data(
        {
            "total": np.array(
                [
                    [0.0, 2.0],
                    [-0.5, 1.0],
                    [0.0, 1.0],
                ]
            )
        },
        {
            "total": np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            )
        },
    )

    result = analyze_band_edges(data)

    assert result.classification == "semiconductor"
    assert result.is_direct is True
    assert result.vbm.kpoint_index == result.cbm.kpoint_index == 2


def test_fermi_sign_change_across_disconnected_segments_is_not_a_crossing():
    data = replace(
        band_data(
            {"total": np.array([[-1.0], [1.0]])},
            {"total": np.array([[1.0], [0.0]])},
        ),
        segment_ranges=((0, 1), (1, 2)),
    )

    result = analyze_band_edges(data)

    assert result.classification == "semiconductor"
    assert result.channels[0].status == "gapped"


def test_missing_or_degenerate_occupations_are_not_replaced_by_fermi_guessing():
    energies = {"total": np.array([[-1.0, 1.0], [0.0, 2.0]])}
    missing = band_data(energies, {})
    degenerate = replace(missing, occupations={"total": np.zeros((2, 2))})

    missing_result = analyze_band_edges(missing)
    degenerate_result = analyze_band_edges(degenerate)

    assert missing_result.classification == "unavailable"
    assert degenerate_result.classification == "unavailable"
    assert "占据数" in format_band_edge_summary(missing_result)
    assert "全为零" in format_band_edge_summary(degenerate_result)

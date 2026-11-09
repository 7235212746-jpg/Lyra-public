from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from lyra.models import ParsedBandData, SpinKind


_SAMPLED_PATH_WARNING = (
    "结论仅覆盖文件中实际采样的 k 点；高对称线性路径不能证明整个布里渊区的严格全局带隙。"
)


@dataclass(frozen=True)
class BandEdgeState:
    spin: str
    kpoint_index: int
    band_index: int
    energy: float
    occupation: float


@dataclass(frozen=True)
class SpinBandEdgeAnalysis:
    spin: str
    status: str
    vbm: BandEdgeState | None
    cbm: BandEdgeState | None
    gap: float | None
    is_direct: bool | None
    full_occupation: float | None
    partial_state_count: int
    evidence: str
    vbm_candidates: tuple[BandEdgeState, ...] = ()
    cbm_candidates: tuple[BandEdgeState, ...] = ()


@dataclass(frozen=True)
class BandEdgeAnalysis:
    classification: str
    method: str
    reference_energy: float
    channels: tuple[SpinBandEdgeAnalysis, ...]
    vbm: BandEdgeState | None = None
    cbm: BandEdgeState | None = None
    gap: float | None = None
    is_direct: bool | None = None
    confidence: str = "unavailable"
    warnings: tuple[str, ...] = ()
    vbm_candidates: tuple[BandEdgeState, ...] = ()
    cbm_candidates: tuple[BandEdgeState, ...] = ()


def _same_kpoint(
    data: ParsedBandData,
    first: BandEdgeState,
    second: BandEdgeState,
    tolerance: float,
) -> bool:
    kpoints = np.asarray(data.kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        return first.kpoint_index == second.kpoint_index
    delta = kpoints[first.kpoint_index] - kpoints[second.kpoint_index]
    delta -= np.rint(delta)
    return bool(np.max(np.abs(delta)) <= tolerance)


def _edge_candidates(
    spin: str,
    energies: np.ndarray,
    occupations: np.ndarray,
    mask: np.ndarray,
    edge_energy: float,
    tolerance: float,
) -> tuple[BandEdgeState, ...]:
    candidate_mask = mask & np.isclose(
        energies, edge_energy, atol=tolerance, rtol=0.0
    )
    return tuple(
        BandEdgeState(
            spin=spin,
            kpoint_index=int(kpoint_index),
            band_index=int(band_index),
            energy=float(energies[kpoint_index, band_index]),
            occupation=float(occupations[kpoint_index, band_index]),
        )
        for kpoint_index, band_index in np.argwhere(candidate_mask)
    )


def _select_edge_pair(
    data: ParsedBandData,
    vbm_candidates: tuple[BandEdgeState, ...],
    cbm_candidates: tuple[BandEdgeState, ...],
    kpoint_tolerance: float,
) -> tuple[BandEdgeState, BandEdgeState, bool]:
    for vbm in vbm_candidates:
        for cbm in cbm_candidates:
            if _same_kpoint(data, vbm, cbm, kpoint_tolerance):
                return vbm, cbm, True
    return vbm_candidates[0], cbm_candidates[0], False


def _crosses_fermi_level(
    data: ParsedBandData,
    energies: np.ndarray,
    tolerance: float,
) -> bool:
    shifted = energies - float(data.efermi)
    if np.any(np.isfinite(shifted) & (np.abs(shifted) <= tolerance)):
        return True
    for start, end in data.segment_ranges:
        start = max(0, int(start))
        end = min(len(shifted), int(end))
        if end - start < 2:
            continue
        segment = shifted[start:end]
        finite_pairs = np.isfinite(segment[:-1]) & np.isfinite(segment[1:])
        if np.any(finite_pairs & (segment[:-1] * segment[1:] < 0.0)):
            return True
    return False


def _full_occupation(
    data: ParsedBandData,
    spin: str,
    occupations: np.ndarray,
    finite: np.ndarray,
    tolerance: float,
) -> float:
    if data.spin_kind is SpinKind.NON_SPIN and spin == "total":
        maximum = float(np.max(occupations[finite]))
        return 2.0 if maximum > 1.0 + tolerance else 1.0
    return 1.0


def _analyze_channel(
    data: ParsedBandData,
    spin: str,
    *,
    occupation_tolerance: float,
    energy_tolerance: float,
    kpoint_tolerance: float,
) -> SpinBandEdgeAnalysis:
    energies = np.asarray(data.energies[spin], dtype=float)
    raw_occupations = data.occupations.get(spin)
    if raw_occupations is None:
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, None, 0,
            "该通道没有占据数，未使用 E-fermi 猜测带边。",
        )
    occupations = np.asarray(raw_occupations, dtype=float)
    if energies.ndim != 2 or occupations.shape != energies.shape:
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, None, 0,
            "占据数与能量张量形状不一致。",
        )
    finite = np.isfinite(energies) & np.isfinite(occupations)
    if not np.any(finite):
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, None, 0,
            "能量或占据数没有有限值。",
        )
    full = _full_occupation(
        data, spin, occupations, finite, occupation_tolerance
    )
    if np.any(occupations[finite] < -occupation_tolerance) or np.any(
        occupations[finite] > full + occupation_tolerance
    ):
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, full, 0,
            f"占据数超出该通道的物理范围 0 至 {full:g}。",
        )
    occupied = finite & (occupations >= full - occupation_tolerance)
    empty = finite & (occupations <= occupation_tolerance)
    partial = finite & ~occupied & ~empty
    partial_count = int(np.count_nonzero(partial))
    crosses_fermi = _crosses_fermi_level(data, energies, energy_tolerance)
    if crosses_fermi:
        return SpinBandEdgeAnalysis(
            spin, "metallic", None, None, None, None, full, partial_count,
            f"能带在连续路径段穿越 E-fermi；同时发现 {partial_count} 个部分占据态。",
        )
    if partial_count:
        return SpinBandEdgeAnalysis(
            spin, "ambiguous", None, None, None, None, full, partial_count,
            f"发现 {partial_count} 个部分占据态，但采样路径未穿越 E-fermi；"
            "可能来自展宽，不能可靠判为金属。",
        )
    if not np.any(occupied):
        detail = "占据数全为零，无法从该路径确定价带。" if np.all(occupations[finite] <= occupation_tolerance) else "没有可靠的完全占据态。"
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, full, 0, detail
        )
    if not np.any(empty):
        return SpinBandEdgeAnalysis(
            spin, "unavailable", None, None, None, None, full, 0,
            "没有可靠的空态，所选能带范围不足以确定导带。",
        )

    occupied_energies = np.where(occupied, energies, -np.inf)
    empty_energies = np.where(empty, energies, np.inf)
    vbm_energy = float(np.max(occupied_energies))
    cbm_energy = float(np.min(empty_energies))
    vbm_candidates = _edge_candidates(
        spin, energies, occupations, occupied, vbm_energy, energy_tolerance
    )
    cbm_candidates = _edge_candidates(
        spin, energies, occupations, empty, cbm_energy, energy_tolerance
    )
    vbm, cbm, direct = _select_edge_pair(
        data, vbm_candidates, cbm_candidates, kpoint_tolerance
    )
    gap = float(cbm_energy - vbm_energy)
    if gap < -energy_tolerance:
        return SpinBandEdgeAnalysis(
            spin, "metallic", vbm, cbm, None, direct, full, 0,
            f"完全占据态与空态能量重叠 {abs(gap):.6f} eV。",
            vbm_candidates, cbm_candidates,
        )
    if gap <= energy_tolerance:
        return SpinBandEdgeAnalysis(
            spin, "zero_gap", vbm, cbm, 0.0, direct, full, 0,
            f"占据态与空态间隔不超过 {energy_tolerance:g} eV。",
            vbm_candidates, cbm_candidates,
        )
    return SpinBandEdgeAnalysis(
        spin, "gapped", vbm, cbm, gap, direct, full, 0,
        f"占据数呈 0/{full:g} 二值分布；带隙 {gap:.6f} eV。",
        vbm_candidates, cbm_candidates,
    )


def analyze_band_edges(
    data: ParsedBandData,
    *,
    reference_energy: float | None = None,
    occupation_tolerance: float = 1e-3,
    energy_tolerance: float = 1e-5,
    kpoint_tolerance: float = 1e-8,
) -> BandEdgeAnalysis:
    reference = data.efermi if reference_energy is None else float(reference_energy)
    if not math.isfinite(reference):
        raise ValueError("Band-edge reference energy must be finite.")
    tolerances = (occupation_tolerance, energy_tolerance, kpoint_tolerance)
    if (
        not all(math.isfinite(value) for value in tolerances)
        or occupation_tolerance <= 0
        or energy_tolerance < 0
        or kpoint_tolerance < 0
    ):
        raise ValueError("Band-edge tolerances must be non-negative and finite.")
    spins = tuple(data.energies)
    channels = tuple(
        _analyze_channel(
            data,
            spin,
            occupation_tolerance=occupation_tolerance,
            energy_tolerance=energy_tolerance,
            kpoint_tolerance=kpoint_tolerance,
        )
        for spin in spins
    )
    if not channels or any(
        channel.status in {"unavailable", "ambiguous"} for channel in channels
    ):
        warnings = tuple(
            channel.evidence
            for channel in channels
            if channel.status in {"unavailable", "ambiguous"}
        )
        if not channels:
            warnings = ("能带数据没有能量通道。",)
        return BandEdgeAnalysis(
            "unavailable", "occupations", reference, channels,
            warnings=warnings, confidence="unavailable",
        )

    metallic = [channel for channel in channels if channel.status in {"metallic", "zero_gap"}]
    gapped = [channel for channel in channels if channel.status == "gapped"]
    if metallic:
        if len(channels) == 2 and len(metallic) == 1 and len(gapped) == 1:
            classification = "half_metal"
        elif all(channel.status == "zero_gap" for channel in channels):
            classification = "zero_gap"
        else:
            classification = "metal"
        return BandEdgeAnalysis(
            classification, "occupations", reference, channels,
            confidence="high", warnings=(_SAMPLED_PATH_WARNING,),
        )

    vbm_energy = max(channel.vbm.energy for channel in channels)
    cbm_energy = min(channel.cbm.energy for channel in channels)
    vbm_candidates = tuple(
        state
        for channel in channels
        for state in channel.vbm_candidates
        if math.isclose(state.energy, vbm_energy, abs_tol=energy_tolerance, rel_tol=0.0)
    )
    cbm_candidates = tuple(
        state
        for channel in channels
        for state in channel.cbm_candidates
        if math.isclose(state.energy, cbm_energy, abs_tol=energy_tolerance, rel_tol=0.0)
    )
    vbm, cbm, direct = _select_edge_pair(
        data, vbm_candidates, cbm_candidates, kpoint_tolerance
    )
    gap = float(cbm_energy - vbm_energy)
    if gap < -energy_tolerance:
        classification = "semimetal"
        gap_value = None
    elif gap <= energy_tolerance:
        classification = "zero_gap"
        gap_value = 0.0
    else:
        classification = "semiconductor"
        gap_value = gap
    return BandEdgeAnalysis(
        classification,
        "occupations",
        reference,
        channels,
        vbm=vbm,
        cbm=cbm,
        gap=gap_value,
        is_direct=direct,
        confidence="high",
        warnings=(
            *(_SAMPLED_PATH_WARNING,),
            *(("价带顶与导带底来自不同自旋通道。",) if vbm.spin != cbm.spin else ()),
        ),
        vbm_candidates=vbm_candidates,
        cbm_candidates=cbm_candidates,
    )


def format_band_edge_summary(result: BandEdgeAnalysis) -> str:
    if result.classification == "unavailable":
        detail = "；".join(result.warnings) or "缺少可靠占据数。"
        return f"自动带边不可用：{detail}"
    labels = {
        "metal": "金属",
        "half_metal": "half-metal",
        "zero_gap": "零带隙/半金属候选",
        "semimetal": "能带重叠/半金属候选",
        "semiconductor": "半导体",
    }
    if result.gap is None:
        evidence = "；".join(
            f"{channel.spin}: {channel.evidence}" for channel in result.channels
        )
        return f"自动带边（所采样 k 点）：{labels[result.classification]} | {evidence}"
    kind = "直接带隙" if result.is_direct else "间接带隙"
    return (
        f"自动带边（所采样 k 点）：{labels[result.classification]}，{kind} {result.gap:.6f} eV | "
        f"VBM {result.vbm.spin}/Band {result.vbm.band_index + 1}/k {result.vbm.kpoint_index + 1} | "
        f"CBM {result.cbm.spin}/Band {result.cbm.band_index + 1}/k {result.cbm.kpoint_index + 1}"
    )

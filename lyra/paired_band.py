from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from lyra.cache import CacheIdentity
from lyra.models import BandElementIndex, BandStateSelection, ParsedBandData


@dataclass(frozen=True)
class PairCompatibility:
    structure_compatible: bool | None
    path_compatible: bool
    sync_enabled: bool
    mapping: tuple[int, ...]
    maximum_k_delta: float
    reasons: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class BandCandidate:
    selection: BandStateSelection
    energy: float
    energy_delta: float
    kpoint_delta: float
    composition_similarity: float
    score: float
    evidence: str

    @property
    def band_index(self) -> int:
        return self.selection.band_index

    @property
    def kpoint_index(self) -> int:
        return self.selection.kpoint_index


@dataclass(frozen=True)
class PairedBandSession:
    left: BandElementIndex
    right: BandElementIndex
    left_identity: CacheIdentity
    right_identity: CacheIdentity
    left_folder: Path
    right_folder: Path
    left_mode: str
    right_mode: str
    compatibility: PairCompatibility
    left_detail_available: bool = True
    right_detail_available: bool = True


def build_standard_index(
    data: ParsedBandData,
    reference_energy: float | None = None,
) -> BandElementIndex:
    reference = data.efermi if reference_energy is None else float(reference_energy)
    spin_names = tuple(data.energies)
    first = np.asarray(data.energies[spin_names[0]], dtype=float)
    elements = tuple(data.elements)
    shape = (len(spin_names), len(elements), *first.shape)
    return BandElementIndex(
        spin_names=spin_names,
        elements=elements,
        orbitals=(),
        kpoints=np.asarray(data.kpoints, dtype=float),
        distances=np.asarray(data.distances, dtype=float),
        segment_ranges=data.segment_ranges,
        tick_positions=data.tick_positions,
        tick_labels=data.tick_labels,
        energies={
            spin: np.asarray(values, dtype=float) - reference
            for spin, values in data.energies.items()
        },
        reference_energy=reference,
        raw_element_totals=np.full(shape, np.nan, dtype=float),
        raw_projection_totals=np.full((len(spin_names), *first.shape), np.nan, dtype=float),
        relative_element_percentages=np.full(shape, np.nan, dtype=float),
        structure=data.structure,
    )


def normalise_label(label: str) -> str:
    value = str(label).strip().replace("$", "").replace("{", "").replace("}", "")
    compact = value.replace("\\", "").replace("_", "").replace(" ", "")
    if compact.casefold() in {"g", "gamma", "Γ".casefold()}:
        return "G"
    return compact.upper()


def periodic_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return delta - np.rint(delta)


def nearest_periodic_kpoint(
    source: np.ndarray,
    targets: np.ndarray,
) -> tuple[int, float]:
    target_array = np.asarray(targets, dtype=float)
    if target_array.ndim != 2 or target_array.shape[1] != 3 or not len(target_array):
        raise ValueError("Target k-points must be a non-empty (n, 3) array.")
    distances = np.linalg.norm(periodic_delta(target_array, source), axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def assess_index_compatibility(
    left: BandElementIndex,
    right: BandElementIndex,
    *,
    tolerance: float = 1e-8,
) -> PairCompatibility:
    direct_match = len(left.kpoints) == len(right.kpoints) and bool(
        np.all(np.linalg.norm(periodic_delta(left.kpoints, right.kpoints), axis=1) <= tolerance)
    )
    structure_compatible: bool | None = None
    if left.structure is not None and right.structure is not None:
        same_elements = left.structure.atom_elements == right.structure.atom_elements
        same_lattice = bool(
            np.max(np.abs(left.structure.lattice - right.structure.lattice)) <= 1e-6
        )
        same_positions = (
            left.structure.fractional_positions.shape
            == right.structure.fractional_positions.shape
            and bool(
                np.max(
                    np.abs(
                        periodic_delta(
                            left.structure.fractional_positions,
                            right.structure.fractional_positions,
                        )
                    )
                )
                <= 1e-6
            )
        )
        structure_compatible = bool(same_elements and same_lattice and same_positions)
    mapping_with_delta = (
        tuple(
            (index, float(np.linalg.norm(periodic_delta(kpoint, right.kpoints[index]))))
            for index, kpoint in enumerate(left.kpoints)
        )
        if direct_match
        else tuple(nearest_periodic_kpoint(kpoint, right.kpoints) for kpoint in left.kpoints)
    )
    mapping = tuple(index for index, _delta in mapping_with_delta)
    maximum_delta = max((delta for _index, delta in mapping_with_delta), default=math.inf)
    labels_match = tuple(map(normalise_label, left.tick_labels)) == tuple(
        map(normalise_label, right.tick_labels)
    )
    exact_points = direct_match and mapping == tuple(range(len(left.kpoints)))
    topology_match = left.segment_ranges == right.segment_ranges and labels_match
    compatible = bool(exact_points and topology_match)
    reasons: list[str] = []
    if not exact_points:
        reasons.append("k-point sampling differs; nearest periodic points are approximate")
    if left.segment_ranges != right.segment_ranges:
        reasons.append("k-path segment topology differs")
    if not labels_match:
        reasons.append("high-symmetry labels differ")
    if structure_compatible is False:
        reasons.append("ordered elements, lattice, or periodic atomic positions differ")
    evidence = (
        f"left k-points={len(left.kpoints)}",
        f"right k-points={len(right.kpoints)}",
        f"maximum periodic k delta={maximum_delta:.3g}",
        "normalised labels=" + ("match" if labels_match else "differ"),
    )
    return PairCompatibility(
        structure_compatible=structure_compatible,
        path_compatible=compatible,
        sync_enabled=compatible and structure_compatible is not False,
        mapping=mapping,
        maximum_k_delta=maximum_delta,
        reasons=tuple(reasons),
        evidence=evidence,
    )


def _composition(index: BandElementIndex, selection: BandStateSelection) -> dict[str, float]:
    spin_index = index.spin_names.index(selection.spin)
    values = np.asarray(
        index.relative_element_percentages[
            spin_index, :, selection.kpoint_index, selection.band_index
        ],
        dtype=float,
    )
    return {
        element: float(value) if np.isfinite(value) else 0.0
        for element, value in zip(index.elements, values, strict=True)
    }


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    common = tuple(element for element in left if element in right)
    if not common:
        return float("nan")
    a = np.asarray([left[element] for element in common], dtype=float)
    b = np.asarray([right[element] for element in common], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def rank_projection_candidates(
    source: BandElementIndex,
    target: BandElementIndex,
    source_selection: BandStateSelection,
    target_spin: str,
    *,
    energy_window: float = 1.0,
    energy_scale: float = 0.25,
    limit: int = 5,
) -> tuple[BandCandidate, ...]:
    if target_spin not in target.spin_names:
        raise ValueError(f"Unknown target spin channel {target_spin!r}.")
    source_energy = float(
        source.energies[source_selection.spin][
            source_selection.kpoint_index, source_selection.band_index
        ]
    )
    target_k, k_delta = nearest_periodic_kpoint(
        source.kpoints[source_selection.kpoint_index], target.kpoints
    )
    target_energies = np.asarray(target.energies[target_spin][target_k], dtype=float)
    source_composition = _composition(source, source_selection)
    candidates: list[BandCandidate] = []
    for band_index, energy in enumerate(target_energies):
        energy_delta = float(energy - source_energy)
        if abs(energy_delta) > energy_window:
            continue
        selection = BandStateSelection(target_spin, target_k, band_index)
        similarity = _cosine_similarity(source_composition, _composition(target, selection))
        energy_score = math.exp(-abs(energy_delta) / max(energy_scale, 1e-9))
        score = energy_score if not np.isfinite(similarity) else 0.65 * similarity + 0.35 * energy_score
        candidates.append(
            BandCandidate(
                selection=selection,
                energy=float(energy),
                energy_delta=energy_delta,
                kpoint_delta=k_delta,
                composition_similarity=similarity,
                score=float(score),
                evidence="Projection-similarity candidate; not a wavefunction identity assignment.",
            )
        )
    candidates.sort(key=lambda item: (-item.score, abs(item.energy_delta), item.band_index))
    return tuple(candidates[: max(1, int(limit))])

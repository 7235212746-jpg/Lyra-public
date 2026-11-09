from __future__ import annotations

import numpy as np

from lyra.models import (
    BandElementIndex,
    BandStateDetail,
    BandStateSelection,
    ParsedBandData,
)
from lyra.projection_store import BandProjectionStore


NORMALIZATION_THRESHOLD = 1e-8
DEFAULT_DEGENERACY_TOLERANCE_EV = 0.005


def _relative_percentages(values: np.ndarray, total: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    denominator = np.asarray(total, dtype=float)
    valid = np.isfinite(denominator) & (denominator > NORMALIZATION_THRESHOLD)
    np.divide(
        np.asarray(values, dtype=float) * 100.0,
        np.expand_dims(denominator, axis=1),
        out=result,
        where=np.expand_dims(valid, axis=1),
    )
    return result


def _validate_store(data: ParsedBandData, store: BandProjectionStore) -> None:
    if tuple(data.energies) != store.spin_names:
        raise ValueError("Band energies and projection store spin channels do not match.")
    if tuple(data.elements) != store.elements:
        raise ValueError("Band data and projection store elements do not match.")
    if any(
        np.asarray(data.energies[spin]).shape != store.tensor.shape[3:]
        for spin in store.spin_names
    ):
        raise ValueError("Band energies and projection tensor dimensions do not match.")


def build_element_index(
    data: ParsedBandData,
    store: BandProjectionStore,
    reference_energy: float | None = None,
) -> BandElementIndex:
    _validate_store(data, store)
    raw_elements = np.sum(store.tensor, axis=2, dtype=np.float32)
    raw_totals = np.sum(raw_elements, axis=1, dtype=np.float32)
    relative = _relative_percentages(raw_elements, raw_totals)
    reference = data.efermi if reference_energy is None else float(reference_energy)
    return BandElementIndex(
        spin_names=store.spin_names,
        elements=store.elements,
        orbitals=store.orbitals,
        kpoints=np.asarray(data.kpoints, dtype=float),
        distances=np.asarray(data.distances, dtype=float),
        segment_ranges=data.segment_ranges,
        tick_positions=data.tick_positions,
        tick_labels=data.tick_labels,
        energies={
            spin: np.asarray(data.energies[spin], dtype=float) - reference
            for spin in store.spin_names
        },
        reference_energy=reference,
        raw_element_totals=raw_elements,
        raw_projection_totals=raw_totals,
        relative_element_percentages=relative,
        structure=data.structure,
    )


def _selection_indices(
    data: ParsedBandData,
    store: BandProjectionStore,
    selection: BandStateSelection,
) -> tuple[int, int, int]:
    _validate_store(data, store)
    try:
        spin_index = store.spin_names.index(selection.spin)
    except ValueError as exc:
        raise ValueError(f"Unknown band-state spin channel {selection.spin!r}.") from exc
    nkpoints, nbands = store.tensor.shape[3:]
    if selection.kpoint_index >= nkpoints or selection.band_index >= nbands:
        raise ValueError("Band-state selection is outside the available data.")
    return spin_index, selection.kpoint_index, selection.band_index


def _detail_from_weights(
    selection: BandStateSelection,
    store: BandProjectionStore,
    weights: np.ndarray,
    spin_components: np.ndarray | None,
    band_indices: tuple[int, ...],
    *,
    whole_band: bool,
) -> BandStateDetail:
    raw_total = float(np.sum(weights, dtype=np.float64))
    relative = np.full(weights.shape, np.nan, dtype=float)
    if np.isfinite(raw_total) and raw_total > NORMALIZATION_THRESHOLD:
        relative = np.asarray(weights, dtype=float) * (100.0 / raw_total)
    return BandStateDetail(
        selection=selection,
        elements=store.elements,
        orbitals=store.orbitals,
        band_indices=band_indices,
        raw_orbital_weights=np.asarray(weights, dtype=float),
        relative_orbital_percentages=relative,
        raw_projection_total=raw_total,
        spin_components=(
            np.asarray(spin_components, dtype=float)
            if spin_components is not None
            else None
        ),
        whole_band=whole_band,
    )


def band_state_detail(
    data: ParsedBandData,
    store: BandProjectionStore,
    selection: BandStateSelection,
    band_indices: tuple[int, ...] | None = None,
) -> BandStateDetail:
    spin_index, kpoint_index, band_index = _selection_indices(data, store, selection)
    members = band_indices or (band_index,)
    if any(index < 0 or index >= store.tensor.shape[-1] for index in members):
        raise ValueError("Combined band selection is outside the available data.")
    selected_weights = np.take(
        store.tensor[spin_index, :, :, kpoint_index, :],
        members,
        axis=-1,
    )
    weights = np.sum(selected_weights, axis=-1, dtype=np.float32)
    components = None
    if store.spin_components is not None:
        components = np.sum(
            np.take(
                store.spin_components[:, :, :, kpoint_index, :],
                members,
                axis=-1,
            ),
            axis=-1,
            dtype=np.float32,
        )
    return _detail_from_weights(
        selection,
        store,
        weights,
        components,
        tuple(members),
        whole_band=False,
    )


def _path_weighted_mean(
    values: np.ndarray,
    distances: np.ndarray,
    segment_ranges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    integral = np.zeros(values.shape[:-2], dtype=np.float64)
    total_length = 0.0
    for start, stop in segment_ranges:
        x = distances[start:stop]
        if len(x) < 2:
            continue
        length = float(x[-1] - x[0])
        if length <= 0.0:
            continue
        integral += np.trapezoid(values[..., start:stop, :], x=x, axis=-2)[..., 0]
        total_length += length
    if total_length <= 0.0:
        raise ValueError("Whole-band summary requires a non-zero continuous path length.")
    return integral / total_length


def whole_band_detail(
    data: ParsedBandData,
    store: BandProjectionStore,
    selection: BandStateSelection,
    band_indices: tuple[int, ...] | None = None,
) -> BandStateDetail:
    spin_index, _kpoint_index, band_index = _selection_indices(data, store, selection)
    members = band_indices or (band_index,)
    if any(index < 0 or index >= store.tensor.shape[-1] for index in members):
        raise ValueError("Combined band selection is outside the available data.")
    selected = np.sum(
        np.take(store.tensor[spin_index], members, axis=-1),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    weights = _path_weighted_mean(selected, data.distances, data.segment_ranges)
    components = None
    if store.spin_components is not None:
        selected_components = np.sum(
            np.take(store.spin_components, members, axis=-1),
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        components = _path_weighted_mean(
            selected_components,
            data.distances,
            data.segment_ranges,
        )
    return _detail_from_weights(
        selection,
        store,
        weights,
        components,
        tuple(members),
        whole_band=True,
    )


def near_degenerate_bands(
    index: BandElementIndex,
    selection: BandStateSelection,
    tolerance_ev: float = DEFAULT_DEGENERACY_TOLERANCE_EV,
) -> tuple[int, ...]:
    if tolerance_ev < 0.0:
        raise ValueError("Degeneracy tolerance must be non-negative.")
    try:
        energies = np.asarray(index.energies[selection.spin], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Unknown band-state spin channel {selection.spin!r}.") from exc
    if (
        selection.kpoint_index >= energies.shape[0]
        or selection.band_index >= energies.shape[1]
    ):
        raise ValueError("Band-state selection is outside the available data.")
    selected_energy = energies[selection.kpoint_index, selection.band_index]
    candidates = np.flatnonzero(
        np.abs(energies[selection.kpoint_index] - selected_energy) <= tolerance_ev
    )
    return tuple(int(value) for value in candidates)

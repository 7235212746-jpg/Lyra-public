from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from lyra.adapters.common import normalise_orbital, orbital_group
from lyra.models import ParsedBandData, ParsedDosData, ProjectionChannel


@dataclass(frozen=True)
class BandProjectionStore:
    """Compact resolved band projections with spin, element, and orbital axes."""

    spin_names: tuple[str, ...]
    elements: tuple[str, ...]
    orbitals: tuple[str, ...]
    tensor: np.ndarray
    spin_components: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.spin_names or not self.elements or not self.orbitals:
            raise ValueError("Projection store axes must not be empty.")
        if len(set(self.spin_names)) != len(self.spin_names):
            raise ValueError("Projection store spin names must be unique.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("Projection store elements must be unique.")
        if len(set(self.orbitals)) != len(self.orbitals):
            raise ValueError("Projection store orbitals must be unique.")
        if not isinstance(self.tensor, np.ndarray) or self.tensor.ndim != 5:
            raise ValueError("Projection tensor must have five dimensions.")
        if self.tensor.dtype != np.dtype(np.float32):
            raise ValueError("Projection tensor must use float32 values.")
        if not self.tensor.flags.c_contiguous:
            raise ValueError("Projection tensor must be C-contiguous.")
        expected = (len(self.spin_names), len(self.elements), len(self.orbitals))
        if self.tensor.shape[:3] != expected or not all(self.tensor.shape[3:]):
            raise ValueError("Projection tensor shape does not match its metadata.")
        if self.spin_components is not None:
            if (
                not isinstance(self.spin_components, np.ndarray)
                or self.spin_components.ndim != 5
                or self.spin_components.dtype != np.dtype(np.float32)
                or not self.spin_components.flags.c_contiguous
                or self.spin_components.shape
                != (3, len(self.elements), len(self.orbitals), *self.tensor.shape[3:])
            ):
                raise ValueError(
                    "SOC spin components must be a contiguous float32 "
                    "(3, element, orbital, k-point, band) tensor."
                )

    @classmethod
    def from_parsed(cls, data: ParsedBandData) -> BandProjectionStore:
        spin_names = tuple(data.energies)
        if not spin_names:
            raise ValueError("Band data must contain at least one energy channel.")
        energy_shape = tuple(np.asarray(data.energies[spin_names[0]]).shape)
        if len(energy_shape) != 2 or not all(energy_shape):
            raise ValueError("Band energies must have non-empty k-point and band axes.")
        for spin_name in spin_names:
            if tuple(np.asarray(data.energies[spin_name]).shape) != energy_shape:
                raise ValueError("Band energy channels must have matching shapes.")

        tensor = np.empty(
            (len(spin_names), len(data.elements), len(data.orbitals), *energy_shape),
            dtype=np.float32,
        )
        for element_index, element in enumerate(data.elements):
            for orbital_index, orbital in enumerate(data.orbitals):
                key = f"{element}:{orbital}"
                try:
                    values_by_spin = data.projections[key]
                except KeyError as exc:
                    raise ValueError(f"Missing resolved projection channel {key!r}.") from exc
                for spin_index, spin_name in enumerate(spin_names):
                    try:
                        values = np.asarray(values_by_spin[spin_name])
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing {spin_name!r} values for projection channel {key!r}."
                        ) from exc
                    if values.shape != energy_shape:
                        raise ValueError(
                            f"Projection channel {key!r} does not match the band energy shape."
                        )
                    tensor[spin_index, element_index, orbital_index] = values
        return cls(
            spin_names=spin_names,
            elements=tuple(data.elements),
            orbitals=tuple(data.orbitals),
            tensor=np.ascontiguousarray(tensor),
        )

    def materialize(
        self,
        base_data: ParsedBandData,
        channels: Iterable[ProjectionChannel],
    ) -> ParsedBandData:
        if tuple(base_data.energies) != self.spin_names:
            raise ValueError("Band data energy channels do not match the projection store.")
        if tuple(base_data.elements) != self.elements:
            raise ValueError("Band data elements do not match the projection store.")
        expected_shape = self.tensor.shape[3:]
        if any(np.asarray(values).shape != expected_shape for values in base_data.energies.values()):
            raise ValueError("Band data energy shape does not match the projection store.")

        projections: dict[str, dict[str, np.ndarray]] = {}
        element_indices = {element: index for index, element in enumerate(self.elements)}
        for channel in channels:
            try:
                element_index = element_indices[channel.element]
            except KeyError as exc:
                raise ValueError(f"Unknown projection element {channel.element!r}.") from exc
            orbital_indices, key = self._orbital_indices(channel)
            projections[key] = {
                spin_name: np.sum(
                    self.tensor[spin_index, element_index, orbital_indices],
                    axis=0,
                    dtype=np.float32,
                )
                for spin_index, spin_name in enumerate(self.spin_names)
            }
        projection_total = {
            spin_name: np.sum(
                self.tensor[spin_index],
                axis=(0, 1),
                dtype=np.float32,
            )
            for spin_index, spin_name in enumerate(self.spin_names)
        }
        return replace(
            base_data,
            projections=projections,
            projection_total=projection_total,
        )

    def _orbital_indices(self, channel: ProjectionChannel) -> tuple[np.ndarray | slice, str]:
        if channel.orbital is None:
            return slice(None), channel.element
        if channel.orbital in {"s", "p", "d", "f"}:
            indices = np.array(
                [
                    index
                    for index, orbital in enumerate(self.orbitals)
                    if orbital_group(orbital) == channel.orbital
                ],
                dtype=int,
            )
        else:
            requested = normalise_orbital(channel.orbital)
            indices = np.array(
                [
                    index
                    for index, orbital in enumerate(self.orbitals)
                    if normalise_orbital(orbital) == requested
                ],
                dtype=int,
            )
        if not len(indices):
            raise ValueError(f"Unknown projection orbital {channel.orbital!r}.")
        return indices, f"{channel.element}:{channel.orbital}"


@dataclass(frozen=True)
class DosProjectionStore:
    """Compact resolved DOS projections with spin, element, and orbital axes."""

    spin_names: tuple[str, ...]
    elements: tuple[str, ...]
    orbitals: tuple[str, ...]
    tensor: np.ndarray

    def __post_init__(self) -> None:
        if not self.spin_names or not self.elements or not self.orbitals:
            raise ValueError("DOS projection store axes must not be empty.")
        if len(set(self.spin_names)) != len(self.spin_names):
            raise ValueError("DOS projection store spin names must be unique.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("DOS projection store elements must be unique.")
        if len(set(self.orbitals)) != len(self.orbitals):
            raise ValueError("DOS projection store orbitals must be unique.")
        if not isinstance(self.tensor, np.ndarray) or self.tensor.ndim != 4:
            raise ValueError("DOS projection tensor must have four dimensions.")
        if self.tensor.dtype != np.dtype(np.float32):
            raise ValueError("DOS projection tensor must use float32 values.")
        if not self.tensor.flags.c_contiguous:
            raise ValueError("DOS projection tensor must be C-contiguous.")
        expected = (len(self.spin_names), len(self.elements), len(self.orbitals))
        if self.tensor.shape[:3] != expected or not self.tensor.shape[3]:
            raise ValueError("DOS projection tensor shape does not match its metadata.")

    @classmethod
    def from_parsed(cls, data: ParsedDosData) -> DosProjectionStore:
        spin_names = tuple(data.densities)
        if not spin_names:
            raise ValueError("DOS data must contain at least one density channel.")
        energy_count = np.asarray(data.energies).size
        if np.asarray(data.energies).ndim != 1 or not energy_count:
            raise ValueError("DOS energies must have a non-empty one-dimensional axis.")
        if any(np.asarray(data.densities[spin]).shape != (energy_count,) for spin in spin_names):
            raise ValueError("DOS density channels must match the energy axis.")

        tensor = np.empty(
            (len(spin_names), len(data.elements), len(data.orbitals), energy_count),
            dtype=np.float32,
        )
        for element_index, element in enumerate(data.elements):
            for orbital_index, orbital in enumerate(data.orbitals):
                key = f"{element}:{orbital}"
                try:
                    values_by_spin = data.projections[key]
                except KeyError as exc:
                    raise ValueError(f"Missing resolved DOS projection channel {key!r}.") from exc
                for spin_index, spin_name in enumerate(spin_names):
                    try:
                        values = np.asarray(values_by_spin[spin_name])
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing {spin_name!r} values for DOS projection channel {key!r}."
                        ) from exc
                    if values.shape != (energy_count,):
                        raise ValueError(
                            f"DOS projection channel {key!r} does not match the energy axis."
                        )
                    tensor[spin_index, element_index, orbital_index] = values
        return cls(
            spin_names=spin_names,
            elements=tuple(data.elements),
            orbitals=tuple(data.orbitals),
            tensor=np.ascontiguousarray(tensor),
        )

    def materialize(
        self,
        base_data: ParsedDosData,
        channels: Iterable[ProjectionChannel],
    ) -> ParsedDosData:
        if tuple(base_data.densities) != self.spin_names:
            raise ValueError("DOS density channels do not match the projection store.")
        if tuple(base_data.elements) != self.elements:
            raise ValueError("DOS data elements do not match the projection store.")
        energy_count = self.tensor.shape[3]
        if np.asarray(base_data.energies).shape != (energy_count,) or any(
            np.asarray(values).shape != (energy_count,)
            for values in base_data.densities.values()
        ):
            raise ValueError("DOS data shape does not match the projection store.")

        projections: dict[str, dict[str, np.ndarray]] = {}
        element_indices = {element: index for index, element in enumerate(self.elements)}
        for channel in channels:
            try:
                element_index = element_indices[channel.element]
            except KeyError as exc:
                raise ValueError(f"Unknown projection element {channel.element!r}.") from exc
            orbital_indices, key = self._orbital_indices(channel)
            projections[key] = {
                spin_name: np.sum(
                    self.tensor[spin_index, element_index, orbital_indices],
                    axis=0,
                    dtype=np.float32,
                )
                for spin_index, spin_name in enumerate(self.spin_names)
            }
        return replace(base_data, projections=projections)

    def _orbital_indices(self, channel: ProjectionChannel) -> tuple[np.ndarray | slice, str]:
        if channel.orbital is None:
            return slice(None), channel.element
        if channel.orbital in {"s", "p", "d", "f"}:
            indices = np.array(
                [
                    index
                    for index, orbital in enumerate(self.orbitals)
                    if orbital_group(orbital) == channel.orbital
                ],
                dtype=int,
            )
        else:
            requested = normalise_orbital(channel.orbital)
            indices = np.array(
                [
                    index
                    for index, orbital in enumerate(self.orbitals)
                    if normalise_orbital(orbital) == requested
                ],
                dtype=int,
            )
        if not len(indices):
            raise ValueError(f"Unknown projection orbital {channel.orbital!r}.")
        return indices, f"{channel.element}:{channel.orbital}"

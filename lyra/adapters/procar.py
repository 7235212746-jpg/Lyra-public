from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
import re

import numpy as np

from lyra.adapters.common import normalise_orbital, orbital_group
from lyra.models import ParsedBandData, ProjectionChannel, SpinKind
from lyra.projection_store import BandProjectionStore


DIMENSIONS = re.compile(
    r"# of k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)",
    re.IGNORECASE,
)
SPIN_COMPONENT = re.compile(r"spin component\s+(\d+)", re.IGNORECASE)
KPOINT = re.compile(r"k-point\s+(\d+)\s*:\s*(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)
BAND = re.compile(r"band\s+(\d+)\s*#\s*energy\s+(\S+)", re.IGNORECASE)


class ProcarAdapter:
    @staticmethod
    def attach(folder: str | Path, band_data: ParsedBandData) -> ParsedBandData:
        store = ProcarAdapter.load_store(folder, band_data)
        channels = tuple(
            channel
            for element in store.elements
            for channel in (
                ProjectionChannel(element),
                *(ProjectionChannel(element, orbital_group(orbital)) for orbital in store.orbitals),
                *(ProjectionChannel(element, orbital) for orbital in store.orbitals),
            )
        )
        return replace(store.materialize(band_data, channels), orbitals=store.orbitals)

    @staticmethod
    def load_store(
        folder: str | Path,
        band_data: ParsedBandData,
        progress: Callable[[dict], None] | None = None,
    ) -> BandProjectionStore:
        folder = Path(folder)
        procar_path = folder / "PROCAR"
        emit = ProcarAdapter._progress_emitter(progress)
        emit("metadata", 0, 1, "Reading PROCAR metadata.")
        nkpoints, nbands, nions, raw_orbitals = ProcarAdapter._read_metadata(procar_path)
        spin_names = tuple(band_data.energies)
        if not spin_names:
            raise RuntimeError("EIGENVAL contains no energy channels.")
        expected_shape = tuple(np.asarray(band_data.energies[spin_names[0]]).shape)
        if (nkpoints, nbands) != expected_shape:
            raise RuntimeError(
                f"PROCAR dimensions {(nkpoints, nbands)} do not match EIGENVAL {expected_shape}."
            )
        if any(tuple(np.asarray(values).shape) != expected_shape for values in band_data.energies.values()):
            raise RuntimeError("EIGENVAL energy channels do not have matching dimensions.")
        site_elements = ProcarAdapter._read_poscar_site_elements(folder / "POSCAR")
        if len(site_elements) != nions:
            raise RuntimeError(
                f"PROCAR contains {nions} ions but POSCAR contains {len(site_elements)}."
            )
        source_components = ProcarAdapter._source_component_count(band_data, spin_names)
        interleaved_soc = (
            band_data.spin_kind is SpinKind.SOC
            and not ProcarAdapter._has_spin_component_headers(procar_path)
        )
        expected_tables = nkpoints * nbands * source_components
        if raw_orbitals is None:
            raise RuntimeError(f"PROCAR contains 0 projection tables; expected {expected_tables}.")

        table_orbitals = tuple(normalise_orbital(name) for name in raw_orbitals)
        orbitals = tuple(sorted(set(table_orbitals)))
        orbital_indices = np.array([orbitals.index(orbital) for orbital in table_orbitals], dtype=int)
        element_ranges = ProcarAdapter._element_ranges(site_elements, band_data.elements)
        tensor = np.empty(
            (len(spin_names), len(band_data.elements), len(orbitals), nkpoints, nbands),
            dtype=np.float32,
        )
        spin_components = (
            np.empty(
                (3, len(band_data.elements), len(orbitals), nkpoints, nbands),
                dtype=np.float32,
            )
            if band_data.spin_kind is SpinKind.SOC
            else None
        )
        table_seen = np.zeros((source_components, nkpoints, nbands), dtype=bool)
        emit(
            "metadata",
            1,
            1,
            f"Prepared {expected_tables} PROCAR projection tables for {len(orbitals)} orbitals.",
        )
        emit("parse", 0, expected_tables, "Parsing PROCAR projection tables.")

        stride = ProcarAdapter.progress_stride(expected_tables)
        parsed_tables = 0
        explicit_components = False
        expected_component = 1
        active_component: int | None = (
            1 if source_components == 1 or interleaved_soc else None
        )
        current_kpoint: int | None = None
        next_kpoint = 0
        next_band = 0
        pending_band: tuple[int, int, int] | None = None

        with procar_path.open("r", errors="ignore") as handle:
            lines = iter(handle)
            for line in lines:
                match = SPIN_COMPONENT.search(line)
                if match:
                    if interleaved_soc:
                        raise RuntimeError(
                            "PROCAR mixes interleaved SOC tables with global spin component headers."
                        )
                    component = int(match.group(1))
                    if active_component is not None:
                        ProcarAdapter._validate_component_complete(
                            active_component,
                            current_kpoint,
                            next_kpoint,
                            next_band,
                            pending_band,
                            nkpoints,
                            nbands,
                    )
                    if component != expected_component or component > source_components:
                        raise RuntimeError(
                            "PROCAR spin component headers are missing, reused, or out of order; "
                            "projection tables are invalid."
                        )
                    explicit_components = True
                    expected_component += 1
                    active_component = component
                    current_kpoint = None
                    next_kpoint = 0
                    next_band = 0
                    pending_band = None
                    continue

                match = KPOINT.search(line)
                if match:
                    if active_component is None:
                        raise RuntimeError("PROCAR is missing spin component 1.")
                    if pending_band is not None or (current_kpoint is not None and next_band != nbands):
                        raise RuntimeError("PROCAR k-point is missing a fresh band header.")
                    kpoint = int(match.group(1)) - 1
                    if kpoint != next_kpoint or kpoint < 0 or kpoint >= nkpoints:
                        raise RuntimeError("PROCAR k-point headers are missing, reused, or out of order.")
                    try:
                        procar_point = np.array(match.groups()[1:], dtype=float)
                    except ValueError as exc:
                        raise RuntimeError("Malformed PROCAR k-point coordinates.") from exc
                    ProcarAdapter._validate_kpoint_coordinates(
                        kpoint,
                        procar_point,
                        band_data.kpoints[kpoint],
                    )
                    current_kpoint = kpoint
                    next_kpoint += 1
                    next_band = 0
                    continue

                match = BAND.search(line)
                if match:
                    if active_component is None or current_kpoint is None or pending_band is not None:
                        raise RuntimeError("PROCAR band header is missing or out of order.")
                    band = int(match.group(1)) - 1
                    if band != next_band or band < 0 or band >= nbands:
                        raise RuntimeError("PROCAR band header is missing, reused, or out of order.")
                    try:
                        procar_energy = float(match.group(2))
                    except ValueError as exc:
                        raise RuntimeError("Malformed PROCAR band energy.") from exc
                    spin_name = (
                        spin_names[0]
                        if band_data.spin_kind is SpinKind.SOC
                        else spin_names[active_component - 1]
                    )
                    if interleaved_soc:
                        component_description = (
                            "PROCAR components 1-4 "
                            "(total/m_x/m_y/m_z, interleaved)"
                        )
                    elif band_data.spin_kind is SpinKind.SOC:
                        component_name = ("total", "m_x", "m_y", "m_z")[
                            active_component - 1
                        ]
                        component_description = (
                            f"PROCAR component {active_component} ({component_name})"
                        )
                    else:
                        component_description = (
                            f"PROCAR component {active_component} ({spin_name})"
                        )
                    ProcarAdapter._validate_band_energy(
                        kpoint,
                        band,
                        procar_energy,
                        band_data.energies[spin_name][kpoint, band],
                        spin_name,
                        component_description,
                    )
                    pending_band = (active_component - 1, current_kpoint, band)
                    next_band += 1
                    continue

                parts = line.split()
                if not parts or parts[0].lower() != "ion":
                    continue
                if pending_band is None:
                    raise RuntimeError("PROCAR ion table requires a fresh unconsumed band header.")
                raw_table_orbitals = parts[1:-1] if parts[-1].lower() == "tot" else parts[1:]
                if tuple(normalise_orbital(name) for name in raw_table_orbitals) != table_orbitals:
                    raise RuntimeError("PROCAR orbital columns do not match across projection tables.")

                header_source_index, kpoint_index, band_index = pending_band
                source_indices = (
                    range(source_components)
                    if interleaved_soc
                    else (header_source_index,)
                )
                for source_index in source_indices:
                    ion_table = ProcarAdapter._read_ion_table(
                        lines,
                        nions,
                        len(table_orbitals),
                    )
                    if table_seen[source_index, kpoint_index, band_index]:
                        raise RuntimeError("PROCAR contains a duplicate projection table.")
                    if band_data.spin_kind is SpinKind.SOC and source_index > 0:
                        component_index = source_index - 1
                        for element_index, (start, stop) in enumerate(element_ranges):
                            values = np.sum(
                                ion_table[start:stop],
                                axis=0,
                                dtype=np.float32,
                            )
                            resolved = spin_components[
                                component_index,
                                element_index,
                                :,
                                kpoint_index,
                                band_index,
                            ]
                            resolved.fill(0.0)
                            np.add.at(resolved, orbital_indices, values)
                    else:
                        spin_index = (
                            0
                            if band_data.spin_kind is SpinKind.SOC
                            else source_index
                        )
                        for element_index, (start, stop) in enumerate(element_ranges):
                            values = np.sum(
                                ion_table[start:stop],
                                axis=0,
                                dtype=np.float32,
                            )
                            resolved = tensor[
                                spin_index,
                                element_index,
                                :,
                                kpoint_index,
                                band_index,
                            ]
                            resolved.fill(0.0)
                            np.add.at(resolved, orbital_indices, values)

                    table_seen[source_index, kpoint_index, band_index] = True
                    parsed_tables += 1
                    if parsed_tables % stride == 0 or parsed_tables == expected_tables:
                        emit(
                            "parse",
                            parsed_tables,
                            expected_tables,
                            ProcarAdapter._table_detail(
                                parsed_tables,
                                expected_tables,
                                source_index + 1,
                                source_components,
                                kpoint_index,
                                nkpoints,
                                band_index,
                                nbands,
                            ),
                        )
                pending_band = None

        if active_component is None:
            raise RuntimeError("PROCAR is missing spin component 1.")
        ProcarAdapter._validate_component_complete(
            active_component,
            current_kpoint,
            next_kpoint,
            next_band,
            pending_band,
            nkpoints,
            nbands,
        )
        if (
            not interleaved_soc
            and source_components > 1
            and expected_component <= source_components
        ):
            raise RuntimeError(f"PROCAR is missing spin component {expected_component}.")
        if explicit_components and expected_component != source_components + 1:
            raise RuntimeError(
                "PROCAR spin component headers are missing, reused, or out of order; "
                "projection tables are invalid."
            )
        if parsed_tables != expected_tables or not np.all(table_seen):
            raise RuntimeError(
                f"PROCAR contains {parsed_tables} projection tables; expected {expected_tables}."
            )
        emit("complete", parsed_tables, expected_tables, "Finished parsing PROCAR projection tables.")
        return BandProjectionStore(
            spin_names=spin_names,
            elements=tuple(band_data.elements),
            orbitals=orbitals,
            tensor=np.ascontiguousarray(tensor),
            spin_components=(
                np.ascontiguousarray(spin_components)
                if spin_components is not None
                else None
            ),
        )

    @staticmethod
    def progress_stride(total: int) -> int:
        return max(1, (total + 99) // 100)

    @staticmethod
    def _source_component_count(band_data: ParsedBandData, spin_names: tuple[str, ...]) -> int:
        return 4 if band_data.spin_kind is SpinKind.SOC and len(spin_names) == 1 else len(spin_names)

    @staticmethod
    def _validate_kpoint_coordinates(
        kpoint_index: int,
        procar_point: np.ndarray,
        eigenval_point: np.ndarray,
    ) -> None:
        delta = procar_point - eigenval_point
        delta -= np.rint(delta)
        if not np.allclose(delta, 0.0, rtol=0.0, atol=1e-7):
            raise RuntimeError(
                f"k-point {kpoint_index + 1} coordinates do not match: PROCAR "
                f"{tuple(float(value) for value in procar_point)}; EIGENVAL "
                f"{tuple(float(value) for value in eigenval_point)}."
            )

    @staticmethod
    def _validate_band_energy(
        kpoint_index: int,
        band_index: int,
        procar_energy: float,
        eigenval_energy: float,
        spin_name: str,
        component_description: str,
    ) -> None:
        if not np.isclose(procar_energy, eigenval_energy, rtol=0.0, atol=1e-4):
            raise RuntimeError(
                f"k-point {kpoint_index + 1}, band {band_index + 1} energy mismatch "
                f"for EIGENVAL spin channel {spin_name} and {component_description}: "
                f"PROCAR {procar_energy:.6f} eV; EIGENVAL {eigenval_energy:.6f} eV."
            )

    @staticmethod
    def _validate_component_complete(
        component: int,
        current_kpoint: int | None,
        next_kpoint: int,
        next_band: int,
        pending_band: tuple[int, int, int] | None,
        nkpoints: int,
        nbands: int,
    ) -> None:
        if (
            current_kpoint is None
            or next_kpoint != nkpoints
            or next_band != nbands
            or pending_band is not None
        ):
            raise RuntimeError(f"PROCAR spin component {component} is missing projection tables.")

    @staticmethod
    def _read_metadata(path: Path) -> tuple[int, int, int, tuple[str, ...] | None]:
        dimensions: tuple[int, int, int] | None = None
        raw_orbitals: tuple[str, ...] | None = None
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if dimensions is None:
                    match = DIMENSIONS.search(line)
                    if match:
                        dimensions = tuple(int(value) for value in match.groups())
                parts = line.split()
                if parts and parts[0].lower() == "ion":
                    labels = parts[1:-1] if parts[-1].lower() == "tot" else parts[1:]
                    raw_orbitals = tuple(labels)
                if dimensions is not None and raw_orbitals is not None:
                    break
        if dimensions is None:
            raise RuntimeError("Could not read PROCAR dimensions.")
        return (*dimensions, raw_orbitals)

    @staticmethod
    def _has_spin_component_headers(path: Path) -> bool:
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if SPIN_COMPONENT.search(line):
                    return True
                parts = line.split()
                if parts and parts[0].lower() == "ion":
                    return False
        return False

    @staticmethod
    def _read_poscar_site_elements(path: Path) -> tuple[str, ...]:
        with path.open("r", errors="ignore") as handle:
            lines = [next(handle, "") for _ in range(7)]
        if not all(lines):
            raise RuntimeError("POSCAR is too short.")
        elements = tuple(lines[5].split())
        try:
            counts = tuple(int(value) for value in lines[6].split())
        except ValueError as exc:
            raise RuntimeError("Lyra Lite Mode requires VASP 5 POSCAR element names.") from exc
        if len(elements) != len(counts):
            raise RuntimeError("POSCAR element names and counts do not match.")
        return tuple(element for element, count in zip(elements, counts) for _ in range(count))

    @staticmethod
    def _element_ranges(
        site_elements: tuple[str, ...], elements: tuple[str, ...]
    ) -> tuple[tuple[int, int], ...]:
        ranges = []
        start = 0
        for element in elements:
            stop = start
            while stop < len(site_elements) and site_elements[stop] == element:
                stop += 1
            if stop == start:
                raise RuntimeError("PROCAR elements do not match EIGENVAL.")
            ranges.append((start, stop))
            start = stop
        if start != len(site_elements):
            raise RuntimeError("PROCAR elements do not match EIGENVAL.")
        return tuple(ranges)

    @staticmethod
    def _read_ion_table(
        lines: Iterator[str], nions: int, orbital_count: int
    ) -> np.ndarray:
        rows = [next(lines, "").split() for _ in range(nions)]
        try:
            parsed = np.fromstring(
                " ".join(" ".join(row[1:]) for row in rows), sep=" ", dtype=np.float32
            )
        except ValueError:
            return ProcarAdapter._read_ion_table_legacy(rows, lines, nions, orbital_count)
        if len(parsed) != nions * (orbital_count + 1):
            return ProcarAdapter._read_ion_table_legacy(rows, lines, nions, orbital_count)

        seen = np.zeros(nions, dtype=bool)
        ion_indices = np.empty(nions, dtype=np.intp)
        for row_index, row in enumerate(rows):
            try:
                ion_index = int(row[0]) - 1
            except (IndexError, ValueError) as exc:
                raise RuntimeError("Malformed ion projection row in PROCAR.") from exc
            if (
                len(row) != orbital_count + 2
                or ion_index < 0
                or ion_index >= nions
                or seen[ion_index]
            ):
                raise RuntimeError("Malformed ion projection row in PROCAR.")
            seen[ion_index] = True
            ion_indices[row_index] = ion_index

        ProcarAdapter._read_ion_summary(lines, orbital_count)

        values = np.empty((nions, orbital_count), dtype=np.float32)
        values[ion_indices] = parsed.reshape(nions, orbital_count + 1)[:, :-1]
        return values

    @staticmethod
    def _read_ion_table_legacy(
        rows: list[list[str]], lines: Iterator[str], nions: int, orbital_count: int
    ) -> np.ndarray:
        values = np.empty((nions, orbital_count), dtype=np.float32)
        seen = np.zeros(nions, dtype=bool)
        for row in rows:
            try:
                ion_index = int(row[0]) - 1
                parsed = np.fromstring(
                    " ".join(row[1 : 1 + orbital_count]), sep=" ", dtype=np.float32
                )
            except (IndexError, ValueError) as exc:
                raise RuntimeError("Malformed ion projection row in PROCAR.") from exc
            if (
                len(row) != orbital_count + 2
                or ion_index < 0
                or ion_index >= nions
                or seen[ion_index]
                or len(parsed) != orbital_count
            ):
                raise RuntimeError("Malformed ion projection row in PROCAR.")
            seen[ion_index] = True
            values[ion_index] = parsed

        ProcarAdapter._read_ion_summary(lines, orbital_count)
        return values

    @staticmethod
    def _read_ion_summary(lines: Iterator[str], orbital_count: int) -> None:

        summary = next(lines, "").split()
        try:
            summary_values = np.fromstring(
                " ".join(summary[1:]), sep=" ", dtype=np.float32
            )
        except ValueError as exc:
            raise RuntimeError("Malformed PROCAR total summary row.") from exc
        if (
            not summary
            or summary[0].lower() != "tot"
            or len(summary) != orbital_count + 2
            or len(summary_values) != orbital_count + 1
        ):
            raise RuntimeError("Malformed PROCAR total summary row.")

    @staticmethod
    def _progress_emitter(progress: Callable[[dict], None] | None) -> Callable[[str, int, int, str], None]:
        if progress is None:
            return lambda _phase, _current, _total, _detail: None

        def emit(phase: str, current: int, total: int, detail: str) -> None:
            progress({"phase": phase, "current": current, "total": total, "detail": detail})

        return emit

    @staticmethod
    def _table_detail(
        current: int,
        total: int,
        component: int,
        component_total: int,
        kpoint_index: int,
        nkpoints: int,
        band_index: int,
        nbands: int,
    ) -> str:
        return (
            f"Parsed {current}/{total} projection tables (component {component}/{component_total}, "
            f"k-point {kpoint_index + 1}/{nkpoints}, band {band_index + 1}/{nbands})."
        )

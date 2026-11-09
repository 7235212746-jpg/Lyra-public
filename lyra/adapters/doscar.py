from __future__ import annotations

from pathlib import Path

import numpy as np

from lyra.adapters.common import (
    normalise_orbital,
    orbital_group,
    read_outcar_metadata,
    read_poscar,
    read_poscar_structure,
)
from lyra.models import ParsedDosData


RESOLVED_ORBITALS = (
    "s",
    "py",
    "pz",
    "px",
    "dxy",
    "dyz",
    "dz2",
    "dxz",
    "dx2-y2",
)
F_ORBITALS = ("f1", "f2", "f3", "f4", "f5", "f6", "f7")


def _orbital_labels(count: int) -> tuple[str, ...]:
    if count == 1:
        return ("s",)
    if count == 3:
        return ("s", "p", "d")
    if count == 4:
        return ("s", "p", "d", "f")
    if count == 9:
        return RESOLVED_ORBITALS
    if count == 16:
        return RESOLVED_ORBITALS + F_ORBITALS
    raise RuntimeError(f"Unsupported DOSCAR projected orbital column count: {count}.")


class DoscarAdapter:
    @staticmethod
    def load(folder: str | Path) -> ParsedDosData:
        folder = Path(folder)
        metadata = read_outcar_metadata(folder / "OUTCAR")
        structure = read_poscar_structure(folder / "POSCAR")
        _, elements, _ = read_poscar(folder / "POSCAR")
        site_elements = structure.atom_elements
        lines = (folder / "DOSCAR").read_text(errors="ignore").splitlines()
        try:
            nions = int(lines[0].split()[0])
            total_header = lines[5].split()
            nedos = int(float(total_header[2]))
            doscar_efermi = float(total_header[3])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Could not read DOSCAR dimensions.") from exc
        if nions != len(site_elements):
            raise RuntimeError(
                f"DOSCAR contains {nions} ions but POSCAR contains {len(site_elements)}."
            )
        if not np.isclose(doscar_efermi, metadata.efermi, atol=1e-4, rtol=0.0):
            raise RuntimeError(
                "DOSCAR E-fermi does not match OUTCAR E-fermi "
                f"({doscar_efermi:.6f} vs {metadata.efermi:.6f} eV)."
            )
        total_rows = [lines[index].split() for index in range(6, 6 + nedos)]
        collinear_spin = metadata.ispin == 2 and not (
            metadata.lsorbit or metadata.noncollinear
        )
        try:
            energies = np.array([float(row[0]) for row in total_rows], dtype=float)
            if collinear_spin:
                densities = {
                    "up": np.array([float(row[1]) for row in total_rows]),
                    "down": np.array([float(row[2]) for row in total_rows]),
                }
            else:
                densities = {"total": np.array([float(row[1]) for row in total_rows])}
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Malformed total DOS rows in DOSCAR.") from exc

        projections: dict[str, dict[str, np.ndarray]] = {}
        orbital_names: set[str] = set()

        def add(key: str, spin: str, values: np.ndarray) -> None:
            spin_arrays = projections.setdefault(key, {})
            target = spin_arrays.setdefault(spin, np.zeros(nedos, dtype=float))
            target += values

        line_index = 6 + nedos
        has_projected_blocks = any(line.strip() for line in lines[line_index:])
        ion_indices = range(nions) if has_projected_blocks else ()
        for ion_index in ion_indices:
            if line_index >= len(lines):
                raise RuntimeError("Incomplete projected DOS blocks in DOSCAR.")
            header = lines[line_index].split()
            if len(header) < 4:
                raise RuntimeError("Incomplete projected DOS blocks in DOSCAR.")
            line_index += 1
            if line_index + nedos > len(lines):
                raise RuntimeError("Incomplete projected DOS blocks in DOSCAR.")
            rows = [lines[index].split() for index in range(line_index, line_index + nedos)]
            line_index += nedos
            if len(rows) != nedos or not rows:
                raise RuntimeError("Incomplete projected DOS blocks in DOSCAR.")
            row_value_counts = {len(row) - 1 for row in rows}
            if len(row_value_counts) != 1:
                raise RuntimeError("Inconsistent projected DOS row widths in DOSCAR.")
            value_count = row_value_counts.pop()
            if collinear_spin:
                component_count = 2
                spin_names = ("up", "down")
            else:
                component_count = 4 if metadata.lsorbit or metadata.noncollinear else 1
                spin_names = ("total",)
            if value_count % component_count:
                raise RuntimeError(
                    "DOSCAR projection columns do not match the expected spin layout."
                )
            orbital_count = value_count // component_count
            orbitals = tuple(normalise_orbital(name) for name in _orbital_labels(orbital_count))
            orbital_names.update(orbitals)
            element = site_elements[ion_index]
            try:
                block_energies = np.array([float(row[0]) for row in rows], dtype=float)
                numeric = np.array(
                    [[float(value) for value in row[1:]] for row in rows], dtype=float
                )
            except ValueError as exc:
                raise RuntimeError("Malformed projected DOS rows in DOSCAR.") from exc
            if not np.allclose(block_energies, energies, atol=1e-7, rtol=0.0):
                raise RuntimeError(
                    "Projected DOS energy axis does not match the total DOS energy axis."
                )
            for spin_offset, spin_name in enumerate(spin_names):
                values_by_orbital = numeric[:, spin_offset::component_count]
                add(element, spin_name, values_by_orbital.sum(axis=1))
                for orbital_index, orbital in enumerate(orbitals):
                    values = values_by_orbital[:, orbital_index]
                    add(f"{element}:{orbital}", spin_name, values)
                    group = orbital_group(orbital)
                    if group != orbital:
                        add(f"{element}:{group}", spin_name, values)
        if has_projected_blocks and any(line.strip() for line in lines[line_index:]):
            raise RuntimeError("DOSCAR contains extra projected DOS blocks.")
        return ParsedDosData(
            folder=folder,
            energies=energies,
            densities=densities,
            efermi=metadata.efermi,
            spin_kind=metadata.spin_kind,
            elements=elements,
            projections=projections,
            orbitals=tuple(sorted(orbital_names)),
            structure=structure,
        )

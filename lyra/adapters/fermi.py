from __future__ import annotations

from pathlib import Path

import numpy as np

from lyra.adapters.common import read_outcar_metadata, read_poscar
from lyra.adapters.eigenval import EigenvalAdapter
from lyra.models import ParsedFermiData


GRID_ERROR = (
    "Fermi-surface data must contain a complete three-dimensional uniform k-point "
    "grid. Line-mode or symmetry-reduced meshes are not sufficient; provide an "
    "explicit full regular mesh or disable symmetry reduction with ISYM=-1."
)


def _rounded_unique(values: np.ndarray) -> np.ndarray:
    rounded = np.round(np.mod(values, 1.0), decimals=10)
    rounded[np.isclose(rounded, 1.0, atol=1e-10)] = 0.0
    return np.unique(rounded)


def regular_grid_from_kpoints(
    kpoints: np.ndarray,
    energies: dict[str, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, np.ndarray],
]:
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(GRID_ERROR)
    normalized = np.round(np.mod(points, 1.0), decimals=10)
    normalized[np.isclose(normalized, 1.0, atol=1e-10)] = 0.0
    axes = tuple(_rounded_unique(normalized[:, index]) for index in range(3))
    shape = tuple(len(axis) for axis in axes)
    if any(size < 2 for size in shape) or int(np.prod(shape)) != len(points):
        raise ValueError(GRID_ERROR)
    for axis in axes:
        periodic_gaps = np.diff(np.concatenate((axis, (axis[0] + 1.0,))))
        if not np.allclose(
            periodic_gaps,
            periodic_gaps[0],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(GRID_ERROR)

    indices = np.empty((len(points), 3), dtype=int)
    for dimension, axis in enumerate(axes):
        nearest = np.argmin(
            np.abs(normalized[:, dimension, None] - axis[None, :]),
            axis=1,
        )
        if np.any(
            np.abs(normalized[:, dimension] - axis[nearest]) > 1e-7
        ):
            raise ValueError(GRID_ERROR)
        indices[:, dimension] = nearest
    if len({tuple(index) for index in indices}) != len(points):
        raise ValueError(GRID_ERROR)

    gridded = {}
    for spin, values in energies.items():
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(points):
            raise ValueError("Fermi-surface eigenvalue dimensions do not match k-points.")
        grid = np.empty(shape + (matrix.shape[1],), dtype=float)
        for point_index, grid_index in enumerate(indices):
            grid[tuple(grid_index)] = matrix[point_index]
        gridded[spin] = grid
    return axes, gridded


def _line_mode(path: Path) -> bool:
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return len(lines) >= 3 and "line" in lines[2].lower()
    except OSError:
        return False


class FermiAdapter:
    @staticmethod
    def load_lite(folder: str | Path) -> ParsedFermiData:
        folder = Path(folder)
        if _line_mode(folder / "KPOINTS"):
            raise ValueError(
                "Line-mode KPOINTS cannot produce a Fermi surface; select a complete "
                "uniform three-dimensional k-point calculation."
            )
        metadata = read_outcar_metadata(folder / "OUTCAR")
        reciprocal, elements, _ = read_poscar(folder / "POSCAR")
        effective_ispin = (
            1
            if metadata.lsorbit or metadata.noncollinear
            else metadata.ispin
        )
        kpoints, energies, _occupations = EigenvalAdapter._read_eigenval(
            folder / "EIGENVAL",
            effective_ispin,
        )
        axes, gridded = regular_grid_from_kpoints(kpoints, energies)
        return ParsedFermiData(
            folder=folder,
            axes=axes,
            energies=gridded,
            efermi=metadata.efermi,
            reciprocal_lattice=reciprocal,
            spin_kind=metadata.spin_kind,
            elements=elements,
        )

    @staticmethod
    def load_full(folder: str | Path) -> ParsedFermiData:
        from lyra.adapters.vasprun import (
            _import_pymatgen,
            _ordered_elements,
            _spin_kind,
            _spin_names,
            _xml_path,
        )

        folder = Path(folder)
        Vasprun, _ = _import_pymatgen()
        vasprun = Vasprun(
            _xml_path(folder),
            parse_dos=False,
            parse_eigen=True,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )
        band_structure = vasprun.get_band_structure(line_mode=False)
        spin_map = _spin_names(band_structure.bands)
        kpoints = np.asarray(
            [point.frac_coords for point in band_structure.kpoints],
            dtype=float,
        )
        energies = {
            spin_map[spin]: np.asarray(values, dtype=float).T
            for spin, values in band_structure.bands.items()
        }
        axes, gridded = regular_grid_from_kpoints(kpoints, energies)
        reciprocal = np.asarray(band_structure.lattice_rec.matrix, dtype=float)
        return ParsedFermiData(
            folder=folder,
            axes=axes,
            energies=gridded,
            efermi=float(getattr(band_structure, "efermi", vasprun.efermi)),
            reciprocal_lattice=reciprocal,
            spin_kind=_spin_kind(vasprun, len(energies)),
            elements=_ordered_elements(band_structure.structure.sites),
        )

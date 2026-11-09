from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from lyra.adapters.common import (
    build_segmented_axis,
    parse_line_mode_kpoints,
    read_poscar_kpoint_scale,
    read_outcar_metadata,
    read_poscar,
    read_poscar_structure,
    reconstruct_line_mode_kpoints,
    validate_line_mode_kpoints,
)
from lyra.models import ParsedBandData


class EigenvalAdapter:
    @staticmethod
    def load(folder: str | Path) -> ParsedBandData:
        folder = Path(folder)
        metadata = read_outcar_metadata(folder / "OUTCAR")
        structure = read_poscar_structure(folder / "POSCAR")
        reciprocal, elements, _ = read_poscar(folder / "POSCAR")
        effective_ispin = (
            1
            if metadata.lsorbit or metadata.noncollinear
            else metadata.ispin
        )
        kpoints, energies, occupations = EigenvalAdapter._read_eigenval(
            folder / "EIGENVAL",
            effective_ispin,
        )
        line_mode = parse_line_mode_kpoints(folder / "KPOINTS")
        cartesian_scale = (
            read_poscar_kpoint_scale(folder / "POSCAR")
            if line_mode.coordinate_mode == "cartesian"
            else None
        )
        expected_kpoints = reconstruct_line_mode_kpoints(
            line_mode,
            reciprocal,
            cartesian_scale,
        )
        validate_line_mode_kpoints(line_mode, expected_kpoints, kpoints)
        segments = tuple(
            (segment.start.label, segment.end.label)
            for segment in line_mode.segments
        )
        distances, ranges, ticks, labels = build_segmented_axis(
            expected_kpoints,
            reciprocal,
            line_mode.points_per_segment,
            segments,
        )
        return ParsedBandData(
            folder=folder,
            kpoints=kpoints,
            distances=distances,
            segment_ranges=ranges,
            tick_positions=ticks,
            tick_labels=labels,
            energies=energies,
            occupations=occupations,
            efermi=metadata.efermi,
            spin_kind=metadata.spin_kind,
            elements=elements,
            structure=structure,
        )

    @staticmethod
    def _read_eigenval(
        path: Path, ispin: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        lines = path.read_text(errors="ignore").splitlines()
        try:
            header = lines[5].split()
            nkpoints = int(header[1])
            nbands = int(header[2])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Could not read EIGENVAL dimensions.") from exc
        kpoints = np.zeros((nkpoints, 3), dtype=float)
        if ispin == 2:
            energies = {
                "up": np.zeros((nkpoints, nbands), dtype=float),
                "down": np.zeros((nkpoints, nbands), dtype=float),
            }
            occupations = {
                "up": np.zeros((nkpoints, nbands), dtype=float),
                "down": np.zeros((nkpoints, nbands), dtype=float),
            }
        else:
            energies = {"total": np.zeros((nkpoints, nbands), dtype=float)}
            occupations = {"total": np.zeros((nkpoints, nbands), dtype=float)}
        line_index = 6
        for kpoint_index in range(nkpoints):
            while line_index < len(lines) and not lines[line_index].split():
                line_index += 1
            if line_index >= len(lines):
                raise RuntimeError(
                    f"EIGENVAL k-point {kpoint_index + 1} row is missing."
                )
            try:
                point = lines[line_index].split()
                kpoints[kpoint_index] = [float(value) for value in point[:3]]
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Malformed EIGENVAL near k-point {kpoint_index + 1}.") from exc
            if len(point) < 3 or not np.all(np.isfinite(kpoints[kpoint_index])):
                raise RuntimeError(
                    f"Malformed EIGENVAL near k-point {kpoint_index + 1}."
                )
            line_index += 1
            for band_index in range(nbands):
                row_number = band_index + 1
                if line_index >= len(lines) or not lines[line_index].split():
                    raise RuntimeError(
                        f"EIGENVAL k-point {kpoint_index + 1}, band row "
                        f"{row_number} is missing."
                    )
                parts = lines[line_index].split()
                band_id_text = parts[0]
                if re.fullmatch(r"[+-]?\d+", band_id_text) is None:
                    raise RuntimeError(
                        f"EIGENVAL k-point {kpoint_index + 1}, band row "
                        f"{row_number} requires an integer band ID; found "
                        f"{band_id_text!r}."
                    )
                band_id = int(band_id_text)
                if band_id != row_number:
                    raise RuntimeError(
                        f"EIGENVAL k-point {kpoint_index + 1}, band row "
                        f"{row_number} expected ID {row_number}, found {band_id}."
                    )
                try:
                    if ispin == 2:
                        energies["up"][kpoint_index, band_index] = float(parts[1])
                        energies["down"][kpoint_index, band_index] = float(parts[2])
                        occupations["up"][kpoint_index, band_index] = float(parts[3])
                        occupations["down"][kpoint_index, band_index] = float(parts[4])
                    else:
                        energies["total"][kpoint_index, band_index] = float(parts[1])
                        occupations["total"][kpoint_index, band_index] = float(parts[2])
                except (IndexError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed EIGENVAL energy at k-point {kpoint_index + 1}, "
                        f"band row {row_number}."
                    ) from exc
                line_index += 1
        return kpoints, energies, occupations

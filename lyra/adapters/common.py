from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from lyra.models import ParsedStructure, SpinKind


# VASP text k-point coordinates are normally printed to at least six decimals.
# This allows one final-place rounding unit without accepting a shifted path.
KPOINT_COORDINATE_ATOL = 1e-6


@dataclass(frozen=True)
class OutcarMetadata:
    efermi: float
    ispin: int
    lsorbit: bool
    noncollinear: bool

    @property
    def spin_kind(self) -> SpinKind:
        if self.lsorbit or self.noncollinear:
            return SpinKind.SOC
        if self.ispin == 2:
            return SpinKind.COLLINEAR
        return SpinKind.NON_SPIN


@dataclass(frozen=True)
class KpointEndpoint:
    coordinates: tuple[float, float, float]
    label: str


@dataclass(frozen=True)
class KpointSegment:
    start: KpointEndpoint
    end: KpointEndpoint


@dataclass(frozen=True)
class LineModeKpoints:
    points_per_segment: int
    coordinate_mode: str
    segments: tuple[KpointSegment, ...]
    source: str


def read_outcar_metadata(path: str | Path) -> OutcarMetadata:
    efermi = None
    ispin = 1
    lsorbit = False
    noncollinear = False
    efermi_pattern = re.compile(r"E-fermi\s*:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)")
    int_pattern = re.compile(r"\bISPIN\s*=\s*(\d+)")
    bool_patterns = {
        "lsorbit": re.compile(r"\bLSORBIT\s*=\s*([TF])", re.IGNORECASE),
        "noncollinear": re.compile(r"\bLNONCOLLINEAR\s*=\s*([TF])", re.IGNORECASE),
    }
    with Path(path).open("r", errors="ignore") as handle:
        for line in handle:
            match = efermi_pattern.search(line)
            if match:
                efermi = float(match.group(1))
            match = int_pattern.search(line)
            if match:
                ispin = int(match.group(1))
            match = bool_patterns["lsorbit"].search(line)
            if match:
                lsorbit = match.group(1).upper() == "T"
            match = bool_patterns["noncollinear"].search(line)
            if match:
                noncollinear = match.group(1).upper() == "T"
    if efermi is None:
        raise RuntimeError("Could not find E-fermi in OUTCAR.")
    return OutcarMetadata(efermi, ispin, lsorbit, noncollinear)


def _parse_poscar_header(
    path: str | Path,
) -> tuple[
    Path,
    list[str],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[int, ...],
]:
    poscar_path = Path(path)
    lines = poscar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise RuntimeError("POSCAR is too short.")
    try:
        scales = np.asarray([float(value) for value in lines[1].split()], dtype=float)
        raw_lattice = np.asarray(
            [
                [float(value) for value in lines[index].split()[:3]]
                for index in range(2, 5)
            ],
            dtype=float,
        )
    except (IndexError, ValueError) as exc:
        raise RuntimeError("POSCAR contains invalid lattice scaling data.") from exc
    if raw_lattice.shape != (3, 3) or not np.all(np.isfinite(raw_lattice)):
        raise RuntimeError("POSCAR contains invalid lattice data.")
    if scales.size == 1:
        scale = float(scales[0])
        if not np.isfinite(scale) or scale == 0.0:
            raise RuntimeError("POSCAR universal scale factor must be finite and non-zero.")
        if scale < 0.0:
            raw_volume = abs(float(np.linalg.det(raw_lattice)))
            if raw_volume <= 1e-12:
                raise RuntimeError("POSCAR lattice must have non-zero volume.")
            scale = ((-scale) / raw_volume) ** (1.0 / 3.0)
        lattice = raw_lattice * scale
        cartesian_scale = np.full(3, scale)
    elif scales.size == 3:
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise RuntimeError("POSCAR Cartesian scale factors must be finite and positive.")
        lattice = raw_lattice * scales
        cartesian_scale = scales
    else:
        raise RuntimeError("POSCAR scaling line must contain one or three values.")
    if not np.all(np.isfinite(lattice)) or abs(float(np.linalg.det(lattice))) <= 1e-12:
        raise RuntimeError("POSCAR lattice must have non-zero volume.")
    elements = tuple(lines[5].split())
    try:
        counts = tuple(int(value) for value in lines[6].split())
    except ValueError as exc:
        raise RuntimeError("Lyra Lite Mode requires VASP 5 POSCAR element names.") from exc
    if not elements or len(elements) != len(counts) or any(count < 0 for count in counts):
        raise RuntimeError("POSCAR element names and counts do not match.")
    return poscar_path, lines, lattice, cartesian_scale, elements, counts


def _parse_poscar(path: str | Path) -> tuple[ParsedStructure, tuple[str, ...], tuple[int, ...]]:
    (
        poscar_path,
        lines,
        lattice,
        cartesian_scale,
        elements,
        counts,
    ) = _parse_poscar_header(path)
    position_line = 7
    if position_line < len(lines) and lines[position_line].strip().lower().startswith("s"):
        position_line += 1
    if position_line >= len(lines):
        raise RuntimeError("POSCAR coordinate mode must be Direct or Cartesian.")
    coordinate_mode = lines[position_line].strip().lower()
    if coordinate_mode.startswith("d"):
        direct = True
    elif coordinate_mode.startswith(("c", "k")):
        direct = False
    else:
        raise RuntimeError("POSCAR coordinate mode must be Direct or Cartesian.")
    atom_count = sum(counts)
    coordinate_lines = lines[position_line + 1 : position_line + 1 + atom_count]
    if len(coordinate_lines) != atom_count:
        raise RuntimeError("POSCAR does not contain every atom position.")
    try:
        positions = np.asarray(
            [[float(value) for value in line.split()[:3]] for line in coordinate_lines],
            dtype=float,
        ).reshape(atom_count, 3)
    except ValueError as exc:
        raise RuntimeError("POSCAR contains invalid atom coordinates.") from exc
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("POSCAR contains invalid atom coordinates.")
    if direct:
        fractional_positions = positions
    else:
        fractional_positions = (positions * cartesian_scale) @ np.linalg.inv(lattice)
    atom_elements = tuple(element for element, count in zip(elements, counts) for _ in range(count))
    return (
        ParsedStructure(
            lattice=lattice,
            atom_elements=atom_elements,
            fractional_positions=fractional_positions,
            source=str(poscar_path),
        ),
        elements,
        counts,
    )


def read_poscar_structure(path: str | Path) -> ParsedStructure:
    return _parse_poscar(path)[0]


def read_poscar(path: str | Path) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    _path, _lines, lattice, _cartesian_scale, elements, counts = _parse_poscar_header(path)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    return reciprocal, elements, counts


def read_poscar_kpoint_scale(path: str | Path) -> float:
    """Return VASP's universal ``a`` for Cartesian KPOINTS coordinates."""
    _path, lines, _lattice, cartesian_scale, _elements, _counts = (
        _parse_poscar_header(path)
    )
    try:
        declared_scales = [float(value) for value in lines[1].split()]
    except (IndexError, ValueError) as exc:
        raise RuntimeError("POSCAR contains invalid lattice scaling data.") from exc
    if len(declared_scales) != 1:
        raise RuntimeError(
            "Cartesian line-mode KPOINTS cannot be verified because POSCAR "
            "uses three component scale factors and has no single documented "
            "universal scale a."
        )
    return float(cartesian_scale[0])


def normalise_label(label: str) -> str:
    return label.replace("\\Gamma", "G").replace("Gamma", "G").replace("Γ", "G")


def normalise_orbital(name: str) -> str:
    aliases = {"dx2": "dx2-y2", "x2-y2": "dx2-y2", "z2": "dz2"}
    return aliases.get(name, name)


def orbital_group(name: str) -> str:
    if name == "s":
        return "s"
    if name in {"p", "px", "py", "pz"}:
        return "p"
    if name == "d" or name.startswith("d") or name in {"x2-y2", "z2"}:
        return "d"
    if name == "f" or name.startswith("f"):
        return "f"
    return name


def parse_line_mode_kpoints(path: str | Path) -> LineModeKpoints:
    kpoints_path = Path(path)
    lines = kpoints_path.read_text(errors="ignore").splitlines()
    if len(lines) < 6:
        raise RuntimeError("KPOINTS is too short for a line-mode path.")
    try:
        points_per_segment = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Could not read points per segment from KPOINTS.") from exc
    if points_per_segment < 2:
        raise RuntimeError(
            "KPOINTS line mode requires at least two points per segment."
        )
    if not lines[2].lstrip().lower().startswith("l"):
        raise RuntimeError("KPOINTS is not a line-mode path.")
    coordinate_line = lines[3].strip()
    if not coordinate_line:
        raise RuntimeError("KPOINTS line-mode coordinate convention is missing.")
    coordinate_mode = (
        "cartesian"
        if coordinate_line[0].lower() in {"c", "k"}
        else "reciprocal"
    )

    endpoints = []
    for line_number, line in enumerate(lines[4:], start=5):
        if not line.strip():
            continue
        coordinate_text, separator, comment = line.partition("!")
        parts = coordinate_text.split()
        if len(parts) < 3:
            raise RuntimeError(
                f"KPOINTS endpoint on line {line_number} must contain three coordinates."
            )
        try:
            coordinates_array = np.asarray(
                [float(value) for value in parts[:3]],
                dtype=float,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"KPOINTS endpoint on line {line_number} contains invalid coordinates."
            ) from exc
        if not np.all(np.isfinite(coordinates_array)):
            raise RuntimeError(
                f"KPOINTS endpoint on line {line_number} contains invalid coordinates."
            )
        label = comment.strip() if separator else " ".join(parts[3:]).strip()
        endpoints.append(
            KpointEndpoint(
                tuple(float(value) for value in coordinates_array),
                normalise_label(label),
            )
        )
    if len(endpoints) < 2 or len(endpoints) % 2:
        raise RuntimeError("KPOINTS endpoint lines must form endpoint pairs.")
    segments = tuple(
        KpointSegment(endpoints[index], endpoints[index + 1])
        for index in range(0, len(endpoints), 2)
    )
    return LineModeKpoints(
        points_per_segment=points_per_segment,
        coordinate_mode=coordinate_mode,
        segments=segments,
        source=str(kpoints_path),
    )


def reconstruct_line_mode_kpoints(
    definition: LineModeKpoints,
    reciprocal: np.ndarray,
    cartesian_scale: float | None = None,
) -> np.ndarray:
    reciprocal_array = np.asarray(reciprocal, dtype=float)
    if reciprocal_array.shape != (3, 3) or not np.all(np.isfinite(reciprocal_array)):
        raise RuntimeError("POSCAR reciprocal lattice is invalid.")
    endpoint_coordinates = np.asarray(
        [
            endpoint.coordinates
            for segment in definition.segments
            for endpoint in (segment.start, segment.end)
        ],
        dtype=float,
    )
    if definition.coordinate_mode == "cartesian":
        if cartesian_scale is None or not np.isfinite(cartesian_scale) or cartesian_scale <= 0:
            raise RuntimeError(
                "Cartesian line-mode KPOINTS requires a finite positive POSCAR "
                "universal scale."
            )
        physical_coordinates = endpoint_coordinates * (2.0 * np.pi / cartesian_scale)
        endpoint_coordinates = physical_coordinates @ np.linalg.inv(reciprocal_array)
    points = []
    for segment_index in range(len(definition.segments)):
        start = endpoint_coordinates[2 * segment_index]
        end = endpoint_coordinates[2 * segment_index + 1]
        points.append(
            np.linspace(
                start,
                end,
                definition.points_per_segment,
                endpoint=True,
            )
        )
    return np.concatenate(points, axis=0)


def validate_line_mode_kpoints(
    definition: LineModeKpoints,
    expected: np.ndarray,
    eigenval_kpoints: np.ndarray,
    *,
    atol: float = KPOINT_COORDINATE_ATOL,
) -> None:
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(eigenval_kpoints, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise RuntimeError(
            f"KPOINTS defines {len(definition.segments)} segments; expected "
            f"{len(expected_array)} k-points, found {len(actual_array)} in EIGENVAL."
        )
    if not np.all(np.isfinite(actual_array)):
        raise RuntimeError("EIGENVAL contains a non-finite k-point coordinate.")
    periodic_delta = expected_array - actual_array
    periodic_delta -= np.round(periodic_delta)
    mismatches = np.argwhere(np.max(np.abs(periodic_delta), axis=1) > atol)
    if not len(mismatches):
        return
    global_index = int(mismatches[0, 0])
    segment_index, point_index = divmod(
        global_index,
        definition.points_per_segment,
    )
    expected_text = ", ".join(f"{value:.8g}" for value in expected_array[global_index])
    actual_text = ", ".join(f"{value:.8g}" for value in actual_array[global_index])
    raise RuntimeError(
        "KPOINTS and EIGENVAL k-point coordinates disagree at "
        f"segment {segment_index + 1}, point {point_index + 1}: "
        f"KPOINTS=({expected_text}), EIGENVAL=({actual_text}); "
        f"periodic reciprocal tolerance is {atol:.0e}."
    )


def parse_kpoints_segments(path: str | Path) -> tuple[int, tuple[tuple[str, str], ...]]:
    lines = Path(path).read_text(errors="ignore").splitlines()
    try:
        n_per_segment = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Could not read points per segment from KPOINTS.") from exc
    endpoint_lines = [line for line in lines[4:] if len(line.split()) >= 3]
    if len(endpoint_lines) < 2 or len(endpoint_lines) % 2:
        raise RuntimeError("KPOINTS endpoint lines must form endpoint pairs.")

    def label(line: str) -> str:
        if "!" in line:
            value = line.rsplit("!", 1)[1].strip()
        else:
            parts = line.split()
            value = parts[-1] if len(parts) >= 4 else ""
        return normalise_label(value)

    segments = tuple(
        (label(endpoint_lines[index]), label(endpoint_lines[index + 1]))
        for index in range(0, len(endpoint_lines), 2)
    )
    return n_per_segment, segments


def build_segmented_axis(
    kpoints: np.ndarray,
    reciprocal: np.ndarray,
    n_per_segment: int,
    segments: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[float, ...], tuple[str, ...]]:
    expected = n_per_segment * len(segments)
    if len(kpoints) != expected:
        raise RuntimeError(f"KPOINTS defines {len(segments)} segments; expected {expected} k-points, found {len(kpoints)}.")
    distances = np.empty(len(kpoints), dtype=float)
    ranges = []
    offset = 0.0
    for segment_index in range(len(segments)):
        start = segment_index * n_per_segment
        stop = start + n_per_segment
        distances[start] = offset
        for point_index in range(start + 1, stop):
            delta = np.dot(kpoints[point_index] - kpoints[point_index - 1], reciprocal)
            offset += float(np.linalg.norm(delta))
            distances[point_index] = offset
        ranges.append((start, stop))
    ticks = [float(distances[0])]
    labels = [segments[0][0]]
    for index, (start_label, end_label) in enumerate(segments):
        if index and start_label != segments[index - 1][1]:
            labels[-1] = labels[-1] + "|" + start_label
        ticks.append(float(distances[ranges[index][1] - 1]))
        labels.append(end_label)
    return distances, tuple(ranges), tuple(ticks), tuple(labels)

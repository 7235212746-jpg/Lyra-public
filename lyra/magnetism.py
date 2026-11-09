from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import tempfile

import numpy as np

from lyra.adapters.common import read_poscar_structure


_MAGNETIZATION_START = re.compile(r"^\s*magnetization\s*\(\s*([xyz])\s*\)", re.IGNORECASE)
_SAXIS = re.compile(
    r"\bSAXIS\s*=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
)
_NONCOLLINEAR = re.compile(r"\b(?:LSORBIT|LNONCOLLINEAR)\s*=\s*T", re.IGNORECASE)


def _finite_array(value: object, *, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(array.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MagneticMomentData:
    """Final VASP on-site projected magnetic moments, expressed in Bohr magnetons."""

    lattice: np.ndarray
    atom_elements: tuple[str, ...]
    fractional_positions: np.ndarray
    spinor_vectors: np.ndarray
    cartesian_vectors: np.ndarray
    saxis: np.ndarray
    is_noncollinear: bool

    def __post_init__(self) -> None:
        lattice = _finite_array(self.lattice, name="lattice", shape=(3, 3))
        fractional_positions = _finite_array(
            self.fractional_positions, name="fractional_positions", shape=(None, 3)
        )
        spinor_vectors = _finite_array(self.spinor_vectors, name="spinor_vectors", shape=(None, 3))
        cartesian_vectors = _finite_array(
            self.cartesian_vectors, name="cartesian_vectors", shape=(None, 3)
        )
        saxis = _finite_array(self.saxis, name="saxis", shape=(3,))
        if abs(float(np.linalg.det(lattice))) <= 1e-12:
            raise ValueError("lattice must have non-zero volume.")
        if np.linalg.norm(saxis) <= 1e-12:
            raise ValueError("saxis must have non-zero length.")
        count = len(self.atom_elements)
        if not all(isinstance(element, str) and element for element in self.atom_elements):
            raise ValueError("atom_elements must contain non-empty element symbols.")
        if any(array.shape[0] != count for array in (fractional_positions, spinor_vectors, cartesian_vectors)):
            raise ValueError("magnetic data atom counts must match.")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", fractional_positions)
        object.__setattr__(self, "spinor_vectors", spinor_vectors)
        object.__setattr__(self, "cartesian_vectors", cartesian_vectors)
        object.__setattr__(self, "saxis", saxis / np.linalg.norm(saxis))


def spinor_to_cartesian(vectors: object, saxis: object) -> np.ndarray:
    """Rotate VASP SAXIS spinor coordinates with ``Rz(alpha) @ Ry(beta)``."""

    spinor = _finite_array(vectors, name="vectors", shape=(None, 3))
    axis = _finite_array(saxis, name="saxis", shape=(3,))
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("saxis must have non-zero length.")
    axis = axis / norm
    alpha = math.atan2(float(axis[1]), float(axis[0]))
    beta = math.acos(float(np.clip(axis[2], -1.0, 1.0)))
    rz = np.array(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [math.cos(beta), 0.0, math.sin(beta)],
            [0.0, 1.0, 0.0],
            [-math.sin(beta), 0.0, math.cos(beta)],
        ]
    )
    return spinor @ (rz @ ry).T


def _read_table(lines: list[str], start: int) -> tuple[int, tuple[float, ...], bool]:
    moments: list[float] = []
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if _MAGNETIZATION_START.match(line):
            return index, tuple(moments), False
        fields = line.split()
        if fields and fields[0].lower() == "tot":
            return index + 1, tuple(moments), bool(moments)
        if fields and fields[0].isdigit():
            if int(fields[0]) != len(moments) + 1:
                return index + 1, tuple(moments), False
            try:
                moment = float(fields[-1])
            except ValueError:
                return index + 1, tuple(moments), False
            if not math.isfinite(moment):
                return index + 1, tuple(moments), False
            moments.append(moment)
        index += 1
    return index, tuple(moments), False


def _magnetization_blocks(lines: list[str]) -> list[tuple[str, tuple[float, ...], bool]]:
    blocks: list[tuple[str, tuple[float, ...], bool]] = []
    index = 0
    while index < len(lines):
        match = _MAGNETIZATION_START.match(lines[index])
        if match is None:
            index += 1
            continue
        next_index, moments, complete = _read_table(lines, index)
        blocks.append((match.group(1).lower(), moments, complete))
        index = next_index
    return blocks


def _last_complete_vector_group(
    blocks: list[tuple[str, tuple[float, ...], bool]],
) -> tuple[tuple[str, tuple[float, ...], bool], ...]:
    groups: list[list[tuple[str, tuple[float, ...], bool]]] = []
    current: list[tuple[str, tuple[float, ...], bool]] | None = None
    for block in blocks:
        if block[0] == "x":
            if current is not None:
                groups.append(current)
            current = [block]
        elif current is not None:
            current.append(block)
    if current is not None:
        groups.append(current)
    for group in reversed(groups):
        if (
            tuple(block[0] for block in group) == ("x", "y", "z")
            and all(block[2] for block in group)
        ):
            return tuple(group)
    raise RuntimeError("OUTCAR does not contain a complete magnetization (x, y, z) group.")


def _read_saxis(text: str) -> np.ndarray:
    matches = list(_SAXIS.finditer(text))
    if not matches:
        return np.array([0.0, 0.0, 1.0])
    values = np.asarray([float(value) for value in matches[-1].groups()], dtype=float)
    if not np.all(np.isfinite(values)) or np.linalg.norm(values) <= 1e-12:
        raise RuntimeError("OUTCAR SAXIS must be a finite non-zero vector.")
    return values / np.linalg.norm(values)


def read_magnetic_moments(folder: str | Path) -> MagneticMomentData:
    """Read the final complete VASP on-site projected magnetic-moment table."""

    folder_path = Path(folder)
    structure = read_poscar_structure(folder_path / "POSCAR")
    lattice = structure.lattice
    atom_elements = structure.atom_elements
    fractional_positions = structure.fractional_positions
    outcar_path = folder_path / "OUTCAR"
    text = outcar_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    blocks = _magnetization_blocks(lines)
    if not blocks:
        raise RuntimeError("OUTCAR does not contain a complete magnetization table.")
    is_noncollinear = bool(_NONCOLLINEAR.search(text))
    saxis = _read_saxis(text)
    if is_noncollinear:
        final_blocks = _last_complete_vector_group(blocks)
        component_vectors = [block[1] for block in final_blocks]
        if any(len(component) != len(atom_elements) for component in component_vectors):
            raise RuntimeError("OUTCAR magnetization atom count does not match POSCAR.")
        spinor_vectors = np.asarray(component_vectors, dtype=float).T
        cartesian_vectors = spinor_to_cartesian(spinor_vectors, saxis)
    else:
        try:
            _, moments, _ = next(
                block for block in reversed(blocks) if block[0] == "x" and block[2]
            )
        except StopIteration as exc:
            raise RuntimeError("OUTCAR final complete magnetization (x) block is missing.") from exc
        if len(moments) != len(atom_elements):
            raise RuntimeError("OUTCAR magnetization atom count does not match POSCAR.")
        spinor_vectors = np.column_stack((np.zeros(len(moments)), np.zeros(len(moments)), moments))
        cartesian_vectors = spinor_vectors.copy()
        saxis = np.array([0.0, 0.0, 1.0])
    return MagneticMomentData(
        lattice=lattice,
        atom_elements=atom_elements,
        fractional_positions=fractional_positions,
        spinor_vectors=spinor_vectors,
        cartesian_vectors=cartesian_vectors,
        saxis=saxis,
        is_noncollinear=is_noncollinear,
    )


def _atom_labels(elements: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
        labels.append(f"{element}{counts[element]}")
    return tuple(labels)


def _cell_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    lengths = np.linalg.norm(lattice, axis=1)

    def angle(first: int, second: int) -> float:
        cosine = float(np.dot(lattice[first], lattice[second]) / (lengths[first] * lengths[second]))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    return (
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        angle(1, 2),
        angle(0, 2),
        angle(0, 1),
    )


def _prepare_temp_text(path: Path, text: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _reserve_backup_path(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".bak",
        delete=False,
    ) as backup:
        return Path(backup.name)


def _replace_output_pair(
    prepared_paths: tuple[Path, Path],
    target_paths: tuple[Path, Path],
) -> None:
    backup_paths: list[Path] = []
    backups: dict[Path, Path] = {}
    installed: set[Path] = set()
    retained_backups: set[Path] = set()
    try:
        for target in target_paths:
            if target.exists() and not target.is_file():
                raise IsADirectoryError(f"output target is not a file: {target}")
        for target in target_paths:
            if target.exists():
                backup = _reserve_backup_path(target)
                backup_paths.append(backup)
                target.replace(backup)
                backups[target] = backup
        for prepared, target in zip(prepared_paths, target_paths):
            try:
                prepared.replace(target)
            except Exception:
                if not prepared.exists() and target.exists():
                    installed.add(target)
                raise
            installed.add(target)
    except Exception as exc:
        rollback_errors: list[OSError] = []
        for target in installed:
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for target, backup in backups.items():
            try:
                if backup.exists():
                    backup.replace(target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            retained_backups.update(backup for backup in backups.values() if backup.exists())
            raise RuntimeError("failed to roll back magnetic output pair") from exc
        raise
    finally:
        for prepared in prepared_paths:
            prepared.unlink(missing_ok=True)
        for backup in backup_paths:
            if backup not in retained_backups:
                backup.unlink(missing_ok=True)


def _cif_loop(lines: list[str], required_header: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        cursor = index + 1
        headers: list[str] = []
        while cursor < len(lines) and lines[cursor].strip().startswith("_"):
            headers.append(lines[cursor].strip())
            cursor += 1
        if required_header not in headers:
            continue
        rows: list[tuple[str, ...]] = []
        while cursor < len(lines):
            row = lines[cursor].strip()
            if not row:
                break
            if row == "loop_" or row.startswith("_") or row.startswith("data_"):
                break
            fields = tuple(row.split())
            if len(fields) != len(headers):
                raise ValueError(f"incomplete {required_header} row")
            rows.append(fields)
            cursor += 1
        return tuple(headers), tuple(rows)
    return None


def validate_magnetic_cif(path: str | Path, expected_atoms: tuple[str, ...]) -> None:
    """Check Lyra's compact magnetic-CIF subset before it is handed to VESTA."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        atom_loop = _cif_loop(lines, "_atom_site_label")
    except ValueError as exc:
        raise ValueError("complete atom-site row required") from exc
    if atom_loop is None:
        raise ValueError("missing labels: atom-site loop")
    atom_headers, atom_rows = atom_loop
    required_atom_headers = (
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    )
    if any(header not in atom_headers for header in required_atom_headers):
        raise ValueError("complete atom-site headers required")
    atom_indices = tuple(atom_headers.index(header) for header in required_atom_headers)
    atom_labels = tuple(row[atom_indices[0]] for row in atom_rows)
    if (
        any(label in {"", ".", "?"} for label in atom_labels)
        or len(set(atom_labels)) != len(atom_labels)
        or set(atom_labels) != set(expected_atoms)
    ):
        raise ValueError("missing labels: atom-site loop")
    for row in atom_rows:
        try:
            fractional = tuple(float(row[index]) for index in atom_indices[2:])
        except ValueError as exc:
            raise ValueError("finite fractional coordinates required") from exc
        if not all(math.isfinite(component) for component in fractional):
            raise ValueError("finite fractional coordinates required")

    moment_headers_present = any(line.strip().startswith("_atom_site_moment") for line in lines)
    moment_loop = _cif_loop(lines, "_atom_site_moment_label")
    if moment_loop is None:
        if moment_headers_present:
            raise ValueError("incomplete magnetic moment loop")
        return
    moment_headers, moment_rows = moment_loop
    required_headers = (
        "_atom_site_moment_label",
        "_atom_site_moment_crystalaxis_x",
        "_atom_site_moment_crystalaxis_y",
        "_atom_site_moment_crystalaxis_z",
    )
    if any(header not in moment_headers for header in required_headers):
        raise ValueError("incomplete magnetic moment loop")
    indices = tuple(moment_headers.index(header) for header in required_headers)
    moment_labels: list[str] = []
    for row in moment_rows:
        label = row[indices[0]]
        if label in {"", ".", "?"} or label not in atom_labels:
            raise ValueError("missing labels: magnetic moment loop")
        try:
            components = tuple(float(row[index]) for index in indices[1:])
        except ValueError as exc:
            raise ValueError("magnetic moment components must be finite") from exc
        if not all(math.isfinite(component) for component in components):
            raise ValueError("magnetic moment components must be finite")
        moment_labels.append(label)
    if len(set(moment_labels)) != len(moment_labels):
        raise ValueError("missing labels: duplicate magnetic moment label")


def write_vesta_magnetic_files(data: MagneticMomentData, destination: str | Path) -> tuple[Path, Path]:
    """Write a VESTA-readable magnetic CIF and a traceable moment summary."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    labels = _atom_labels(data.atom_elements)
    lengths_and_angles = _cell_parameters(data.lattice)
    unit_m = data.lattice / np.linalg.norm(data.lattice, axis=1)[:, None]
    crystal_moments = data.cartesian_vectors @ np.linalg.inv(unit_m)

    cif_lines = [
        "data_lyra_magnetic_moments",
        "_audit_creation_method 'Lyra magnetic-moment export'",
        f"_cell_length_a {lengths_and_angles[0]:.8f}",
        f"_cell_length_b {lengths_and_angles[1]:.8f}",
        f"_cell_length_c {lengths_and_angles[2]:.8f}",
        f"_cell_angle_alpha {lengths_and_angles[3]:.8f}",
        f"_cell_angle_beta {lengths_and_angles[4]:.8f}",
        f"_cell_angle_gamma {lengths_and_angles[5]:.8f}",
        "_space_group_crystal_system triclinic",
        "_space_group_name_H-M_alt 'P 1'",
        "_space_group_IT_number 1",
        "_space_group_magn.name_BNS 'P 1'",
        "_space_group_magn.number_BNS 1.1",
        "",
        "loop_",
        "_space_group_symop_magn_operation.id",
        "_space_group_symop_magn_operation.xyz",
        "1 x,y,z,+1",
        "",
        "loop_",
        "_space_group_symop_magn_centering.id",
        "_space_group_symop_magn_centering.xyz",
        "1 x,y,z,+1",
        "",
        "loop_",
        "_space_group_symop_operation_xyz",
        "x,y,z",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    cif_lines.extend(
        f"{label} {element} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}"
        for label, element, position in zip(labels, data.atom_elements, data.fractional_positions)
    )
    nonzero_moments = [
        (label, moment)
        for label, moment in zip(labels, crystal_moments)
        if bool(np.any(moment != 0.0))
    ]
    if nonzero_moments:
        cif_lines.extend(
            [
                "",
                "loop_",
                "_atom_site_moment_label",
                "_atom_site_moment_crystalaxis_x",
                "_atom_site_moment_crystalaxis_y",
                "_atom_site_moment_crystalaxis_z",
            ]
        )
        cif_lines.extend(
            f"{label} {moment[0]:.8f} {moment[1]:.8f} {moment[2]:.8f}"
            for label, moment in nonzero_moments
        )
    summary_lines = [
        "Lyra magnetic moment evidence",
        "units: mu_B (Bohr magnetons)",
        "scope: OUTCAR on-site projected moments; not a whole-cell magnetic moment.",
        f"SAXIS: {data.saxis[0]:.8f} {data.saxis[1]:.8f} {data.saxis[2]:.8f}",
        "collinear convention: scalar moments are represented as Cartesian (0, 0, m).",
        "",
        "label element OUTCAR_spinor_x OUTCAR_spinor_y OUTCAR_spinor_z Cartesian_x Cartesian_y Cartesian_z magnitude",
    ]
    summary_lines.extend(
        f"{label} {element} "
        f"{spinor[0]:.8f} {spinor[1]:.8f} {spinor[2]:.8f} "
        f"{cartesian[0]:.8f} {cartesian[1]:.8f} {cartesian[2]:.8f} "
        f"{np.linalg.norm(cartesian):.8f}"
        for label, element, spinor, cartesian in zip(
            labels, data.atom_elements, data.spinor_vectors, data.cartesian_vectors
        )
    )
    magnetic_cif = destination_path / "vesta_magnetic_moments.mcif"
    summary = destination_path / "magnetic_moment_evidence.txt"
    prepared: list[Path] = []
    try:
        prepared_cif = _prepare_temp_text(magnetic_cif, "\n".join(cif_lines) + "\n")
        prepared.append(prepared_cif)
        prepared_summary = _prepare_temp_text(summary, "\n".join(summary_lines) + "\n")
        prepared.append(prepared_summary)
        validate_magnetic_cif(prepared_cif, labels)
        _replace_output_pair(
            (prepared_cif, prepared_summary),
            (magnetic_cif, summary),
        )
    finally:
        for temporary_path in prepared:
            temporary_path.unlink(missing_ok=True)
    return magnetic_cif, summary

from pathlib import Path

import numpy as np


def write_poscar(
    folder: Path,
    elements=("X",),
    counts=(1,),
    *,
    scale=1.0,
    lattice=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    coordinate_mode="Direct",
    positions=None,
    selective_dynamics=False,
):
    if positions is None:
        positions = ((0, 0, 0),) * sum(counts)
    text = "\n".join([
        "Synthetic",
        " ".join(str(value) for value in (scale if isinstance(scale, tuple) else (scale,))),
        *(" ".join(str(value) for value in vector) for vector in lattice),
        " ".join(elements),
        " ".join(str(value) for value in counts),
        *( ["Selective dynamics"] if selective_dynamics else [] ),
        coordinate_mode,
        *(" ".join(str(value) for value in position) for position in positions),
    ])
    (folder / "POSCAR").write_text(text, encoding="utf-8")


def write_outcar(folder: Path, efermi=5.0, ispin=1, lsorbit=False, noncollinear=False):
    text = "\n".join([
        f" ISPIN  = {ispin}",
        f" LSORBIT = {'T' if lsorbit else 'F'}",
        f" LNONCOLLINEAR = {'T' if noncollinear else 'F'}",
        f" E-fermi : {efermi:.6f}     XC(G=0): 0.0",
    ])
    (folder / "OUTCAR").write_text(text, encoding="utf-8")


def write_magnetic_poscar(
    folder: Path,
    *,
    scale=1.0,
    coordinate_mode="Direct",
    elements=("Fe", "O"),
    counts=(1, 1),
    positions=((0.25, 0.0, 0.0), (0.0, 0.5, 0.0)),
    lattice=((2, 0, 0), (0, 3, 0), (0, 0, 4)),
):
    text = "\n".join(
        [
            "Synthetic magnetic structure",
            str(scale),
            *(" ".join(str(value) for value in vector) for vector in lattice),
            " ".join(elements),
            " ".join(str(value) for value in counts),
            coordinate_mode,
            *(" ".join(str(value) for value in position) for position in positions),
        ]
    )
    (folder / "POSCAR").write_text(text, encoding="utf-8")


def _magnetization_table(component, moments):
    lines = [
        f" magnetization ({component})",
        " # of ion       s       p       d       tot",
        " --------------------------------------------",
    ]
    for index, moment in enumerate(moments, start=1):
        lines.append(f" {index:4d}  0.000  0.000  0.000  {moment:.6f}")
    lines.extend([" --------------------------------------------", " tot     0.000  0.000  0.000  0.000"])
    return lines


def write_magnetic_outcar(
    folder: Path,
    *,
    scalar_blocks=(),
    vector_blocks=(),
    component_blocks=(),
    saxis=(0.0, 0.0, 1.0),
):
    lines = [
        " ISPIN  = 2",
        " LSORBIT = T" if vector_blocks or component_blocks else " LSORBIT = F",
        " LNONCOLLINEAR = T" if vector_blocks or component_blocks else " LNONCOLLINEAR = F",
        " SAXIS = " + " ".join(str(value) for value in saxis),
    ]
    for moments in scalar_blocks:
        lines.extend(_magnetization_table("x", moments))
    for block in vector_blocks:
        for component, moments in zip(("x", "y", "z"), block, strict=True):
            lines.extend(_magnetization_table(component, moments))
    for component, moments, complete in component_blocks:
        table = _magnetization_table(component, moments)
        lines.extend(table if complete else table[:-2])
    (folder / "OUTCAR").write_text("\n".join(lines), encoding="utf-8")


def write_truncated_magnetic_outcar(folder: Path, *, component="x", moments=(1.0,)):
    lines = _magnetization_table(component, moments)
    (folder / "OUTCAR").write_text("\n".join(lines[:-2]), encoding="utf-8")


DEFAULT_KPOINT_SEGMENTS = (
    (((0, 0, 1), "H"), ((0, 0, 0), "A")),
    (((1, 0, 1), "L"), ((1, 0, 0), "M")),
    (((0, 1, 1), "H"), ((0, 1, 0), "K")),
)


def write_kpoints(
    folder: Path,
    n_per_segment=3,
    *,
    coordinate_mode="Reciprocal",
    segments=DEFAULT_KPOINT_SEGMENTS,
):
    lines = ["Band structure", str(n_per_segment), "Line-mode", coordinate_mode]
    for segment_index, segment in enumerate(segments):
        for coordinates, label in segment:
            lines.append(
                "{} ! {}".format(
                    " ".join(str(value) for value in coordinates),
                    label,
                )
            )
        if segment_index + 1 < len(segments):
            lines.append("")
    (folder / "KPOINTS").write_text("\n".join(lines), encoding="utf-8")


def band_fixture_kpoints(nkpts: int) -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, 0.5),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.5),
        (0.0, 1.0, 0.0),
    ][:nkpts]


def band_fixture_energy(kpoint_index: int, band_index: int, spin_index: int = 0) -> float:
    return 4.0 + band_index + kpoint_index * 0.05 + spin_index * 0.2


def write_eigenval(
    folder: Path,
    ispin=1,
    nkpts=9,
    nbands=2,
    *,
    kpoints=None,
    band_ids=None,
    omitted_band_rows=(),
):
    kpoints = band_fixture_kpoints(nkpts) if kpoints is None else list(kpoints)
    if len(kpoints) != nkpts:
        raise ValueError("kpoints must match nkpts")
    band_ids = band_ids or {}
    omitted_band_rows = set(omitted_band_rows)
    lines = ["header"] * 5 + [f"8 {nkpts} {nbands}", ""]
    for index, point in enumerate(kpoints):
        lines.append("{} {} {} 0.0".format(*point))
        for band in range(nbands):
            if (index, band) in omitted_band_rows:
                continue
            up = band_fixture_energy(index, band)
            band_id = band_ids.get(index, tuple(range(1, nbands + 1)))[band]
            if ispin == 2:
                down = band_fixture_energy(index, band, spin_index=1)
                lines.append(f"{band_id} {up:.6f} {down:.6f} 1.0 1.0")
            else:
                lines.append(f"{band_id} {up:.6f} 1.0")
        lines.append("")
    (folder / "EIGENVAL").write_text("\n".join(lines), encoding="utf-8")


def write_band_fixture(folder: Path, ispin=1):
    write_poscar(folder)
    write_outcar(folder, ispin=ispin)
    write_kpoints(folder)
    write_eigenval(folder, ispin=ispin)


def write_fermi_fixture(folder: Path, mesh=(4, 4, 4), ispin=1):
    write_poscar(folder)
    write_outcar(folder, efermi=0.0, ispin=ispin)
    (folder / "KPOINTS").write_text(
        "\n".join(
            [
                "Uniform Fermi-surface mesh",
                "0",
                "Gamma",
                " ".join(str(value) for value in mesh),
                "0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    axes = [np.arange(size, dtype=float) / size for size in mesh]
    kpoints = [
        (x, y, z)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    ]
    lines = ["header"] * 5 + [f"8 {len(kpoints)} 2", ""]
    for point in kpoints:
        lines.append("{} {} {} 1.0".format(*point))
        crossing = sum(np.cos(2.0 * np.pi * value) for value in point)
        if ispin == 2:
            lines.append(f"1 {crossing:.8f} {crossing + 0.25:.8f} 1.0 1.0")
            lines.append("2 4.00000000 4.25000000 0.0 0.0")
        else:
            lines.append(f"1 {crossing:.8f} 1.0")
            lines.append("2 4.00000000 0.0")
        lines.append("")
    (folder / "EIGENVAL").write_text("\n".join(lines), encoding="utf-8")


def write_procar(
    folder: Path,
    ispin=1,
    nkpts=9,
    nbands=2,
    nions=1,
    *,
    kpoint_overrides: dict[int, tuple[float, float, float]] | None = None,
    energy_overrides: dict[tuple[int, int, int], float] | None = None,
):
    orbitals = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2")
    lines = [
        "PROCAR lm decomposed",
        f" # of k-points: {nkpts}         # of bands: {nbands}         # of ions: {nions}",
        "",
    ]
    kpoints = band_fixture_kpoints(nkpts)
    kpoint_overrides = kpoint_overrides or {}
    energy_overrides = energy_overrides or {}
    for spin_index in range(ispin):
        if ispin == 2:
            lines.extend([f" spin component {spin_index + 1}", ""])
        for kpoint_index in range(nkpts):
            point = kpoint_overrides.get(kpoint_index, kpoints[kpoint_index])
            lines.extend([
                f" k-point {kpoint_index + 1} : {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} weight = 0.111111",
                "",
            ])
            for band_index in range(nbands):
                energy = energy_overrides.get(
                    (spin_index, kpoint_index, band_index),
                    band_fixture_energy(kpoint_index, band_index, spin_index),
                )
                lines.extend([
                    f" band {band_index + 1} # energy {energy:.6f} # occ. 1.0",
                    "",
                    " ion " + " ".join(orbitals) + " tot",
                ])
                totals = [0.0] * len(orbitals)
                for ion_index in range(nions):
                    base = 0.01 * (spin_index + 1) * (band_index + 1) * (ion_index + 1)
                    values = [base * (orbital_index + 1) for orbital_index in range(len(orbitals))]
                    totals = [left + right for left, right in zip(totals, values)]
                    lines.append(
                        f" {ion_index + 1} "
                        + " ".join(f"{value:.6f}" for value in values)
                        + f" {sum(values):.6f}"
                    )
                lines.append(
                    " tot "
                    + " ".join(f"{value:.6f}" for value in totals)
                    + f" {sum(totals):.6f}"
                )
                lines.append("")
    (folder / "PROCAR").write_text("\n".join(lines), encoding="utf-8")


def write_doscar(folder: Path, ispin=1, projected=True, nedos=4, nions=1):
    orbitals = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2")
    lines = [
        f" {nions} 1 1 0",
        "header",
        "header",
        "header",
        "header",
        f" 10.0 -10.0 {nedos} 5.0 0.0",
    ]
    for energy_index in range(nedos):
        energy = 4.0 + energy_index
        if ispin == 2:
            lines.append(f" {energy:.3f} {1 + energy_index:.3f} {2 + energy_index:.3f} 0 0")
        else:
            lines.append(f" {energy:.3f} {1 + energy_index:.3f} 0")
    if projected:
        for ion_index in range(nions):
            lines.append(f" 10.0 -10.0 {nedos} 5.0 0.0")
            for energy_index in range(nedos):
                energy = 4.0 + energy_index
                values = [0.01 * (ion_index + 1) * (index + 1) for index in range(len(orbitals))]
                if ispin == 2:
                    columns = []
                    for value in values:
                        columns.extend((value, value * 2))
                else:
                    columns = values
                lines.append(
                    f" {energy:.3f} " + " ".join(f"{value:.6f}" for value in columns)
                )
    (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")


def write_dos_fixture(folder: Path, ispin=1, projected=True):
    write_poscar(folder)
    write_outcar(folder, efermi=5.0, ispin=ispin)
    write_doscar(folder, ispin=ispin, projected=projected)

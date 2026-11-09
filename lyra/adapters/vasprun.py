from __future__ import annotations

import gzip
from importlib import import_module
import math
from pathlib import Path
from xml.etree.ElementTree import XMLPullParser

import numpy as np

from lyra.adapters.common import (
    build_segmented_axis,
    normalise_orbital,
    orbital_group,
    parse_kpoints_segments,
)
from lyra.models import ParsedBandData, ParsedDosData, SpinKind


def _import_pymatgen():
    try:
        outputs = import_module("pymatgen.io.vasp.outputs")
        electronic = import_module("pymatgen.electronic_structure.core")
    except ImportError as exc:
        raise RuntimeError(
            "Full Mode requires pymatgen. Install Lyra with the 'full' optional dependencies."
        ) from exc
    return outputs.Vasprun, tuple(orbital.name for orbital in electronic.Orbital)


def _xml_path(folder: Path) -> Path:
    for name in ("vasprun.xml", "vasprun.xml.gz"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No vasprun.xml or vasprun.xml.gz was found in {folder}.")


def _read_xml_efermi(path: Path) -> float:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    parser = XMLPullParser(events=("start", "end"))
    inside_main_dos = False
    try:
        with opener(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                parser.feed(chunk)
                for event, element in parser.read_events():
                    tag = element.tag.rsplit("}", 1)[-1]
                    if event == "start" and tag == "dos":
                        inside_main_dos = element.attrib.get("comment") != "kpoints_opt"
                    elif event == "end":
                        if (
                            inside_main_dos
                            and tag == "i"
                            and element.attrib.get("name") == "efermi"
                        ):
                            try:
                                value = float(element.text)
                            except (TypeError, ValueError) as exc:
                                raise RuntimeError(
                                    "vasprun.xml contains an invalid E-fermi value."
                                ) from exc
                            if not math.isfinite(value):
                                raise RuntimeError(
                                    "vasprun.xml contains an invalid E-fermi value."
                                )
                            return value
                        if tag == "dos":
                            inside_main_dos = False
                        element.clear()
    except OSError as exc:
        raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
    raise RuntimeError(
        "Could not find E-fermi in vasprun.xml. The VASP output may be incomplete; "
        "check that the calculation finished and the XML contains a DOS section."
    )


def _element_name(site) -> str:
    specie = site.specie
    return str(getattr(specie, "symbol", specie))


def _ordered_elements(sites) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_element_name(site) for site in sites))


def _spin_names(spins) -> dict[object, str]:
    spins = tuple(spins)
    if len(spins) == 1:
        return {spins[0]: "total"}
    names = {}
    for index, spin in enumerate(spins):
        raw = str(getattr(spin, "name", spin)).lower()
        if "down" in raw or raw in {"-1", "-1.0"}:
            names[spin] = "down"
        elif "up" in raw or raw in {"1", "1.0"}:
            names[spin] = "up"
        else:
            names[spin] = "up" if index == 0 else "down"
    return names


def _spin_kind(vasprun, channel_count: int) -> SpinKind:
    parameters = getattr(vasprun, "parameters", {}) or {}
    if bool(parameters.get("LSORBIT")) or bool(parameters.get("LNONCOLLINEAR")):
        return SpinKind.SOC
    return SpinKind.COLLINEAR if channel_count == 2 else SpinKind.NON_SPIN


def _band_occupations(vasprun, spin_map, energies) -> dict[str, np.ndarray]:
    eigenvalues = getattr(vasprun, "eigenvalues", None)
    if not eigenvalues:
        return {}
    result = {}
    for spin, raw_values in eigenvalues.items():
        spin_name = spin_map.get(spin)
        if spin_name is None or spin_name not in energies:
            return {}
        values = np.asarray(raw_values, dtype=float)
        expected_shape = np.asarray(energies[spin_name]).shape
        if values.ndim != 3 or values.shape[:2] != expected_shape or values.shape[2] < 2:
            return {}
        occupations = np.asarray(values[:, :, 1], dtype=float)
        if not np.all(np.isfinite(occupations)):
            return {}
        result[spin_name] = occupations
    return result if set(result) == set(energies) else {}


def _accumulate_projection(target, key, spin, values):
    channel = target.setdefault(key, {})
    if spin not in channel:
        channel[spin] = np.zeros_like(values, dtype=float)
    channel[spin] += values


_PYMATGEN_ORBITAL_ENUM = (
    "s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2",
    "f_3", "f_2", "f_1", "f0", "f1", "f2", "f3",
)
_VALID_PROJECTED_ORBITAL_COUNTS = {1, 4, 9, 16}


def _projection_orbital_names(orbital_names, count: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in orbital_names)
    if count == len(names):
        return names
    if (
        names == _PYMATGEN_ORBITAL_ENUM
        and count in _VALID_PROJECTED_ORBITAL_COUNTS
    ):
        return names[:count]
    raise RuntimeError(
        "Full Mode projected eigenvalue tensor orbital dimension does not "
        f"match the orbital metadata (tensor={count}, metadata={len(names)})."
    )


def _band_projections(band_structure, orbital_names, spin_map):
    raw = getattr(band_structure, "projections", None) or {}
    if not raw:
        return {}, ()
    sites = band_structure.structure.sites
    result = {}
    seen_orbitals = []
    for spin, projected in raw.items():
        values = np.asarray(projected, dtype=float)
        if values.ndim != 4:
            raise RuntimeError(
                "Full Mode projected eigenvalue tensor must have four dimensions "
                "(band, k-point, orbital, ion)."
            )
        projected_orbitals = _projection_orbital_names(
            orbital_names, values.shape[2]
        )
        spin_name = spin_map.get(spin, "total")
        if values.shape[3] != len(sites):
            raise RuntimeError(
                "Full Mode projected eigenvalue tensor ion dimension does not "
                "match the structure."
            )
        for site_index, site in enumerate(sites):
            element = _element_name(site)
            for orbital_index, raw_orbital in enumerate(projected_orbitals):
                orbital = normalise_orbital(raw_orbital)
                seen_orbitals.append(orbital)
                weights = values[:, :, orbital_index, site_index].T
                _accumulate_projection(result, element, spin_name, weights)
                resolved_key = f"{element}:{orbital}"
                group_key = f"{element}:{orbital_group(orbital)}"
                _accumulate_projection(result, resolved_key, spin_name, weights)
                if group_key != resolved_key:
                    _accumulate_projection(result, group_key, spin_name, weights)
    return result, tuple(dict.fromkeys(seen_orbitals))


def _dos_projections(complete_dos, spin_map):
    pdos = getattr(complete_dos, "pdos", None) or {}
    result = {}
    seen_orbitals = []
    for site, orbital_map in pdos.items():
        element = _element_name(site)
        for orbital, orbital_dos in orbital_map.items():
            orbital_name = normalise_orbital(
                str(getattr(orbital, "name", orbital))
            )
            seen_orbitals.append(orbital_name)
            spin_densities = (
                orbital_dos
                if isinstance(orbital_dos, dict)
                else orbital_dos.densities
            )
            for spin, density in spin_densities.items():
                spin_name = spin_map.get(spin, "total")
                values = np.asarray(density, dtype=float)
                _accumulate_projection(result, element, spin_name, values)
                resolved_key = f"{element}:{orbital_name}"
                group_key = f"{element}:{orbital_group(orbital_name)}"
                _accumulate_projection(result, resolved_key, spin_name, values)
                if group_key != resolved_key:
                    _accumulate_projection(result, group_key, spin_name, values)
    return result, tuple(dict.fromkeys(seen_orbitals))


class VasprunAdapter:
    @staticmethod
    def load_band(folder: str | Path) -> ParsedBandData:
        folder = Path(folder)
        xml_path = _xml_path(folder)
        kpoints_path = folder / "KPOINTS"
        if not kpoints_path.is_file():
            raise FileNotFoundError(f"Full Band Mode requires KPOINTS in {folder}.")
        efermi = _read_xml_efermi(xml_path)
        Vasprun, orbital_names = _import_pymatgen()
        vasprun = Vasprun(
            xml_path,
            parse_dos=False,
            parse_eigen=True,
            parse_projected_eigen=True,
            parse_potcar_file=False,
        )
        band_structure = vasprun.get_band_structure(
            kpoints_filename=str(kpoints_path), efermi=efermi, line_mode=True
        )
        spin_map = _spin_names(band_structure.bands)
        energies = {
            spin_map[spin]: np.asarray(values, dtype=float).T
            for spin, values in band_structure.bands.items()
        }
        occupations = _band_occupations(vasprun, spin_map, energies)
        kpoints = np.asarray(
            [point.frac_coords for point in band_structure.kpoints], dtype=float
        )
        reciprocal_lattice = getattr(band_structure, "lattice_rec", None)
        reciprocal = np.asarray(
            getattr(reciprocal_lattice, "matrix", np.eye(3)), dtype=float
        )
        n_per_segment, segments = parse_kpoints_segments(kpoints_path)
        distances, ranges, ticks, labels = build_segmented_axis(
            kpoints, reciprocal, n_per_segment, segments
        )
        projections, orbitals = _band_projections(
            band_structure, orbital_names, spin_map
        )
        sites = band_structure.structure.sites
        elements = _ordered_elements(sites)
        projection_total = {
            spin_name: np.sum(
                np.stack(
                    [
                        np.asarray(projections[element][spin_name], dtype=float)
                        for element in elements
                    ],
                    axis=0,
                ),
                axis=0,
            )
            for spin_name in energies
        } if projections else {}
        return ParsedBandData(
            folder=folder,
            kpoints=kpoints,
            distances=distances,
            segment_ranges=ranges,
            tick_positions=ticks,
            tick_labels=labels,
            energies=energies,
            occupations=occupations,
            efermi=efermi,
            spin_kind=_spin_kind(vasprun, len(energies)),
            elements=elements,
            projections=projections,
            orbitals=orbitals,
            projection_total=projection_total,
        )

    @staticmethod
    def load_dos(folder: str | Path) -> ParsedDosData:
        folder = Path(folder)
        xml_path = _xml_path(folder)
        Vasprun, _ = _import_pymatgen()
        vasprun = Vasprun(
            xml_path,
            parse_dos=True,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )
        complete_dos = vasprun.complete_dos
        if complete_dos is None:
            raise RuntimeError(
                "The selected vasprun file does not contain DOS data. Run VASP with DOS output enabled."
            )
        spin_map = _spin_names(complete_dos.densities)
        densities = {
            spin_map[spin]: np.asarray(values, dtype=float)
            for spin, values in complete_dos.densities.items()
        }
        projections, orbitals = _dos_projections(complete_dos, spin_map)
        sites = complete_dos.structure.sites
        return ParsedDosData(
            folder=folder,
            energies=np.asarray(complete_dos.energies, dtype=float),
            densities=densities,
            efermi=float(complete_dos.efermi),
            spin_kind=_spin_kind(vasprun, len(densities)),
            elements=_ordered_elements(sites),
            projections=projections,
            orbitals=orbitals,
        )

from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path

from lyra.models import CapabilityReport, ReadMode


BAND_LITE_REQUIRED = ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR")
DOS_LITE_REQUIRED = ("DOSCAR", "POSCAR", "OUTCAR")
FERMI_LITE_REQUIRED = ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR")
BAND_RECOGNIZED = (*BAND_LITE_REQUIRED, "PROCAR", "vasprun.xml", "vasprun.xml.gz")
DOS_RECOGNIZED = (*DOS_LITE_REQUIRED, "vasprun.xml", "vasprun.xml.gz")
FERMI_RECOGNIZED = (*FERMI_LITE_REQUIRED, "PROCAR", "vasprun.xml", "vasprun.xml.gz")


def _present(folder: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if (folder / name).is_file())


def _xml_file(folder: Path) -> str | None:
    for name in ("vasprun.xml", "vasprun.xml.gz"):
        if (folder / name).is_file():
            return name
    return None


def _xml_has_section(path: Path, section: str) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return _xml_has_section_cached(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns, section
    )


@lru_cache(maxsize=64)
def _xml_has_section_cached(
    path_text: str, _size: int, _mtime_ns: int, section: str
) -> bool:
    path = Path(path_text)
    marker = f"<{section}".encode("ascii")
    opener = gzip.open if path.name.endswith(".gz") else open
    carry = b""
    try:
        with opener(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                data = (carry + chunk).lower()
                if marker in data:
                    return True
                carry = data[-len(marker) :]
    except (OSError, EOFError):
        return False
    return False


def _doscar_has_projected_blocks(path: Path) -> bool:
    try:
        with path.open("r", errors="ignore") as handle:
            header = [next(handle) for _ in range(6)]
            nedos = int(float(header[5].split()[2]))
            for _ in range(nedos):
                next(handle)
            projected_header = next(handle, "")
    except (OSError, StopIteration, ValueError, IndexError):
        return False
    return len(projected_header.split()) >= 4


def _kpoints_is_line_mode(path: Path) -> bool:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return False
    return len(lines) >= 3 and "line" in lines[2].lower()


def inspect_fermi_folder(path: str | Path, mode: ReadMode) -> CapabilityReport:
    folder = Path(path)
    detected = _present(folder, FERMI_RECOGNIZED)
    if mode is ReadMode.FULL:
        xml_name = _xml_file(folder)
        available = xml_name is not None
        missing = () if available else ("vasprun.xml or vasprun.xml.gz",)
        reasons = (
            {}
            if available
            else {"fermi_surface": "Missing vasprun.xml or vasprun.xml.gz"}
        )
        return CapabilityReport(
            folder=folder,
            mode=mode,
            detected_files=detected,
            used_files=(xml_name,) if xml_name else (),
            ignored_files=(),
            missing_files=missing,
            features={"fermi_surface": available, "fermi_slice": available},
            disabled_reasons=reasons,
        )

    missing = tuple(name for name in FERMI_LITE_REQUIRED if name not in detected)
    line_mode = not missing and _kpoints_is_line_mode(folder / "KPOINTS")
    available = not missing and not line_mode
    reasons = {}
    if missing:
        reasons["fermi_surface"] = "Missing " + ", ".join(missing)
    elif line_mode:
        reasons["fermi_surface"] = (
            "Line-mode KPOINTS is a path calculation, not a uniform Fermi-surface mesh."
        )
    if not available:
        reasons["fermi_slice"] = reasons["fermi_surface"]
    ignored = tuple(
        name for name in ("PROCAR", "vasprun.xml", "vasprun.xml.gz")
        if name in detected
    )
    return CapabilityReport(
        folder=folder,
        mode=mode,
        detected_files=detected,
        used_files=tuple(
            name for name in FERMI_LITE_REQUIRED if name in detected
        ),
        ignored_files=ignored,
        missing_files=missing,
        features={"fermi_surface": available, "fermi_slice": available},
        disabled_reasons=reasons,
    )


def inspect_band_folder(
    path: str | Path,
    mode: ReadMode,
    *,
    scan_projection_sections: bool = True,
) -> CapabilityReport:
    folder = Path(path)
    detected = _present(folder, BAND_RECOGNIZED)

    if mode is ReadMode.FULL:
        xml_name = _xml_file(folder)
        missing = tuple(name for name in ("KPOINTS",) if name not in detected)
        if xml_name is None:
            missing += ("vasprun.xml or vasprun.xml.gz",)
        standard = not missing
        used = (("KPOINTS", xml_name) if standard else ())
        projected = bool(
            standard
            and xml_name
            and scan_projection_sections
            and _xml_has_section(folder / xml_name, "projected")
        )
        features = {"standard_band": standard, "projected_band": projected}
        reasons = {}
        if not standard:
            reasons["standard_band"] = "Missing " + ", ".join(missing)
            reasons["projected_band"] = reasons["standard_band"]
        elif not scan_projection_sections:
            reasons["projected_band"] = "Projection data is being inspected."
        elif not projected:
            reasons["projected_band"] = (
                "vasprun.xml has no projected eigenvalue section; projected Band is disabled."
            )
        return CapabilityReport(
            folder=folder,
            mode=mode,
            detected_files=detected,
            used_files=tuple(name for name in used if name),
            ignored_files=(),
            missing_files=missing,
            features=features,
            disabled_reasons=reasons,
        )

    missing = tuple(name for name in BAND_LITE_REQUIRED if name not in detected)
    standard = not missing
    projected = standard and "PROCAR" in detected
    ignored = tuple(name for name in ("vasprun.xml", "vasprun.xml.gz") if name in detected)
    reasons = {}
    if not standard:
        reasons["standard_band"] = "Missing " + ", ".join(missing)
        reasons["projected_band"] = reasons["standard_band"]
    elif not projected:
        reasons["projected_band"] = "PROCAR was not detected; projected Band is disabled."
    return CapabilityReport(
        folder=folder,
        mode=mode,
        detected_files=detected,
        used_files=tuple(name for name in (*BAND_LITE_REQUIRED, "PROCAR") if name in detected),
        ignored_files=ignored,
        missing_files=missing,
        features={"standard_band": standard, "projected_band": projected},
        disabled_reasons=reasons,
    )


def inspect_dos_folder(
    path: str | Path,
    mode: ReadMode,
    *,
    scan_projection_sections: bool = True,
) -> CapabilityReport:
    folder = Path(path)
    detected = _present(folder, DOS_RECOGNIZED)

    if mode is ReadMode.FULL:
        xml_name = _xml_file(folder)
        available = xml_name is not None
        projected = bool(
            xml_name
            and scan_projection_sections
            and _xml_has_section(folder / xml_name, "partial")
        )
        missing = () if available else ("vasprun.xml or vasprun.xml.gz",)
        reason = {}
        if not available:
            reason["total_dos"] = "Missing vasprun.xml or vasprun.xml.gz"
            reason["projected_dos"] = reason["total_dos"]
        elif not scan_projection_sections:
            reason["projected_dos"] = "Projection data is being inspected."
        elif not projected:
            reason["projected_dos"] = (
                "vasprun.xml has no partial DOS section; projected DOS is disabled."
            )
        return CapabilityReport(
            folder=folder,
            mode=mode,
            detected_files=detected,
            used_files=(xml_name,) if xml_name else (),
            ignored_files=(),
            missing_files=missing,
            features={"total_dos": available, "projected_dos": projected},
            disabled_reasons=reason,
        )

    missing = tuple(name for name in DOS_LITE_REQUIRED if name not in detected)
    total = not missing
    projected = total and _doscar_has_projected_blocks(folder / "DOSCAR")
    ignored = tuple(name for name in ("vasprun.xml", "vasprun.xml.gz") if name in detected)
    reasons = {}
    if not total:
        reasons["total_dos"] = "Missing " + ", ".join(missing)
        reasons["projected_dos"] = reasons["total_dos"]
    elif not projected:
        reasons["projected_dos"] = "DOSCAR has no projected site/orbital data."
    return CapabilityReport(
        folder=folder,
        mode=mode,
        detected_files=detected,
        used_files=tuple(name for name in DOS_LITE_REQUIRED if name in detected),
        ignored_files=ignored,
        missing_files=missing,
        features={"total_dos": total, "projected_dos": projected},
        disabled_reasons=reasons,
    )

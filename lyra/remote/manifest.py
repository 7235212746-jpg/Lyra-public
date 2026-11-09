from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

from .transport import RemoteStat, validate_remote_path


@dataclass(frozen=True)
class CapabilityCheck:
    name: str
    available: bool
    required_files: tuple[str, ...]
    missing_files: tuple[str, ...]


@dataclass(frozen=True)
class CalculationManifest:
    directory: str
    files: tuple[RemoteStat, ...]
    capabilities: tuple[CapabilityCheck, ...]
    fingerprint: str

    def capability(self, name: str) -> CapabilityCheck:
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        raise KeyError(name)


_CAPABILITY_REQUIREMENTS = (
    ("band", ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR"), None),
    ("projected_band", ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR"), None),
    ("dos", ("DOSCAR", "POSCAR", "OUTCAR"), None),
    ("magnetic_export", ("POSCAR", "OUTCAR"), None),
    ("fermi_surface", ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR"), None),
    ("wavecar_evidence", ("WAVECAR", "POSCAR", "OUTCAR"), None),
    ("charge_density", ("CHGCAR", "POSCAR"), None),
    ("bader", ("CHGCAR", "AECCAR0", "AECCAR2"), None),
    ("lobster", ("COHPCAR.lobster", "ICOHPLIST.lobster", "POSCAR"), None),
    ("full_mode", (), ("vasprun.xml", "vasprun.xml.gz")),
)


def build_calculation_manifest(
    directory: str,
    entries: Iterable[RemoteStat],
    *,
    file_contents: Mapping[str, str] | None = None,
) -> CalculationManifest:
    normalized_directory = validate_remote_path(directory)
    files = tuple(sorted((entry for entry in entries if not entry.is_directory), key=lambda item: item.name))
    if len({entry.name for entry in files}) != len(files):
        raise ValueError("Remote calculation listing contains duplicate file names.")
    names = {entry.name for entry in files if entry.size > 0}
    contents = file_contents or {}
    capabilities = []
    for name, all_required, any_required in _CAPABILITY_REQUIREMENTS:
        missing = tuple(item for item in all_required if item not in names)
        if any_required is not None and not any(item in names for item in any_required):
            missing = tuple(any_required)
        if name == "fermi_surface" and not _is_regular_kpoint_mesh(
            contents.get("KPOINTS", "")
        ):
            missing = (*missing, "KPOINTS（需规则三维网格，不能为 Line-mode）")
        capabilities.append(
            CapabilityCheck(name, not missing, tuple(all_required or any_required or ()), missing)
        )
    identity = {
        "schema_version": 1,
        "directory": normalized_directory,
        "files": [
            {"name": entry.name, "size": entry.size, "modified_ns": entry.modified_ns}
            for entry in files
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CalculationManifest(
        normalized_directory,
        files,
        tuple(capabilities),
        fingerprint,
    )


def _is_regular_kpoint_mesh(text: str) -> bool:
    lines = [line.split("!", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 4:
        return False
    try:
        count = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return False
    if count != 0:
        return False
    mode = lines[2].casefold()
    return mode.startswith("g") or mode.startswith("m") or mode.startswith("a")

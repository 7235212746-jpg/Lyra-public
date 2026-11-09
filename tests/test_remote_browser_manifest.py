from __future__ import annotations

import pytest

from lyra.remote.browser import RemoteBrowser
from lyra.remote.manifest import build_calculation_manifest
from lyra.remote.transport import RemoteStat
from tests.test_remote_support import FakeSession


def test_browser_sorts_directories_first_and_rejects_unsafe_paths():
    session = FakeSession()
    session.directories["/work"] = (
        RemoteStat("OUTCAR", "/work/OUTCAR", False, 12, 2),
        RemoteStat("calc-b", "/work/calc-b", True, 0, 3),
        RemoteStat("calc-A", "/work/calc-A", True, 0, 1),
    )

    entries = RemoteBrowser(session).list_directory("/work")

    assert [entry.name for entry in entries] == ["calc-A", "calc-b", "OUTCAR"]
    with pytest.raises(ValueError, match="absolute"):
        RemoteBrowser(session).list_directory("relative/path")
    with pytest.raises(ValueError, match="control"):
        RemoteBrowser(session).list_directory("/work\nattack")


def test_manifest_reports_vasp_file_capabilities_and_stable_fingerprint():
    entries = tuple(
        RemoteStat(name, f"/calc/{name}", False, size, mtime)
        for name, size, mtime in (
            ("EIGENVAL", 100, 1),
            ("KPOINTS", 20, 1),
            ("POSCAR", 30, 1),
            ("OUTCAR", 200, 2),
            ("PROCAR", 300, 3),
            ("WAVECAR", 400, 4),
        )
    )

    manifest = build_calculation_manifest(
        "/calc",
        entries,
        file_contents={"KPOINTS": "mesh\n0\nGamma\n8 8 8\n0 0 0\n"},
    )

    assert manifest.capability("band").available
    assert manifest.capability("projected_band").available
    assert manifest.capability("fermi_surface").available
    assert not manifest.capability("dos").available
    assert manifest.capability("dos").missing_files == ("DOSCAR",)
    assert manifest.capability("wavecar_evidence").available
    assert manifest.fingerprint == build_calculation_manifest("/calc", reversed(entries)).fingerprint


def test_full_mode_capability_accepts_compressed_vasprun():
    manifest = build_calculation_manifest(
        "/calc",
        (RemoteStat("vasprun.xml.gz", "/calc/vasprun.xml.gz", False, 10, 1),),
    )
    assert manifest.capability("full_mode").available


def test_manifest_rejects_empty_required_files_and_line_mode_fermi_surface():
    entries = tuple(
        RemoteStat(name, f"/calc/{name}", False, size, 1)
        for name, size in (
            ("EIGENVAL", 100),
            ("KPOINTS", 50),
            ("POSCAR", 0),
            ("OUTCAR", 100),
        )
    )
    manifest = build_calculation_manifest(
        "/calc",
        entries,
        file_contents={"KPOINTS": "path\n20\nLine-mode\nReciprocal\n"},
    )

    assert not manifest.capability("band").available
    assert "POSCAR" in manifest.capability("band").missing_files
    assert not manifest.capability("fermi_surface").available
    assert any("规则三维网格" in item for item in manifest.capability("fermi_surface").missing_files)

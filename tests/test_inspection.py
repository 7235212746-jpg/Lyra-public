import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lyra.adapters.inspection import (
    inspect_band_folder,
    inspect_dos_folder,
    inspect_fermi_folder,
)
from lyra.models import ReadMode


def touch_all(folder, names):
    for name in names:
        (folder / name).write_text("fixture", encoding="utf-8")


def write_doscar(folder, projected):
    lines = [
        "1 generated",
        "header",
        "header",
        "header",
        "header",
        "10 -10 2 5.5 0",
        "-1 1 0",
        "1 2 0",
    ]
    if projected:
        lines.extend([
            "10 -10 2 5.5 0",
            "-1 0.1 0.2 0.3",
            "1 0.2 0.3 0.4",
        ])
    (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")


class InspectionTests(unittest.TestCase):
    def test_lite_fermi_requires_uniform_non_line_mode_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["EIGENVAL", "POSCAR", "OUTCAR"])
            (folder / "KPOINTS").write_text(
                "mesh\n0\nGamma\n4 4 4\n0 0 0",
                encoding="utf-8",
            )

            report = inspect_fermi_folder(folder, ReadMode.LITE)

        self.assertTrue(report.is_available("fermi_surface"))
        self.assertEqual(
            set(report.used_files),
            {"EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR"},
        )

    def test_lite_fermi_disables_line_mode_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["EIGENVAL", "POSCAR", "OUTCAR"])
            (folder / "KPOINTS").write_text(
                "band\n20\nLine-mode\nReciprocal\n0 0 0\n0.5 0 0",
                encoding="utf-8",
            )

            report = inspect_fermi_folder(folder, ReadMode.LITE)

        self.assertFalse(report.is_available("fermi_surface"))
        self.assertIn("Line-mode", report.reason("fermi_surface"))

    def test_full_fermi_uses_explicit_vasprun_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["vasprun.xml.gz"])

            report = inspect_fermi_folder(folder, ReadMode.FULL)

        self.assertTrue(report.is_available("fermi_surface"))
        self.assertEqual(report.used_files, ("vasprun.xml.gz",))

    def test_lite_band_ignores_vasprun_and_downgrades_without_procar(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "vasprun.xml"])

            report = inspect_band_folder(folder, ReadMode.LITE)

        self.assertTrue(report.is_available("standard_band"))
        self.assertFalse(report.is_available("projected_band"))
        self.assertIn("PROCAR", report.reason("projected_band"))
        self.assertIn("vasprun.xml", report.detected_files)
        self.assertIn("vasprun.xml", report.ignored_files)
        self.assertEqual(report.mode, ReadMode.LITE)

    def test_lite_band_enables_projection_when_procar_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR"])

            report = inspect_band_folder(folder, ReadMode.LITE)

        self.assertTrue(report.is_available("standard_band"))
        self.assertTrue(report.is_available("projected_band"))

    def test_missing_lite_band_base_file_disables_only_band_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["EIGENVAL", "KPOINTS", "POSCAR", "PROCAR"])

            report = inspect_band_folder(folder, ReadMode.LITE)

        self.assertFalse(report.is_available("standard_band"))
        self.assertFalse(report.is_available("projected_band"))
        self.assertIn("OUTCAR", report.missing_files)

    def test_full_band_requires_explicit_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["vasprun.xml", "KPOINTS"])

            lite = inspect_band_folder(folder, ReadMode.LITE)
            full = inspect_band_folder(folder, ReadMode.FULL)

        self.assertFalse(lite.is_available("standard_band"))
        self.assertTrue(full.is_available("standard_band"))
        self.assertFalse(full.is_available("projected_band"))
        self.assertEqual(full.used_files, ("KPOINTS", "vasprun.xml"))

    def test_full_band_projection_requires_projected_xml_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "KPOINTS").write_text("fixture", encoding="utf-8")
            (folder / "vasprun.xml").write_text(
                "<modeling><projected><array /></projected></modeling>",
                encoding="utf-8",
            )

            report = inspect_band_folder(folder, ReadMode.FULL)

        self.assertTrue(report.is_available("projected_band"))

    def test_full_projection_scan_can_be_deferred_without_reading_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["vasprun.xml", "KPOINTS"])

            with patch("lyra.adapters.inspection._xml_has_section") as scan:
                report = inspect_band_folder(
                    folder,
                    ReadMode.FULL,
                    scan_projection_sections=False,
                )

        scan.assert_not_called()
        self.assertTrue(report.is_available("standard_band"))
        self.assertFalse(report.is_available("projected_band"))
        self.assertIn("inspected", report.reason("projected_band").lower())

    def test_lite_dos_total_only_disables_projection_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["POSCAR", "OUTCAR"])
            write_doscar(folder, projected=False)

            report = inspect_dos_folder(folder, ReadMode.LITE)

        self.assertTrue(report.is_available("total_dos"))
        self.assertFalse(report.is_available("projected_dos"))
        self.assertIn("projected", report.reason("projected_dos").lower())

    def test_lite_dos_enables_projection_when_doscar_has_site_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["POSCAR", "OUTCAR"])
            write_doscar(folder, projected=True)

            report = inspect_dos_folder(folder, ReadMode.LITE)

        self.assertTrue(report.is_available("total_dos"))
        self.assertTrue(report.is_available("projected_dos"))

    def test_full_dos_uses_only_explicit_vasprun_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            touch_all(folder, ["vasprun.xml.gz"])

            lite = inspect_dos_folder(folder, ReadMode.LITE)
            full = inspect_dos_folder(folder, ReadMode.FULL)

        self.assertFalse(lite.is_available("total_dos"))
        self.assertTrue(full.is_available("total_dos"))
        self.assertFalse(full.is_available("projected_dos"))
        self.assertEqual(full.used_files, ("vasprun.xml.gz",))

    def test_full_dos_projection_detects_partial_section_in_gzip_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with gzip.open(folder / "vasprun.xml.gz", "wt", encoding="utf-8") as handle:
                handle.write("<modeling><partial><array /></partial></modeling>")

            report = inspect_dos_folder(folder, ReadMode.FULL)

        self.assertTrue(report.is_available("total_dos"))
        self.assertTrue(report.is_available("projected_dos"))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lyra.adapters.common import read_poscar_structure
from lyra.models import (
    ParsedBandData,
    ParsedDosData,
    ParsedFermiData,
    ParsedStructure,
    SpinKind,
)
from lyra.quality import (
    assess_band,
    assess_combined,
    assess_dos,
    assess_fermi,
    scan_vasp_evidence,
)
from tests.fixtures import write_band_fixture, write_dos_fixture, write_poscar


class QualityEvidenceTests(unittest.TestCase):
    @staticmethod
    def _combined_data(
        band_structure: ParsedStructure,
        dos_structure: ParsedStructure,
        *,
        band_folder=Path("band"),
        dos_folder=Path("dos"),
    ):
        band = ParsedBandData(
            folder=band_folder,
            kpoints=np.zeros((2, 3)),
            distances=np.array([0.0, 1.0]),
            segment_ranges=((0, 2),),
            tick_positions=(0.0, 1.0),
            tick_labels=("G", "X"),
            energies={"total": np.zeros((2, 1))},
            efermi=5.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("Fe", "O"),
            structure=band_structure,
        )
        dos = ParsedDosData(
            folder=dos_folder,
            energies=np.array([-1.0, 1.0]),
            densities={"total": np.ones(2)},
            efermi=5.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("Fe", "O"),
            structure=dos_structure,
        )
        return band, dos

    def test_fatal_marker_and_missing_final_timing_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "OUTCAR").write_text(
                "NELM = 60\nZBRENT: fatal error in bracketing\n",
                encoding="utf-8",
            )

            evidence = scan_vasp_evidence(folder)

        self.assertTrue(evidence.fatal_markers)
        self.assertFalse(evidence.has_final_timing)

    def test_electronic_iteration_at_nelm_is_not_called_converged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "INCAR").write_text("NELM = 4\n", encoding="utf-8")
            (folder / "OSZICAR").write_text(
                "DAV: 1 -1.0\nDAV: 2 -1.1\nDAV: 3 -1.2\nDAV: 4 -1.3\n",
                encoding="utf-8",
            )

            report = assess_band(None, folder)

        self.assertIn("electronic-step-limit", {issue.code for issue in report.issues})

    def test_relaxation_dos_and_md_eigenval_receive_specific_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "INCAR").write_text("NSW = 20\nIBRION = 2\n", encoding="utf-8")
            dos_report = assess_dos(None, folder)
            (folder / "INCAR").write_text("NSW = 20\nIBRION = 0\n", encoding="utf-8")
            band_report = assess_band(None, folder)

        self.assertIn("relaxation-dos", {issue.code for issue in dos_report.issues})
        self.assertIn("md-eigenval", {issue.code for issue in band_report.issues})

    def test_combined_checks_composition_spin_and_fermi_reference(self):
        band = ParsedBandData(
            folder=Path("band"),
            kpoints=np.zeros((2, 3)),
            distances=np.array([0.0, 1.0]),
            segment_ranges=((0, 2),),
            tick_positions=(0.0, 1.0),
            tick_labels=("G", "X"),
            energies={"up": np.zeros((2, 1)), "down": np.zeros((2, 1))},
            efermi=5.4,
            spin_kind=SpinKind.COLLINEAR,
            elements=("Ta", "Ga"),
        )
        dos = ParsedDosData(
            folder=Path("dos"),
            energies=np.array([-1.0, 1.0]),
            densities={"total": np.ones(2)},
            efermi=5.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("Ta",),
        )

        report = assess_combined(band, dos)
        codes = {issue.code for issue in report.issues}

        self.assertIn("composition-mismatch", codes)
        self.assertIn("spin-kind-mismatch", codes)
        self.assertIn("fermi-reference-difference", codes)
        self.assertIn("DOS/SCF", report.method)

    def test_combined_uses_poscar_counts_not_only_unique_element_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_poscar(band_folder, elements=("Ta", "Ga"), counts=(1, 2))
            write_poscar(dos_folder, elements=("Ta", "Ga"), counts=(1, 3))
            band = ParsedBandData(
                folder=band_folder,
                kpoints=np.zeros((2, 3)),
                distances=np.array([0.0, 1.0]),
                segment_ranges=((0, 2),),
                tick_positions=(0.0, 1.0),
                tick_labels=("G", "X"),
                energies={"total": np.zeros((2, 1))},
                efermi=5.0,
                spin_kind=SpinKind.NON_SPIN,
                elements=("Ta", "Ga"),
            )
            dos = ParsedDosData(
                folder=dos_folder,
                energies=np.array([-1.0, 1.0]),
                densities={"total": np.ones(2)},
                efermi=5.0,
                spin_kind=SpinKind.NON_SPIN,
                elements=("Ta", "Ga"),
            )

            report = assess_combined(band, dos)

        self.assertIn(
            "composition-mismatch",
            {issue.code for issue in report.issues},
        )

    def test_combined_blocks_moved_internal_coordinate(self):
        band_structure = ParsedStructure(
            lattice=np.diag((2.0, 3.0, 4.0)),
            atom_elements=("Fe", "O"),
            fractional_positions=np.array(((0.0, 0.0, 0.0), (0.25, 0.5, 0.75))),
            source="band/POSCAR",
        )
        dos_structure = ParsedStructure(
            lattice=np.diag((2.0, 3.0, 4.0)),
            atom_elements=("Fe", "O"),
            fractional_positions=np.array(((0.0, 0.0, 0.0), (0.25, 0.6, 0.75))),
            source="dos/POSCAR",
        )
        band, dos = self._combined_data(band_structure, dos_structure)

        report = assess_combined(band, dos)

        self.assertIn(
            "structure-position-mismatch",
            {issue.code for issue in report.issues},
        )
        self.assertEqual(report.verdict, "存在阻断问题")

    def test_combined_blocks_swapped_unlike_atom_order(self):
        positions = np.array(((0.0, 0.0, 0.0), (0.25, 0.5, 0.75)))
        band_structure = ParsedStructure(
            lattice=np.eye(3),
            atom_elements=("Fe", "O"),
            fractional_positions=positions,
            source="band/POSCAR",
        )
        dos_structure = ParsedStructure(
            lattice=np.eye(3),
            atom_elements=("O", "Fe"),
            fractional_positions=positions[::-1],
            source="dos/POSCAR",
        )
        band, dos = self._combined_data(band_structure, dos_structure)

        report = assess_combined(band, dos)

        self.assertIn("composition-mismatch", {issue.code for issue in report.issues})
        self.assertEqual(report.verdict, "存在阻断问题")

    def test_combined_accepts_periodic_wrap_and_direct_cartesian_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            lattice = ((2, 0, 0), (0, 3, 0), (0, 0, 4))
            write_poscar(
                band_folder,
                elements=("Fe", "O"),
                counts=(1, 1),
                lattice=lattice,
                positions=((1.0, 0.0, 0.0), (0.25, 0.5, 0.75)),
            )
            write_poscar(
                dos_folder,
                elements=("Fe", "O"),
                counts=(1, 1),
                lattice=lattice,
                coordinate_mode="Cartesian",
                positions=((0.0, 0.0, 0.0), (0.5, 1.5, 3.0)),
            )
            band, dos = self._combined_data(
                read_poscar_structure(band_folder / "POSCAR"),
                read_poscar_structure(dos_folder / "POSCAR"),
                band_folder=band_folder,
                dos_folder=dos_folder,
            )

            report = assess_combined(band, dos)

        self.assertNotIn(
            "structure-position-mismatch",
            {issue.code for issue in report.issues},
        )
        self.assertEqual(report.verdict, "未发现已知阻断问题")

    def test_fermi_report_records_grid_and_linear_interpolation(self):
        data = ParsedFermiData(
            folder=Path("fermi"),
            axes=(
                np.array([0.0, 0.5]),
                np.array([0.0, 0.5]),
                np.array([0.0, 0.5]),
            ),
            energies={"total": np.zeros((2, 2, 2, 1))},
            efermi=0.0,
            reciprocal_lattice=np.eye(3),
            spin_kind=SpinKind.NON_SPIN,
        )

        report = assess_fermi(data)

        self.assertIn("2 x 2 x 2", {metric.value for metric in report.metrics})
        self.assertIn("linear", report.method.lower())


if __name__ == "__main__":
    unittest.main()

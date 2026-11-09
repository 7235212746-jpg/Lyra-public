import tempfile
import unittest
from pathlib import Path

import numpy as np

from lyra.adapters.doscar import DoscarAdapter
from lyra.models import SpinKind
from tests.fixtures import write_dos_fixture, write_doscar, write_outcar, write_poscar


class DoscarAdapterTests(unittest.TestCase):
    def test_reads_total_dos_and_uses_outcar_fermi(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=False)

            data = DoscarAdapter.load(folder)

        np.testing.assert_allclose(data.energies, [4.0, 5.0, 6.0, 7.0])
        np.testing.assert_allclose(data.densities["total"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(data.efermi, 5.0)
        self.assertEqual(data.spin_kind, SpinKind.NON_SPIN)
        self.assertEqual(data.projections, {})

    def test_lite_dos_retains_poscar_structure_and_atom_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=False)
            write_poscar(
                folder,
                elements=("Fe", "O"),
                counts=(1, 2),
                lattice=((2, 0, 0), (0, 3, 0), (0, 0, 4)),
                positions=((0.25, 0, 0), (0, 0.5, 0), (1.25, -0.25, 2)),
            )
            write_doscar(folder, nions=3, projected=False)

            data = DoscarAdapter.load(folder)

        self.assertIsNotNone(data.structure)
        assert data.structure is not None
        np.testing.assert_allclose(data.structure.lattice, [[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        self.assertEqual(data.structure.atom_elements, ("Fe", "O", "O"))
        np.testing.assert_allclose(data.structure.fractional_positions, [[0.25, 0, 0], [0, 0.5, 0], [0.25, 0.75, 0]])

    def test_accepts_total_only_dos_with_trailing_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=False)
            with (folder / "DOSCAR").open("a", encoding="utf-8") as handle:
                handle.write("\n\n")

            data = DoscarAdapter.load(folder)

        self.assertEqual(data.projections, {})

    def test_aggregates_resolved_projected_dos_by_element_and_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=True)

            data = DoscarAdapter.load(folder)

        self.assertIn("X:p", data.projections)
        self.assertIn("X:px", data.projections)
        self.assertIn("X:d", data.projections)
        self.assertIn("X:dx2-y2", data.projections)
        self.assertAlmostEqual(data.projections["X:p"]["total"][0], 0.09)
        self.assertAlmostEqual(data.projections["X:d"]["total"][0], 0.35)

    def test_reads_two_spin_total_and_projected_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, ispin=2, projected=True)

            data = DoscarAdapter.load(folder)

        self.assertEqual(data.spin_kind, SpinKind.COLLINEAR)
        self.assertEqual(tuple(data.densities), ("up", "down"))
        np.testing.assert_allclose(data.densities["down"], [2, 3, 4, 5])
        self.assertAlmostEqual(
            data.projections["X:p"]["down"][0],
            2 * data.projections["X:p"]["up"][0],
        )

    def test_noncollinear_projected_dos_retains_only_orbital_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, ispin=2, lsorbit=True, noncollinear=True)
            lines = [
                " 1 1 1 0",
                "header",
                "header",
                "header",
                "header",
                " 10.0 -10.0 2 5.0 0.0",
                " 4.000 1.000 0.000",
                " 5.000 2.000 0.000",
                " 10.0 -10.0 2 5.0 0.0",
            ]
            for energy_index, energy in enumerate((4.0, 5.0), start=1):
                columns = []
                for orbital_index in range(1, 10):
                    charge = 0.01 * energy_index * orbital_index
                    columns.extend((charge, 10.0, -20.0, 30.0))
                lines.append(
                    f" {energy:.3f} "
                    + " ".join(f"{value:.6f}" for value in columns)
                )
            (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")

            data = DoscarAdapter.load(folder)

        self.assertEqual(data.spin_kind, SpinKind.SOC)
        self.assertEqual(tuple(data.projections["X"]), ("total",))
        np.testing.assert_allclose(data.projections["X"]["total"], [0.45, 0.90])
        np.testing.assert_allclose(data.projections["X:p"]["total"], [0.09, 0.18])
        np.testing.assert_allclose(data.projections["X:px"]["total"], [0.04, 0.08])

    def test_rejects_doscar_fermi_that_disagrees_with_outcar(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=False)
            lines = (folder / "DOSCAR").read_text(encoding="utf-8").splitlines()
            header = lines[5].split()
            header[3] = "5.001"
            lines[5] = " ".join(header)
            (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "E-fermi"):
                DoscarAdapter.load(folder)

    def test_rejects_started_but_truncated_projected_dos(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder, counts=(2,))
            write_outcar(folder)
            write_doscar(folder, nions=2, projected=True)
            lines = (folder / "DOSCAR").read_text(encoding="utf-8").splitlines()
            (folder / "DOSCAR").write_text(
                "\n".join(lines[:-5]), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "projected DOS blocks"):
                DoscarAdapter.load(folder)

    def test_rejects_extra_projected_dos_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=True)
            lines = (folder / "DOSCAR").read_text(encoding="utf-8").splitlines()
            extra_block = lines[10:]
            (folder / "DOSCAR").write_text(
                "\n".join([*lines, *extra_block]), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "projected DOS blocks"):
                DoscarAdapter.load(folder)

    def test_rejects_projected_dos_with_mismatched_energy_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=True)
            lines = (folder / "DOSCAR").read_text(encoding="utf-8").splitlines()
            lines[11] = lines[11].replace("4.000", "4.000001", 1)
            (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "energy axis"):
                DoscarAdapter.load(folder)


if __name__ == "__main__":
    unittest.main()

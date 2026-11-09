import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lyra.adapters.fermi import FermiAdapter, regular_grid_from_kpoints
from tests.fixtures import write_band_fixture, write_fermi_fixture


class FermiAdapterTests(unittest.TestCase):
    def test_rejects_complete_but_nonuniform_cartesian_product(self):
        axes = (
            np.array([0.0, 0.2, 0.7]),
            np.array([0.0, 0.5]),
            np.array([0.0, 0.5]),
        )
        kpoints = np.array(
            [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
        )
        energies = {"total": np.zeros((len(kpoints), 1))}

        with self.assertRaisesRegex(ValueError, "uniform"):
            regular_grid_from_kpoints(kpoints, energies)

    def test_full_mode_reads_uniform_grid_from_vasprun(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text("<modeling />", encoding="utf-8")
            axes = np.arange(3, dtype=float) / 3
            points = [
                np.array((x, y, z))
                for x in axes
                for y in axes
                for z in axes
            ]

            class Point:
                def __init__(self, coords):
                    self.frac_coords = coords

            class Site:
                specie = type("Specie", (), {"symbol": "X"})()

            band_structure = type(
                "BandStructure",
                (),
                {
                    "bands": {"only": np.array([[sum(point) for point in points]])},
                    "kpoints": [Point(point) for point in points],
                    "lattice_rec": type("Lattice", (), {"matrix": np.eye(3)})(),
                    "efermi": 0.0,
                    "structure": type("Structure", (), {"sites": [Site()]})(),
                },
            )()

            class FakeVasprun:
                parameters = {}
                efermi = 0.0

                def __init__(self, *_args, **_kwargs):
                    pass

                def get_band_structure(self, line_mode=False):
                    self.line_mode = line_mode
                    return band_structure

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(FakeVasprun, ()),
            ):
                data = FermiAdapter.load_full(folder)

        self.assertEqual(data.grid_shape, (3, 3, 3))
        self.assertEqual(data.energies["total"].shape, (3, 3, 3, 1))
        self.assertEqual(data.elements, ("X",))

    def test_lite_mode_reads_complete_uniform_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_fermi_fixture(folder, mesh=(3, 4, 5), ispin=2)

            data = FermiAdapter.load_lite(folder)

        self.assertEqual(data.grid_shape, (3, 4, 5))
        self.assertEqual(data.energies["up"].shape, (3, 4, 5, 2))
        self.assertEqual(data.energies["down"].shape, (3, 4, 5, 2))
        np.testing.assert_allclose(data.axes[0], (0.0, 1 / 3, 2 / 3))
        self.assertEqual(data.efermi, 0.0)

    def test_regular_grid_rejects_irreducible_or_incomplete_mesh(self):
        kpoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        )
        energies = {"total": np.zeros((3, 2))}

        with self.assertRaisesRegex(ValueError, "complete.*uniform|ISYM=-1"):
            regular_grid_from_kpoints(kpoints, energies)

    def test_lite_mode_rejects_line_mode_band_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)

            with self.assertRaisesRegex(ValueError, "Line-mode|uniform"):
                FermiAdapter.load_lite(folder)


if __name__ == "__main__":
    unittest.main()

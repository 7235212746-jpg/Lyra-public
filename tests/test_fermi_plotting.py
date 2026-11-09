import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lyra.adapters.fermi import FermiAdapter
from lyra.models import ExportSettings, FermiPlotSettings
from lyra.plotting.fermi import (
    _slice_cartesian_coordinates,
    _slice_plane,
    build_fermi_figure,
    extract_fermi_meshes,
    render_fermi,
)
from tests.fixtures import write_fermi_fixture


class FermiPlottingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        write_fermi_fixture(self.folder, mesh=(5, 5, 5))
        self.data = FermiAdapter.load_lite(self.folder)

    def tearDown(self):
        self.temporary.cleanup()

    def test_surface_extracts_only_bands_crossing_selected_energy(self):
        settings = FermiPlotSettings(energy_offset=0.0, view_mode="surface")

        meshes = extract_fermi_meshes(self.data, settings)

        self.assertGreater(len(meshes), 0)
        self.assertEqual({mesh.band_index for mesh in meshes}, {0})
        self.assertTrue(all(len(mesh.vertices) and len(mesh.faces) for mesh in meshes))

    def test_slice_boundary_wraps_periodically(self):
        volume = self.data.energies["total"][..., 0]

        negative_edge = _slice_plane(volume, self.data.axes[2], -0.5, 2)
        positive_edge = _slice_plane(volume, self.data.axes[2], 0.5, 2)

        self.assertTrue((negative_edge == positive_edge).all())

    def test_slice_coordinates_preserve_oblique_reciprocal_metric(self):
        reciprocal = np.array(
            [
                [2.0, 0.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        data = replace(self.data, reciprocal_lattice=reciprocal)

        x, y = _slice_cartesian_coordinates(data, slice_axis=2)

        f0 = np.append(data.axes[0], data.axes[0][0] + 1.0)
        f1 = np.append(data.axes[1], data.axes[1][0] + 1.0)
        f0 = f0 - data.axes[0][0] - 0.5
        f1 = f1 - data.axes[1][0] - 0.5
        expected_x = 2.0 * f0[:, None] + f1[None, :]
        expected_y = 3.0 * np.broadcast_to(f1[None, :], expected_x.shape)
        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)

    def test_builds_surface_and_slice_figures(self):
        surface = build_fermi_figure(
            self.data,
            FermiPlotSettings(view_mode="surface", energy_offset=0.0),
        )
        slice_figure = build_fermi_figure(
            self.data,
            FermiPlotSettings(
                view_mode="slice",
                energy_offset=0.0,
                slice_axis=2,
                slice_position=0.0,
            ),
        )
        try:
            self.assertEqual(surface.axes[0].name, "3d")
            self.assertNotEqual(slice_figure.axes[0].name, "3d")
        finally:
            surface.clear()
            slice_figure.clear()

    def test_exports_fermi_surface(self):
        settings = FermiPlotSettings(
            energy_offset=0.0,
            export=ExportSettings(
                basename="fermi",
                formats=("png", "pdf", "svg"),
                dpi=72,
                width_mm=120,
            ),
        )

        result = render_fermi(self.data, settings, self.folder / "output")

        self.assertEqual({path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"})
        self.assertTrue(all(path.is_file() for path in result.output_files))


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from lyra.adapters.doscar import DoscarAdapter
from lyra.models import DosPlotSettings, ExportSettings, PlotRange, ProjectionChannel
from lyra.plotting.dos import build_dos_figure, render_dos
from tests.fixtures import write_dos_fixture


class DosPlottingTests(unittest.TestCase):
    def test_spin_down_is_mirrored_and_energy_range_is_editable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, ispin=2, projected=True)
            data = DoscarAdapter.load(folder)
            settings = DosPlotSettings(energy_range=PlotRange(-1.0, 1.5))

            figure = build_dos_figure(data, settings)

        axes = figure.axes[0]
        total_down = next(line for line in axes.lines if line.get_label() == "Total down")
        self.assertTrue(all(value <= 0 for value in total_down.get_ydata()))
        self.assertEqual(tuple(round(value, 2) for value in axes.get_xlim()), (-1.0, 1.5))
        figure.clear()

    def test_density_range_is_editable_independently_of_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=True)
            data = DoscarAdapter.load(folder)
            settings = DosPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                density_range=PlotRange(0.0, 25.0),
            )

            figure = build_dos_figure(data, settings)

        self.assertEqual(tuple(round(value, 2) for value in figure.axes[0].get_ylim()), (0.0, 25.0))
        figure.clear()

    def test_automatic_density_range_ignores_peaks_outside_energy_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder)
            data = DoscarAdapter.load(folder)
            data = replace(
                data,
                energies=np.array((-10.0, -0.5, 0.5, 10.0)),
                efermi=0.0,
                densities={
                    "total": np.array((1000.0, 10.0, 20.0, 1000.0))
                },
            )

            figure = build_dos_figure(
                data,
                DosPlotSettings(energy_range=PlotRange(-1.0, 1.0)),
            )

        self.assertEqual(tuple(round(value, 2) for value in figure.axes[0].get_ylim()), (0.0, 21.0))
        figure.clear()

    def test_selected_element_orbital_is_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder, projected=True)
            data = DoscarAdapter.load(folder)
            settings = DosPlotSettings(
                energy_range=PlotRange(-2, 2),
                projection_channels=(ProjectionChannel("X", "d", "#2457d6"),),
            )

            figure = build_dos_figure(data, settings)

        self.assertIn("X-d", [line.get_label() for line in figure.axes[0].lines])
        figure.clear()

    def test_exports_png_pdf_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder)
            data = DoscarAdapter.load(folder)
            settings = DosPlotSettings(
                energy_range=PlotRange(-2, 2),
                export=ExportSettings(basename="paper_dos", dpi=120),
            )

            result = render_dos(data, settings, folder)

            self.assertEqual({path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"})
            self.assertTrue(all(path.stat().st_size > 0 for path in result.output_files))


if __name__ == "__main__":
    unittest.main()

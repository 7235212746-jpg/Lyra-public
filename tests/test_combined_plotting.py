from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection, PathCollection
import numpy as np

from lyra.adapters.doscar import DoscarAdapter
from lyra.adapters.eigenval import EigenvalAdapter
from lyra.adapters.procar import ProcarAdapter
from lyra.models import CombinedPlotSettings, ExportSettings, PlotRange, ProjectionChannel
from lyra.plotting.combined import build_combined_figure, render_combined
from tests.fixtures import write_band_fixture, write_dos_fixture, write_procar


class CombinedPlottingTests(unittest.TestCase):
    def test_complete_band_backbone_uses_combined_colour_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_procar(band_folder)
            write_dos_fixture(dos_folder)
            band_data = ProcarAdapter.attach(
                band_folder, EigenvalAdapter.load(band_folder)
            )

            figure = build_combined_figure(
                band_data,
                DoscarAdapter.load(dos_folder),
                CombinedPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    band_mode="rgb",
                    band_projection_channels=(
                        ProjectionChannel("X", "p", "#d94841"),
                    ),
                    band_complete_colour="#654321",
                ),
            )

        backbone = next(
            item for item in figure.axes[0].collections
            if getattr(item, "_lyra_layer", None) == "complete-band"
        )
        np.testing.assert_allclose(
            backbone.get_colors()[0][:3],
            [0.396078431372549, 0.2627450980392157, 0.12941176470588237],
        )
        figure.clear()
    def test_uses_dos_fermi_level_for_both_panels_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder)
            band_data = EigenvalAdapter.load(band_folder)
            dos_data = replace(DoscarAdapter.load(dos_folder), efermi=4.5)

            figure = build_combined_figure(
                band_data,
                dos_data,
                CombinedPlotSettings(energy_range=PlotRange(-2.0, 2.0)),
            )

        band_axes, dos_axes = figure.axes
        first_band = next(
            collection
            for collection in band_axes.collections
            if isinstance(collection, LineCollection)
        )
        self.assertAlmostEqual(float(first_band.get_segments()[0][0, 1]), -0.5)
        self.assertAlmostEqual(float(dos_axes.lines[0].get_ydata()[0]), -0.5)
        self.assertEqual(band_axes.get_shared_y_axes().joined(band_axes, dos_axes), True)
        figure.clear()

    def test_dos_is_horizontal_spin_down_is_mirrored_and_ratio_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder, ispin=2)
            band_data = EigenvalAdapter.load(band_folder)
            dos_data = DoscarAdapter.load(dos_folder)

            figure = build_combined_figure(
                band_data,
                dos_data,
                CombinedPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    panel_ratio=3.0,
                ),
            )

        band_axes, dos_axes = figure.axes
        total_down = next(line for line in dos_axes.lines if line.get_label() == "Total down")
        self.assertTrue(all(value <= 0 for value in total_down.get_xdata()))
        width_ratio = band_axes.get_position().width / dos_axes.get_position().width
        self.assertAlmostEqual(width_ratio, 3.0, delta=0.2)
        figure.clear()

    def test_horizontal_dos_density_range_is_editable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder)

            figure = build_combined_figure(
                EigenvalAdapter.load(band_folder),
                DoscarAdapter.load(dos_folder),
                CombinedPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    dos_density_range=PlotRange(0.0, 30.0),
                ),
            )

        self.assertEqual(tuple(round(value, 2) for value in figure.axes[1].get_xlim()), (0.0, 30.0))
        figure.clear()

    def test_horizontal_dos_auto_range_ignores_peaks_outside_energy_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder)
            dos_data = replace(
                DoscarAdapter.load(dos_folder),
                energies=np.array((-10.0, -0.5, 0.5, 10.0)),
                efermi=0.0,
                densities={
                    "total": np.array((1000.0, 10.0, 20.0, 1000.0))
                },
            )

            figure = build_combined_figure(
                EigenvalAdapter.load(band_folder),
                dos_data,
                CombinedPlotSettings(energy_range=PlotRange(-1.0, 1.0)),
            )

        self.assertEqual(tuple(round(value, 2) for value in figure.axes[1].get_xlim()), (0.0, 21.0))
        figure.clear()

    def test_dos_legend_occupies_a_separate_region_outside_the_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder, ispin=2, projected=True)
            channels = (
                ProjectionChannel("X", "s", "#d94841"),
                ProjectionChannel("X", "p", "#2457d6"),
                ProjectionChannel("X", "d", "#159a73"),
            )

            figure = build_combined_figure(
                EigenvalAdapter.load(band_folder),
                DoscarAdapter.load(dos_folder),
                CombinedPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    dos_projection_channels=channels,
                ),
            )

        figure.canvas.draw()
        dos_axes = figure.axes[1]
        legend = dos_axes.get_legend()
        renderer = figure.canvas.get_renderer()
        axes_box = dos_axes.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        self.assertGreaterEqual(legend_box.x0, axes_box.x1)
        self.assertLessEqual(legend_box.x1, figure.bbox.x1)
        figure.clear()

    def test_default_combined_labels_ticks_and_legend_are_readable_when_reduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder, ispin=2, projected=True)

            figure = build_combined_figure(
                EigenvalAdapter.load(band_folder),
                DoscarAdapter.load(dos_folder),
                CombinedPlotSettings(
                    energy_range=PlotRange(-2.0, 2.0),
                    dos_projection_channels=(
                        ProjectionChannel("X", "d", "#159a73"),
                    ),
                ),
            )

        band_axes, dos_axes = figure.axes
        legend = dos_axes.get_legend()
        sizes = (
            band_axes.xaxis.label.get_fontsize(),
            band_axes.yaxis.label.get_fontsize(),
            dos_axes.xaxis.label.get_fontsize(),
            *(label.get_fontsize() for label in band_axes.get_xticklabels()),
            *(label.get_fontsize() for label in legend.get_texts()),
        )
        self.assertTrue(all(size >= 13.0 for size in sizes), sizes)
        figure.clear()

    def test_projected_rgb_and_fat_modes_use_bounded_band_artists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_procar(band_folder)
            write_dos_fixture(dos_folder)
            band_data = ProcarAdapter.attach(band_folder, EigenvalAdapter.load(band_folder))
            dos_data = DoscarAdapter.load(dos_folder)
            channels = (
                ProjectionChannel("X", "p", "#d94841"),
                ProjectionChannel("X", "d", "#2457d6"),
            )
            for mode in ("rgb", "fat"):
                with self.subTest(mode=mode):
                    figure = build_combined_figure(
                        band_data,
                        dos_data,
                        CombinedPlotSettings(
                            energy_range=PlotRange(-2.0, 2.0),
                            band_mode=mode,
                            band_projection_channels=channels,
                        ),
                    )
                    band_axes = figure.axes[0]
                    lines = [
                        collection
                        for collection in band_axes.collections
                        if isinstance(collection, LineCollection)
                    ]
                    scatters = [
                        collection
                        for collection in band_axes.collections
                        if isinstance(collection, PathCollection)
                    ]
                    self.assertEqual(len(lines), 2 if mode == "rgb" else 1)
                    self.assertEqual(len(scatters), 0 if mode == "rgb" else len(channels))
                    figure.clear()

    def test_projected_rgb_and_fat_combined_exports_png_pdf_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            output_folder = root / "output"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_procar(band_folder)
            write_dos_fixture(dos_folder)
            band_data = ProcarAdapter.attach(band_folder, EigenvalAdapter.load(band_folder))
            dos_data = DoscarAdapter.load(dos_folder)
            channels = (
                ProjectionChannel("X", "p", "#d94841"),
                ProjectionChannel("X", "d", "#2457d6"),
            )
            for mode in ("rgb", "fat"):
                with self.subTest(mode=mode):
                    result = render_combined(
                        band_data,
                        dos_data,
                        CombinedPlotSettings(
                            energy_range=PlotRange(-2.0, 2.0),
                            band_mode=mode,
                            band_projection_channels=channels,
                            export=ExportSettings(basename=f"projected_combined_{mode}", dpi=72),
                        ),
                        output_folder,
                    )
                    self.assertEqual(
                        {path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"}
                    )
                    self.assertTrue(all(path.stat().st_size > 0 for path in result.output_files))

    def test_exports_png_pdf_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            output_folder = root / "output"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_dos_fixture(dos_folder)
            settings = CombinedPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=ExportSettings(basename="paper_combined", dpi=120),
            )

            result = render_combined(
                EigenvalAdapter.load(band_folder),
                DoscarAdapter.load(dos_folder),
                settings,
                output_folder,
            )

            self.assertEqual(
                {path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"}
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in result.output_files))


if __name__ == "__main__":
    unittest.main()

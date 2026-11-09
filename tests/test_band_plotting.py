import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

from lyra.adapters.eigenval import EigenvalAdapter
from lyra.models import BandPlotSettings, ExportSettings, PlotRange
from lyra.plotting.band import build_band_figure, render_band
from lyra.theme import COLORS
from tests.fixtures import write_band_fixture


class BandPlottingTests(unittest.TestCase):
    def test_batches_standard_band_segments_per_spin(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)
            data = EigenvalAdapter.load(folder)
            settings = BandPlotSettings(energy_range=PlotRange(-2.0, 2.0))

            figure = build_band_figure(data, settings)

        axes = figure.axes[0]
        band_collections = [
            collection for collection in axes.collections if isinstance(collection, LineCollection)
        ]
        self.assertEqual(len(band_collections), 2)
        self.assertEqual(len(axes.lines), len(data.tick_positions) + 1)
        self.assertEqual(
            [len(collection.get_segments()) for collection in band_collections],
            [2 * len(data.segment_ranges), 2 * len(data.segment_ranges)],
        )
        self.assertEqual(
            [collection.get_capstyle() for collection in band_collections],
            ["projecting", "butt"],
        )
        self.assertEqual(
            [collection.get_joinstyle() for collection in band_collections], ["round", "round"]
        )
        self.assertEqual(
            [collection._lyra_band_labels for collection in band_collections],
            [
                tuple(
                    f"_band_{spin_name}_{band}_{segment}"
                    for band in range(data.energies[spin_name].shape[1])
                    for segment in range(len(data.segment_ranges))
                )
                for spin_name in data.energies
            ],
        )
        expected_segments = []
        for spin_name, matrix in data.energies.items():
            expected_segments.append([
                np.column_stack((data.distances[start:stop], matrix[start:stop, band] - data.efermi))
                for band in range(matrix.shape[1])
                for start, stop in data.segment_ranges
            ])
        for collection, expected in zip(band_collections, expected_segments):
            self.assertEqual(len(collection.get_segments()), len(expected))
            for actual, expected_segment in zip(collection.get_segments(), expected):
                np.testing.assert_array_equal(actual, expected_segment)
        self.assertEqual(tuple(round(value, 3) for value in axes.get_ylim()), (-2.0, 2.0))
        self.assertEqual(tuple(tick.get_text() for tick in axes.get_xticklabels()), data.tick_labels)
        figure.clear()

    def test_exports_png_pdf_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            data = EigenvalAdapter.load(folder)
            export = ExportSettings(basename="paper_band", dpi=120)
            settings = BandPlotSettings(
                energy_range=PlotRange(-2.0, 2.0),
                export=export,
            )

            result = render_band(data, settings, folder)

            self.assertEqual({path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"})
            self.assertTrue(all(path.stat().st_size > 0 for path in result.output_files))
            self.assertEqual(result.preview_file.suffix, ".png")
            expected_background = tuple(
                int(COLORS["preview_background"][index : index + 2], 16)
                for index in (1, 3, 5)
            )
            with Image.open(result.preview_file) as preview:
                preview_rgb = preview.convert("RGB")
                corners = (
                    preview_rgb.getpixel((0, 0)),
                    preview_rgb.getpixel((preview_rgb.width - 1, 0)),
                    preview_rgb.getpixel((0, preview_rgb.height - 1)),
                    preview_rgb.getpixel((preview_rgb.width - 1, preview_rgb.height - 1)),
                )
            self.assertEqual(corners, (expected_background,) * 4)


if __name__ == "__main__":
    unittest.main()

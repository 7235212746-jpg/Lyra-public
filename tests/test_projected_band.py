import base64
from dataclasses import replace
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgb, to_rgba
import numpy as np
from PIL import Image

from lyra.adapters.eigenval import EigenvalAdapter
from lyra.adapters.procar import ProcarAdapter
from lyra.models import (
    BandPlotSettings,
    ExportSettings,
    ParsedBandData,
    PlotRange,
    ProjectionChannel,
    SpinKind,
)
from lyra.plotting.band import (
    _fat_display_indices,
    build_band_figure,
    draw_band_axes,
    render_band,
)
from lyra.plotting.style import publication_rc
from tests.fixtures import write_band_fixture, write_procar


FROZEN_LEGACY_RGB_SHA256 = "06c8780e9488f76d6eee12a6da1aed47015701e081330f3ff972c0a898b064d8"
FROZEN_LEGACY_RGB_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAACdgAAAnYB2mDjTwAAAH9JREFUeJxj+Pjx439cACTHxEAAsOCSuPfyA8O1R88ZWL58+crAyMjEwMzMxPD920+Gs4/fMPz+959BXpiXwUhagIHly/uvDC9uvWFgZmJiYGZlZjBUl2DgYmcDm/Lx40cGFoY/jAwGdjoMTEyMGNb8/fuHgUlCURSrJAwQ9AUAx8Y+z64yFbsAAAAASUVORK5CYII="
)


class ProjectedBandTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def load_data(self, folder, ispin=1):
        write_band_fixture(folder, ispin=ispin)
        write_procar(folder, ispin=ispin)
        return ProcarAdapter.attach(folder, EigenvalAdapter.load(folder))

    @staticmethod
    def _small_rgb_data(segment_ranges=((0, 3), (3, 5)), bands=1, weights=None):
        point_count = 5
        distances = np.linspace(0.0, 1.0, point_count)
        base_energy = np.linspace(-0.3, 0.4, point_count)[:, None]
        if weights is None:
            weights = (
                np.array([[1.0], [0.0], [0.0], [0.2], [0.8]]),
                np.array([[0.0], [1.0], [0.0], [0.7], [0.1]]),
                np.array([[0.0], [0.0], [1.0], [0.1], [0.1]]),
            )
        return ParsedBandData(
            folder=Path("."),
            kpoints=np.zeros((point_count, 3)),
            distances=distances,
            segment_ranges=segment_ranges,
            tick_positions=(0.0, 1.0),
            tick_labels=("", ""),
            energies={"total": np.repeat(base_energy, bands, axis=1)},
            efermi=0.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("X", "Y", "Z"),
            projections={
                "X:s": {"total": np.repeat(weights[0], bands, axis=1)},
                "Y:p": {"total": np.repeat(weights[1], bands, axis=1)},
                "Z:d": {"total": np.repeat(weights[2], bands, axis=1)},
            },
        )

    @classmethod
    def _symmetric_spin_data(cls, segment_ranges=((0, 3), (3, 5))):
        data = cls._small_rgb_data(segment_ranges=segment_ranges)
        energies = data.energies["total"]
        projections = {
            key: {"up": values["total"], "down": values["total"].copy()}
            for key, values in data.projections.items()
        }
        total = np.ones_like(energies)
        return replace(
            data,
            energies={"up": energies, "down": energies.copy()},
            spin_kind=SpinKind.COLLINEAR,
            projections=projections,
            projection_total={"up": total, "down": total.copy()},
        )

    @staticmethod
    def _dense_segmented_spin_data():
        points_per_segment = 201
        distances = np.concatenate(
            (
                np.linspace(0.0, 0.42, points_per_segment),
                np.linspace(0.58, 1.0, points_per_segment),
            )
        )
        point_count = len(distances)
        up = np.full((point_count, 1), 0.2)
        down = np.full((point_count, 1), -0.2)
        weights = np.ones((point_count, 1))
        return ParsedBandData(
            folder=Path("."),
            kpoints=np.zeros((point_count, 3)),
            distances=distances,
            segment_ranges=(
                (0, points_per_segment),
                (points_per_segment, point_count),
            ),
            tick_positions=(0.0, 1.0),
            tick_labels=("", ""),
            energies={"up": up, "down": down},
            efermi=0.0,
            spin_kind=SpinKind.COLLINEAR,
            elements=("X",),
            projections={"X:s": {"up": weights, "down": weights.copy()}},
            projection_total={"up": weights, "down": weights.copy()},
        )

    @staticmethod
    def _rgb_settings(export=None):
        return BandPlotSettings(
            energy_range=PlotRange(-1.0, 1.0),
            mode="rgb",
            projection_channels=(
                ProjectionChannel("X", "s", "#d94841"),
                ProjectionChannel("Y", "p", "#2457d6"),
                ProjectionChannel("Z", "d", "#3d9b68"),
            ),
            export=export or ExportSettings(basename="task5"),
        )

    @staticmethod
    def _frozen_rgb_data():
        return ParsedBandData(
            folder=Path("."),
            kpoints=np.zeros((3, 3)),
            distances=np.array((0.0, 0.5, 1.0)),
            segment_ranges=((0, 3),),
            tick_positions=(0.0, 1.0),
            tick_labels=("", ""),
            energies={"total": np.array(((-0.3,), (0.0,), (0.4,)))},
            efermi=0.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("X", "Y", "Z"),
            projections={
                "X:s": {"total": np.array(((1.0,), (0.0,), (0.0,)))},
                "Y:p": {"total": np.array(((0.0,), (1.0,), (0.0,)))},
                "Z:d": {"total": np.array(((0.0,), (0.0,), (1.0,)))},
            },
        )

    @staticmethod
    def _projection_values(data, channel, spin_name):
        key = channel.element if not channel.orbital else f"{channel.element}:{channel.orbital}"
        channels = data.projections[key]
        if spin_name in channels:
            return channels[spin_name]
        return channels["total"]

    @classmethod
    def _legacy_draw_standard(cls, axes, data, settings, reference, muted=False):
        styles = {
            "total": ("#2457d6", "-"),
            "up": ("#d94841", "-"),
            "down": ("#2457d6", "--"),
        }
        for spin_name, matrix in data.energies.items():
            colour, linestyle = styles.get(spin_name, ("#2457d6", "-"))
            if muted:
                colour = "#8a98aa"
            for band_index in range(matrix.shape[1]):
                shifted = matrix[:, band_index] - reference
                for segment_index, (start, stop) in enumerate(data.segment_ranges):
                    axes.plot(
                        data.distances[start:stop],
                        shifted[start:stop],
                        color=colour,
                        linestyle=linestyle,
                        linewidth=settings.export.line_width * (0.6 if muted else 1.0),
                        alpha=0.65 if muted else 1.0,
                        label=f"_band_{spin_name}_{band_index}_{segment_index}",
                    )

    @classmethod
    def _legacy_draw_rgb(cls, axes, data, settings, reference):
        channel_colours = np.array([to_rgb(channel.colour) for channel in settings.projection_channels])
        for spin_name, matrix in data.energies.items():
            projected = [cls._projection_values(data, channel, spin_name) for channel in settings.projection_channels]
            for band_index in range(matrix.shape[1]):
                shifted = matrix[:, band_index] - reference
                for start, stop in data.segment_ranges:
                    x = data.distances[start:stop]
                    y = shifted[start:stop]
                    points = np.column_stack((x, y))
                    line_segments = np.stack((points[:-1], points[1:]), axis=1)
                    weights = np.stack(
                        [values[start:stop, band_index] for values in projected], axis=1
                    )
                    segment_weights = (weights[:-1] + weights[1:]) / 2.0
                    totals = segment_weights.sum(axis=1, keepdims=True)
                    fractions = np.divide(
                        segment_weights,
                        totals,
                        out=np.zeros_like(segment_weights),
                        where=totals > 1e-12,
                    )
                    colours = fractions @ channel_colours
                    colours[totals[:, 0] <= 1e-12] = to_rgb("#a5adba")
                    axes.add_collection(
                        LineCollection(
                            line_segments,
                            colors=colours,
                            linewidths=settings.export.line_width * 1.35,
                        )
                    )

    @classmethod
    def _legacy_draw_fat(cls, axes, data, settings, reference):
        cls._legacy_draw_standard(axes, data, settings, reference, muted=True)
        for spin_name, matrix in data.energies.items():
            for channel in settings.projection_channels:
                weights = cls._projection_values(data, channel, spin_name)
                for band_index in range(matrix.shape[1]):
                    axes.scatter(
                        data.distances,
                        matrix[:, band_index] - reference,
                        s=np.maximum(weights[:, band_index], 0.0) * settings.fat_scale,
                        color=channel.colour,
                        alpha=0.55,
                        linewidths=0,
                    )

    @classmethod
    def _build_legacy_figure(cls, data, settings):
        width = settings.export.width_mm / 25.4
        reference = data.efermi if settings.reference_energy is None else settings.reference_energy
        with plt.rc_context(publication_rc(settings.export)):
            figure, axes = plt.subplots(figsize=(width, width * 0.72))
            if settings.mode == "standard":
                cls._legacy_draw_standard(axes, data, settings, reference)
            elif settings.mode == "rgb":
                cls._legacy_draw_rgb(axes, data, settings, reference)
            elif settings.mode == "fat":
                cls._legacy_draw_fat(axes, data, settings, reference)
            for position in data.tick_positions:
                axes.axvline(position, color="#20242b", linewidth=0.55, zorder=0)
            axes.axhline(0.0, color="#20242b", linewidth=0.65, linestyle="--", zorder=0)
            axes.set_xticks(data.tick_positions, data.tick_labels)
            axes.set_xlim(float(data.distances[0]), float(data.distances[-1]))
            axes.set_ylim(settings.energy_range.minimum, settings.energy_range.maximum)
            axes.set_xlabel("k path")
            axes.set_ylabel("Energy - $E_F$ (eV)")
            figure.tight_layout()
        return figure

    @staticmethod
    def _expand(values, count):
        values = np.asarray(values)
        if len(values) == count:
            return values
        return np.repeat(values[:1], count, axis=0)

    @classmethod
    def _line_collection_semantics(cls, axes):
        segments = []
        colours = []
        widths = []
        for collection in axes.collections:
            if not isinstance(collection, LineCollection):
                continue
            collection_segments = collection.get_segments()
            collection_colours = cls._expand(collection.get_colors(), len(collection_segments))
            collection_widths = cls._expand(collection.get_linewidths(), len(collection_segments))
            for segment, colour, width in zip(
                collection_segments, collection_colours, collection_widths
            ):
                edge_count = len(segment) - 1
                segments.extend(np.stack((segment[:-1], segment[1:]), axis=1))
                colours.extend([colour] * edge_count)
                widths.extend([width] * edge_count)
        return np.asarray(segments), np.asarray(colours), np.asarray(widths)

    @staticmethod
    def _legacy_line_semantics(axes):
        segments = []
        colours = []
        widths = []
        for line in axes.lines:
            if not line.get_label().startswith("_band_"):
                continue
            points = np.column_stack((line.get_xdata(), line.get_ydata()))
            segments.extend(np.stack((points[:-1], points[1:]), axis=1))
            colours.extend([to_rgba(line.get_color(), line.get_alpha())] * (len(points) - 1))
            widths.extend([line.get_linewidth()] * (len(points) - 1))
        return np.asarray(segments), np.asarray(colours), np.asarray(widths)

    @classmethod
    def _scatter_semantics(cls, axes):
        offsets = []
        sizes = []
        colours = []
        for collection in axes.collections:
            if not isinstance(collection, PathCollection):
                continue
            collection_offsets = np.asarray(collection.get_offsets())
            offsets.extend(collection_offsets)
            sizes.extend(collection.get_sizes())
            colours.extend(cls._expand(collection.get_facecolors(), len(collection_offsets)))
        return np.asarray(offsets), np.asarray(sizes), np.asarray(colours)

    @staticmethod
    def _axes_semantics(axes):
        return {
            "xlim": axes.get_xlim(),
            "ylim": axes.get_ylim(),
            "xlabel": axes.get_xlabel(),
            "ylabel": axes.get_ylabel(),
            "ticks": tuple(label.get_text() for label in axes.get_xticklabels()),
            "guides": tuple(
                (tuple(line.get_xdata()), tuple(line.get_ydata()), line.get_linestyle())
                for line in axes.lines
                if not line.get_label().startswith("_band_")
            ),
        }

    @staticmethod
    def _png_pixels(figure):
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=72)
        image.seek(0)
        with Image.open(image) as decoded:
            return np.asarray(decoded.convert("RGBA"), dtype=np.int16)

    @staticmethod
    def _rendered_colour_mask(figure, axes, y_value, x_range, colour):
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
        left, display_y = axes.transData.transform((x_range[0], y_value))
        right, _ = axes.transData.transform((x_range[1], y_value))
        left = max(0, int(np.ceil(left)))
        right = min(pixels.shape[1], int(np.floor(right)) + 1)
        row = pixels.shape[0] - 1 - int(round(display_y))
        strip = pixels[max(0, row - 2) : row + 3, left:right].astype(float)
        target = np.asarray(to_rgb(colour)) * 255.0
        return np.any(np.linalg.norm(strip - target, axis=2) < 70.0, axis=0)

    @staticmethod
    def _fontless_rgb_png(data, settings):
        figure, axes = plt.subplots(figsize=(0.5, 0.5), dpi=settings.export.dpi)
        draw_band_axes(axes, data, settings)
        axes.set_position((0, 0, 1, 1))
        axes.set_axis_off()
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=settings.export.dpi, facecolor="white")
        figure.clear()
        return image.getvalue()

    def test_rgb_mode_rejects_more_than_three_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.load_data(Path(tmp))
            channels = tuple(
                ProjectionChannel("X", orbital)
                for orbital in ("s", "p", "d", "px")
            )
            settings = BandPlotSettings(
                energy_range=PlotRange(-2, 2), mode="rgb", projection_channels=channels
            )

            with self.assertRaisesRegex(ValueError, "at most three"):
                build_band_figure(data, settings)

    def test_rgb_mode_uses_coloured_line_collections(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.load_data(Path(tmp))
            settings = BandPlotSettings(
                energy_range=PlotRange(-2, 2),
                mode="rgb",
                projection_channels=(
                    ProjectionChannel("X", "p", "#d94841"),
                    ProjectionChannel("X", "d", "#2457d6"),
                ),
            )
            figure = build_band_figure(data, settings)

        self.assertGreater(len(figure.axes[0].collections), 0)
        figure.clear()

    def test_rgb_single_channel_keeps_unselected_character_black(self):
        selected = np.full((5, 1), 0.2)
        zero = np.zeros((5, 1))
        data = replace(
            self._small_rgb_data(weights=(selected, zero, zero)),
            projection_total={"total": np.ones((5, 1))},
        )
        settings = BandPlotSettings(
            energy_range=PlotRange(-1.0, 1.0),
            mode="rgb",
            projection_channels=(ProjectionChannel("X", "s", "#ff0000"),),
        )

        figure = build_band_figure(data, settings)

        backbone, projection = figure.axes[0].collections[:2]
        np.testing.assert_allclose(backbone.get_colors()[0], to_rgba("#000000"))
        colours = projection.get_colors()
        expected = np.tile((1.0, 0.0, 0.0, 0.2), (len(colours), 1))
        np.testing.assert_allclose(colours, expected, atol=1e-7)
        figure.clear()

    def test_rgb_complete_band_backbone_colour_is_configurable(self):
        settings = replace(self._rgb_settings(), complete_band_colour="#654321")

        figure = build_band_figure(self._small_rgb_data(), settings)

        backbone = figure.axes[0].collections[0]
        np.testing.assert_allclose(backbone.get_colors()[0], to_rgba("#654321"))
        figure.clear()

    def test_rgb_degenerate_bands_and_ranges_are_noops(self):
        cases = (
            ("zero bands", ((0, 3), (3, 5)), 0),
            ("empty ranges", (), 1),
            ("zero-length range", ((2, 2),), 1),
            ("single-point range", ((2, 3),), 1),
        )
        for name, ranges, bands in cases:
            with self.subTest(name=name):
                figure = build_band_figure(
                    self._small_rgb_data(ranges, bands), self._rgb_settings()
                )
                collections = [
                    collection
                    for collection in figure.axes[0].collections
                    if isinstance(collection, LineCollection)
                ]
                self.assertEqual(collections, [])
                figure.clear()

    def test_rgb_zero_totals_are_black(self):
        zero = np.zeros((5, 1))
        figure = build_band_figure(
            self._small_rgb_data(weights=(zero, zero, zero)), self._rgb_settings()
        )

        backbone, projection = figure.axes[0].collections[:2]
        np.testing.assert_array_equal(backbone.get_colors()[0], to_rgba("#000000"))
        self.assertTrue(np.all(projection.get_colors()[:, 3] == 0.0))
        figure.clear()

    def test_rgb_multiple_channels_keep_absolute_all_projection_weights(self):
        first = np.full((5, 1), 0.2)
        second = np.full((5, 1), 0.3)
        zero = np.zeros((5, 1))
        data = replace(
            self._small_rgb_data(weights=(first, second, zero)),
            projection_total={"total": np.ones((5, 1))},
        )
        settings = BandPlotSettings(
            energy_range=PlotRange(-1.0, 1.0),
            mode="rgb",
            projection_channels=(
                ProjectionChannel("X", "s", "#ff0000"),
                ProjectionChannel("Y", "p", "#0000ff"),
            ),
        )

        figure = build_band_figure(data, settings)

        colours = figure.axes[0].collections[1].get_colors()
        expected = np.tile((0.4, 0.0, 0.6, 0.5), (len(colours), 1))
        np.testing.assert_allclose(colours, expected, atol=1e-7)
        figure.clear()

    def test_rgb_nan_weights_leave_only_complete_band_backbone(self):
        nan = np.full((5, 1), np.nan)
        data = self._small_rgb_data(weights=(nan, np.ones((5, 1)), np.ones((5, 1))))
        settings = self._rgb_settings()
        candidate = build_band_figure(data, settings)

        backbone, projection = candidate.axes[0].collections[:2]
        np.testing.assert_array_equal(backbone.get_colors()[0], to_rgba("#000000"))
        self.assertTrue(np.all(projection.get_colors()[:, 3] == 0.0))
        candidate.clear()

    def test_rgb_collection_keeps_private_path_labels_as_metadata(self):
        figure = build_band_figure(self._small_rgb_data(), self._rgb_settings())

        collection = next(
            item for item in figure.axes[0].collections
            if getattr(item, "_lyra_layer", None) == "projection"
        )
        self.assertEqual(
            collection._lyra_band_labels,
            ("_band_total_0_0", "_band_total_0_1"),
        )
        figure.clear()

    def test_rgb_metadata_maps_each_edge_to_its_discontinuous_source_path(self):
        figure = build_band_figure(self._small_rgb_data(), self._rgb_settings())

        collection = next(
            item for item in figure.axes[0].collections
            if getattr(item, "_lyra_layer", None) == "projection"
        )
        labels = ("_band_total_0_0", "_band_total_0_1")
        self.assertEqual(len(collection.get_segments()), 3)
        self.assertEqual(collection._lyra_band_labels, labels)
        self.assertEqual(
            collection._lyra_band_edge_labels,
            (labels[0], labels[0], labels[1]),
        )
        self.assertEqual(
            collection._lyra_band_path_slices,
            ((labels[0], 0, 2), (labels[1], 2, 3)),
        )
        for label, start, stop in collection._lyra_band_path_slices:
            self.assertEqual(
                collection._lyra_band_edge_labels[start:stop], (label,) * (stop - start)
            )
        figure.clear()

    def test_rgb_intentionally_replaces_legacy_single_layer_artifact(self):
        export = ExportSettings(basename="task5", dpi=16, width_mm=12, font_family="DejaVu Sans")
        png = self._fontless_rgb_png(self._frozen_rgb_data(), self._rgb_settings(export))

        self.assertEqual(hashlib.sha256(FROZEN_LEGACY_RGB_PNG).hexdigest(), FROZEN_LEGACY_RGB_SHA256)
        self.assertNotEqual(hashlib.sha256(png).hexdigest(), FROZEN_LEGACY_RGB_SHA256)
        self.assertNotEqual(png, FROZEN_LEGACY_RGB_PNG)

    def test_rgb_spin_collections_preserve_colours_and_guides_match_legend(self):
        data = self._symmetric_spin_data()
        figure = build_band_figure(data, self._rgb_settings())

        collections = [
            collection
            for collection in figure.axes[0].collections
            if isinstance(collection, LineCollection)
        ]
        self.assertEqual(len(collections), 4)
        projection_collections = [
            collection for collection in collections
            if getattr(collection, "_lyra_layer", None) == "projection"
        ]
        np.testing.assert_array_equal(
            projection_collections[0].get_colors(),
            projection_collections[1].get_colors(),
        )
        guides = [
            line
            for line in figure.axes[0].lines
            if line.get_label().startswith("_band_spin_guide_")
        ]
        self.assertEqual(len(guides), 4)
        self.assertEqual(
            [line.get_linestyle() for line in guides],
            ["-", "-", "--", "--"],
        )
        self.assertEqual(
            [line.get_color() for line in guides],
            ["#d94841", "#d94841", "#2457d6", "#2457d6"],
        )
        legend = figure.axes[0].get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["Spin up", "Spin down"],
        )
        self.assertEqual(
            [line.get_linestyle() for line in legend.get_lines()],
            ["-", "--"],
        )
        figure.clear()

    def test_dense_segmented_rgb_spin_guides_remain_visible_in_agg_render(self):
        data = self._dense_segmented_spin_data()
        settings = BandPlotSettings(
            energy_range=PlotRange(-0.5, 0.5),
            mode="rgb",
            projection_channels=(ProjectionChannel("X", "s", "#3d9b68"),),
        )
        figure, axes = plt.subplots(figsize=(6.0, 4.0), dpi=100)
        draw_band_axes(axes, data, settings)

        segment_interiors = ((0.03, 0.39), (0.61, 0.97))
        for x_range in segment_interiors:
            with self.subTest(spin="up", x_range=x_range):
                up_mask = self._rendered_colour_mask(
                    figure, axes, 0.2, x_range, "#d94841"
                )
                self.assertGreater(float(np.mean(up_mask)), 0.9)
            with self.subTest(spin="down", x_range=x_range):
                down_mask = self._rendered_colour_mask(
                    figure, axes, -0.2, x_range, "#2457d6"
                )
                coverage = float(np.mean(down_mask))
                transitions = int(np.count_nonzero(down_mask[1:] != down_mask[:-1]))
                self.assertGreater(coverage, 0.15)
                self.assertLess(coverage, 0.85)
                self.assertGreaterEqual(transitions, 6)

        for y_value, colour in ((0.2, "#d94841"), (-0.2, "#2457d6")):
            with self.subTest(gap_spin_colour=colour):
                gap_mask = self._rendered_colour_mask(
                    figure, axes, y_value, (0.47, 0.53), colour
                )
                self.assertFalse(np.any(gap_mask))
        figure.clear()

    def test_rgb_batches_complete_and_projection_layers_per_spin(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.load_data(Path(tmp), ispin=2)
            settings = BandPlotSettings(
                energy_range=PlotRange(-2, 2),
                mode="rgb",
                projection_channels=(
                    ProjectionChannel("X", "p", "#d94841"),
                    ProjectionChannel("X", "d", "#2457d6"),
                    ProjectionChannel("X", "s", "#3d9b68"),
                ),
            )
            candidate = build_band_figure(data, settings)

        candidate_collections = [
            collection for collection in candidate.axes[0].collections if isinstance(collection, LineCollection)
        ]
        self.assertEqual(len(candidate_collections), 2 * len(data.energies))
        self.assertEqual(
            [getattr(collection, "_lyra_layer", None) for collection in candidate_collections],
            ["complete-band", "complete-band", "projection", "projection"],
        )
        candidate.clear()

    def test_fat_mode_uses_projection_weighted_scatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.load_data(Path(tmp))
            settings = BandPlotSettings(
                energy_range=PlotRange(-2, 2),
                mode="fat",
                projection_channels=(ProjectionChannel("X", "d", "#2457d6"),),
                fat_scale=200,
            )
            figure = build_band_figure(data, settings)

        sizes = [
            size
            for collection in figure.axes[0].collections
            if isinstance(collection, PathCollection)
            for size in collection.get_sizes()
        ]
        self.assertTrue(sizes)
        self.assertGreater(max(sizes), 0)
        figure.clear()

    def test_fat_indices_keep_each_segment_point_once_for_both_spins(self):
        ranges = ((-2, 3), (2, 5), (4, 4), (4, 7))
        expected = np.arange(6)
        for spin_name in ("up", "down"):
            with self.subTest(spin=spin_name):
                np.testing.assert_array_equal(
                    _fat_display_indices(6, ranges, spin_name),
                    expected,
                )
                np.testing.assert_array_equal(
                    _fat_display_indices(6, ((2, 3),), spin_name),
                    np.array([2]),
                )

    def test_symmetric_fat_spins_have_identical_coverage_and_marker_areas(self):
        data = self._symmetric_spin_data()
        settings = BandPlotSettings(
            energy_range=PlotRange(-1, 1),
            mode="fat",
            projection_channels=(ProjectionChannel("X", "s", "#8b5cf6"),),
            fat_scale=200,
        )
        figure = build_band_figure(data, settings)

        scatters = [
            collection
            for collection in figure.axes[0].collections
            if isinstance(collection, PathCollection)
        ]
        self.assertEqual(len(scatters), 2)
        np.testing.assert_array_equal(scatters[0].get_offsets(), scatters[1].get_offsets())
        np.testing.assert_array_equal(scatters[0].get_sizes(), scatters[1].get_sizes())
        expected_sizes = (
            np.maximum(data.projections["X:s"]["up"], 0.0) * settings.fat_scale
        ).T.reshape(-1)
        np.testing.assert_array_equal(scatters[0].get_sizes(), expected_sizes)
        figure.clear()

    def test_rgb_rejects_semantically_overlapping_channels(self):
        overlap_cases = (
            (ProjectionChannel("X", "s"), ProjectionChannel("X", "s")),
            (ProjectionChannel("X"), ProjectionChannel("X", "s")),
            (ProjectionChannel("X", ""), ProjectionChannel("X", "s")),
            (ProjectionChannel("X", "p"), ProjectionChannel("X", "px")),
            (ProjectionChannel("X", "px"), ProjectionChannel("X", "px")),
            (ProjectionChannel("X", "x2-y2"), ProjectionChannel("X", "dx2-y2")),
            (ProjectionChannel("X", "z2"), ProjectionChannel("X", "dz2")),
        )
        for first, second in overlap_cases:
            with self.subTest(first=first, second=second):
                settings = BandPlotSettings(
                    energy_range=PlotRange(-1, 1),
                    mode="rgb",
                    projection_channels=(first, second),
                )
                with self.assertRaises(ValueError) as raised:
                    build_band_figure(self._small_rgb_data(), settings)
                message = str(raised.exception)
                self.assertIn(
                    first.element if not first.orbital else f"{first.element}:{first.orbital}",
                    message,
                )
                self.assertIn(
                    second.element if not second.orbital else f"{second.element}:{second.orbital}",
                    message,
                )
                self.assertIn("RGB channels must not overlap", message)

    def test_rgb_accepts_disjoint_elements_and_disjoint_orbitals(self):
        base = self._small_rgb_data()
        orbital_weight = np.full((5, 1), 0.25)
        data = replace(
            base,
            projections={
                **base.projections,
                "X:px": {"total": orbital_weight},
                "X:py": {"total": orbital_weight.copy()},
            },
            projection_total={"total": np.ones((5, 1))},
        )
        channel_sets = (
            (ProjectionChannel("X", "s"), ProjectionChannel("Y", "p")),
            (ProjectionChannel("X", "px"), ProjectionChannel("X", "py")),
        )
        for channels in channel_sets:
            with self.subTest(channels=channels):
                figure = build_band_figure(
                    data,
                    BandPlotSettings(
                        energy_range=PlotRange(-1, 1),
                        mode="rgb",
                        projection_channels=channels,
                    ),
                )
                self.assertTrue(figure.axes[0].collections)
                figure.clear()

    def test_fat_supports_four_channels_with_user_colours(self):
        data = self._small_rgb_data()
        fourth_weight = np.full((5, 1), 0.4)
        data = replace(
            data,
            elements=(*data.elements, "W"),
            projections={**data.projections, "W:f": {"total": fourth_weight}},
        )
        colours = ("#c026d3", "#0f766e", "#ca8a04", "#334155")
        channels = tuple(
            ProjectionChannel(element, orbital, colour)
            for element, orbital, colour in zip(
                ("X", "Y", "Z", "W"), ("s", "p", "d", "f"), colours
            )
        )

        figure = build_band_figure(
            data,
            BandPlotSettings(
                energy_range=PlotRange(-1, 1),
                mode="fat",
                projection_channels=channels,
            ),
        )

        scatters = [
            collection
            for collection in figure.axes[0].collections
            if isinstance(collection, PathCollection)
        ]
        self.assertEqual(len(scatters), 4)
        np.testing.assert_allclose(
            [collection.get_facecolors()[0] for collection in scatters],
            [to_rgba(colour, alpha=0.55) for colour in colours],
        )
        figure.clear()

    def test_fat_preserves_overlap_order_and_zorder(self):
        data = self._small_rgb_data(bands=2)
        settings = BandPlotSettings(
            energy_range=PlotRange(-1, 1),
            mode="fat",
            projection_channels=(
                ProjectionChannel("X", "s", "#d94841"),
                ProjectionChannel("Y", "p", "#2457d6"),
            ),
            fat_scale=200,
        )
        legacy = self._build_legacy_figure(data, settings)
        candidate = build_band_figure(data, settings)

        legacy_scatters = [
            collection for collection in legacy.axes[0].collections if isinstance(collection, PathCollection)
        ]
        candidate_scatters = [
            collection for collection in candidate.axes[0].collections if isinstance(collection, PathCollection)
        ]
        self.assertEqual([collection.get_zorder() for collection in candidate_scatters], [1, 1])
        legacy_offsets, legacy_sizes, legacy_colours = self._scatter_semantics(legacy.axes[0])
        candidate_offsets, candidate_sizes, candidate_colours = self._scatter_semantics(candidate.axes[0])
        np.testing.assert_array_equal(candidate_offsets, legacy_offsets)
        np.testing.assert_array_equal(candidate_sizes, legacy_sizes)
        np.testing.assert_array_equal(candidate_colours, legacy_colours)
        self.assertTrue(np.array_equal(candidate_offsets[:5], candidate_offsets[5:10]))
        self.assertEqual(len(legacy_scatters), 4)
        legacy.clear()
        candidate.clear()

    def test_standard_and_fat_zero_band_cases_create_no_data_artists(self):
        data = self._small_rgb_data(bands=0)
        settings_by_mode = (
            BandPlotSettings(energy_range=PlotRange(-1, 1), mode="standard"),
            BandPlotSettings(
                energy_range=PlotRange(-1, 1),
                mode="fat",
                projection_channels=(ProjectionChannel("X", "s", "#d94841"),),
            ),
        )
        for settings in settings_by_mode:
            with self.subTest(mode=settings.mode):
                figure = build_band_figure(data, settings)
                data_artists = [
                    collection
                    for collection in figure.axes[0].collections
                    if isinstance(collection, (LineCollection, PathCollection))
                ]
                self.assertEqual(data_artists, [])
                figure.clear()

    def test_standard_and_fat_single_point_ranges_keep_their_point_data(self):
        data = self._small_rgb_data(segment_ranges=((2, 3),))
        settings_by_mode = (
            BandPlotSettings(energy_range=PlotRange(-1, 1), mode="standard"),
            BandPlotSettings(
                energy_range=PlotRange(-1, 1),
                mode="fat",
                projection_channels=(ProjectionChannel("X", "s", "#d94841"),),
                fat_scale=200,
            ),
        )
        expected_point = np.array([[data.distances[2], data.energies["total"][2, 0]]])
        for settings in settings_by_mode:
            with self.subTest(mode=settings.mode):
                figure = build_band_figure(data, settings)
                lines = [
                    collection
                    for collection in figure.axes[0].collections
                    if isinstance(collection, LineCollection)
                ]
                self.assertEqual(len(lines), 1)
                np.testing.assert_array_equal(lines[0].get_segments()[0], expected_point)
                if settings.mode == "fat":
                    scatter = next(
                        collection
                        for collection in figure.axes[0].collections
                        if isinstance(collection, PathCollection)
                    )
                    np.testing.assert_array_equal(
                        scatter.get_offsets(),
                        expected_point,
                    )
                figure.clear()

    def test_fat_nan_negative_and_zero_weights_match_legacy_sizes(self):
        weights = np.array([[np.nan], [-0.5], [0.0], [0.25], [1.0]])
        data = self._small_rgb_data(weights=(weights, weights, weights))
        settings = BandPlotSettings(
            energy_range=PlotRange(-1, 1),
            mode="fat",
            projection_channels=(ProjectionChannel("X", "s", "#d94841"),),
            fat_scale=200,
        )
        legacy = self._build_legacy_figure(data, settings)
        candidate = build_band_figure(data, settings)

        legacy_sizes = next(
            collection.get_sizes()
            for collection in legacy.axes[0].collections
            if isinstance(collection, PathCollection)
        )
        candidate_sizes = next(
            collection.get_sizes()
            for collection in candidate.axes[0].collections
            if isinstance(collection, PathCollection)
        )
        np.testing.assert_allclose(candidate_sizes, legacy_sizes, equal_nan=True)
        np.testing.assert_allclose(
            candidate_sizes,
            np.array([np.nan, 0.0, 0.0, 50.0, 200.0]),
            equal_nan=True,
        )
        legacy.clear()
        candidate.clear()

    def test_projected_modes_export_small_png_pdf_and_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = self.load_data(folder)
            for mode, channels in (
                (
                    "rgb",
                    (
                        ProjectionChannel("X", "p", "#d94841"),
                        ProjectionChannel("X", "d", "#2457d6"),
                        ProjectionChannel("X", "s", "#3d9b68"),
                    ),
                ),
                (
                    "fat",
                    (
                        ProjectionChannel("X", "s", "#d94841"),
                        ProjectionChannel("X", "d", "#2457d6"),
                    ),
                ),
            ):
                with self.subTest(mode=mode):
                    result = render_band(
                        data,
                        BandPlotSettings(
                            energy_range=PlotRange(-2, 2),
                            mode=mode,
                            projection_channels=channels,
                            export=ExportSettings(basename=f"projected_{mode}", dpi=72),
                        ),
                        folder,
                    )
                    self.assertEqual(
                        {path.suffix for path in result.output_files}, {".png", ".pdf", ".svg"}
                    )
                    self.assertTrue(all(path.stat().st_size > 0 for path in result.output_files))

    def test_batched_rendering_matches_legacy_low_dpi_pixels_within_one_channel_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.load_data(Path(tmp), ispin=2)
            export = ExportSettings(
                basename="task5",
                dpi=72,
                width_mm=100,
                font_family="DejaVu Sans",
            )
            settings_by_mode = (
                BandPlotSettings(
                    energy_range=PlotRange(-2, 2),
                    mode="standard",
                    export=export,
                ),
            )
            for settings in settings_by_mode:
                legacy = self._build_legacy_figure(data, settings)
                candidate = build_band_figure(data, settings)
                legacy_pixels = self._png_pixels(legacy)
                candidate_pixels = self._png_pixels(candidate)
                delta = np.abs(candidate_pixels - legacy_pixels)
                self.assertEqual(candidate_pixels.shape, legacy_pixels.shape)
                self.assertLessEqual(int(delta.max()), 1)
                self.assertLessEqual(int(np.count_nonzero(delta)), 512)
                legacy.clear()
                candidate.clear()


if __name__ == "__main__":
    unittest.main()

import tempfile
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lyra.local_zoom_window import (
    BandLocalZoomWindow,
    CombinedLocalZoomWindow,
    local_zoom_rc,
    validate_axis_limits,
    validate_font_size,
    validate_line_width,
)
from lyra.models import BandPlotSettings, CombinedPlotSettings, ExportSettings, PlotRange, ProjectionChannel
from tests.test_projection_store import parsed_band_data, parsed_dos_data


def settings():
    return BandPlotSettings(
        energy_range=PlotRange(-2.0, 2.0),
        mode="standard",
        export=ExportSettings("zoom", formats=("png",), dpi=180, width_mm=120),
    )


class LocalZoomModelTests(unittest.TestCase):
    def test_local_zoom_uses_readable_plot_and_legend_fonts(self):
        style = local_zoom_rc(settings().export, 22)

        self.assertEqual(style["font.size"], 22.0)
        self.assertEqual(style["axes.labelsize"], 24.0)
        self.assertEqual(style["xtick.labelsize"], 22.0)
        self.assertEqual(style["ytick.labelsize"], 22.0)
        self.assertEqual(style["legend.fontsize"], 22.0)

    def test_font_size_requires_a_finite_value_in_supported_range(self):
        self.assertEqual(validate_font_size("30"), 30.0)
        for value in ("nope", 7, 49, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_font_size(value)

    def test_line_width_requires_a_finite_value_in_supported_range(self):
        self.assertEqual(validate_line_width("2.5"), 2.5)
        for value in ("nope", 0.2, 6.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_line_width(value)

    def test_axis_limits_require_finite_ordered_values(self):
        self.assertEqual(validate_axis_limits((0, 0.8, -1, 1)), (0.0, 0.8, -1.0, 1.0))
        for values in ((1, 0, -1, 1), (0, 1, 2, 2), (0, float("nan"), -1, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_axis_limits(values)

    def test_combined_zoom_pair_validation_rejects_reversed_ranges(self):
        self.assertEqual(CombinedLocalZoomWindow._pair("-2", "3", "能量"), (-2.0, 3.0))
        with self.assertRaises(ValueError):
            CombinedLocalZoomWindow._pair("5", "1", "DOS")


class LocalZoomWindowTests(unittest.TestCase):
    @staticmethod
    def _gapped_data():
        data = parsed_band_data()
        return replace(
            data,
            energies={
                "up": np.array([[4.0, 6.0, 7.0], [4.8, 6.4, 7.2]]),
                "down": np.array([[3.9, 6.1, 7.1], [4.7, 6.5, 7.3]]),
            },
            occupations={
                "up": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "down": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            },
        )

    def test_band_window_auto_analysis_marks_real_vbm_and_cbm(self):
        root = tk.Tk()
        root.withdraw()
        window = BandLocalZoomWindow(root, self._gapped_data(), settings(), Path.cwd())
        try:
            result = window._run_edge_analysis(show_report=False)

            self.assertEqual(result.classification, "semiconductor")
            self.assertEqual(len(window._edge_artists), 3)
            self.assertIn("自动带边", window.status_var.get())
        finally:
            window.close()
            root.destroy()

    def test_combined_window_auto_analysis_uses_dos_energy_reference_for_markers(self):
        root = tk.Tk()
        root.withdraw()
        dos = replace(parsed_dos_data(), efermi=5.5)
        window = CombinedLocalZoomWindow(
            root,
            self._gapped_data(),
            dos,
            CombinedPlotSettings(PlotRange(-2, 2)),
            Path.cwd(),
        )
        try:
            result = window._run_edge_analysis(show_report=False)

            self.assertEqual(result.reference_energy, 5.5)
            offsets = window._edge_artists[0].get_offsets()
            self.assertAlmostEqual(float(offsets[0, 1]), result.vbm.energy - 5.5)
        finally:
            window.close()
            root.destroy()

    def test_band_window_clicks_snap_and_create_persistent_gap_annotation(self):
        root = tk.Tk()
        root.withdraw()
        data = parsed_band_data()
        window = BandLocalZoomWindow(root, data, settings(), Path.cwd())
        try:
            window._toggle_gap_measurement()
            selected = ((0, 0), (1, 2))
            for kpoint_index, band_index in selected:
                x = float(data.distances[kpoint_index])
                y = float(data.energies["up"][kpoint_index, band_index] - data.efermi)
                display = window.axes.transData.transform((x, y))
                window._on_gap_click(
                    SimpleNamespace(
                        inaxes=window.axes,
                        button=1,
                        x=float(display[0]),
                        y=float(display[1]),
                    )
                )

            self.assertIsNotNone(window._gap_measurement)
            self.assertEqual(len(window._gap_points), 2)
            self.assertGreaterEqual(len(window._gap_artists), 4)
            self.assertIn("所选两点能量差", window.status_var.get())
            window.apply_plot_style()
            self.assertGreaterEqual(len(window._gap_artists), 4)
        finally:
            window.close()
            root.destroy()

    def test_combined_window_owns_two_live_axes_and_applies_real_ranges(self):
        root = tk.Tk()
        root.withdraw()
        window = CombinedLocalZoomWindow(
            root,
            parsed_band_data(),
            parsed_dos_data(),
            CombinedPlotSettings(PlotRange(-2, 2)),
            Path.cwd(),
        )
        try:
            self.assertIsNotNone(window.navigation)
            self.assertEqual(len(window.figure.axes), 2)
            window.energy_min_var.set("-1")
            window.energy_max_var.set("1")
            window.dos_min_var.set("-20")
            window.dos_max_var.set("20")
            window.apply_limits()
            self.assertAlmostEqual(window.band_axes.get_ylim()[0], -1)
            self.assertAlmostEqual(window.dos_axes.get_xlim()[1], 20)
            window._toggle_gap_measurement()
            window._on_gap_click(
                SimpleNamespace(
                    inaxes=window.dos_axes,
                    button=1,
                    x=0.0,
                    y=0.0,
                )
            )
            self.assertEqual(window._gap_points, [])
        finally:
            window.close()
            root.destroy()

    def test_rgb_spin_legend_uses_local_zoom_font_size(self):
        root = tk.Tk()
        root.withdraw()
        rgb_settings = BandPlotSettings(
            energy_range=PlotRange(-2.0, 2.0),
            mode="rgb",
            projection_channels=(ProjectionChannel("A", "s"),),
            export=settings().export,
        )
        window = BandLocalZoomWindow(
            root,
            parsed_band_data(),
            rgb_settings,
            Path.cwd(),
        )
        try:
            window.update()
            legend = window.axes.get_legend()
            self.assertIsNotNone(legend)
            self.assertTrue(
                all(text.get_fontsize() >= 22.0 for text in legend.get_texts())
            )
        finally:
            window.close()
            root.destroy()

    def test_apply_and_reset_preserve_scientific_axes(self):
        root = tk.Tk()
        root.withdraw()
        window = BandLocalZoomWindow(root, parsed_band_data(), settings(), Path.cwd())
        try:
            window.update()
            original = (*window.axes.get_xlim(), *window.axes.get_ylim())
            window.x_min_var.set("0.2")
            window.x_max_var.set("0.7")
            window.y_min_var.set("-0.5")
            window.y_max_var.set("0.6")

            window.apply_limits()

            self.assertEqual(tuple(round(value, 3) for value in window.axes.get_xlim()), (0.2, 0.7))
            self.assertEqual(tuple(round(value, 3) for value in window.axes.get_ylim()), (-0.5, 0.6))
            self.assertEqual(window.axes.get_xlabel(), "k path")
            self.assertIn("Energy", window.axes.get_ylabel())
            self.assertGreaterEqual(window.axes.xaxis.label.get_fontsize(), 24.0)
            self.assertTrue(
                all(label.get_fontsize() >= 22.0 for label in window.axes.get_xticklabels())
            )

            window.font_size_var.set("30")
            window.line_width_var.set("2.6")
            limits_before_style = (*window.axes.get_xlim(), *window.axes.get_ylim())
            window.apply_plot_style()

            self.assertEqual(window.axes.xaxis.label.get_fontsize(), 32.0)
            self.assertTrue(
                all(label.get_fontsize() == 30.0 for label in window.axes.get_xticklabels())
            )
            self.assertEqual(window.axes.collections[0].get_linewidths()[0], 2.6)
            self.assertEqual(
                tuple(round(value, 6) for value in (*window.axes.get_xlim(), *window.axes.get_ylim())),
                tuple(round(value, 6) for value in limits_before_style),
            )
            window.reset_limits()
            self.assertEqual(
                tuple(round(value, 6) for value in (*window.axes.get_xlim(), *window.axes.get_ylim())),
                tuple(round(value, 6) for value in original),
            )
        finally:
            window.close()
            root.destroy()

    def test_save_current_view_writes_selected_vector_or_raster_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = tk.Tk()
            root.withdraw()
            window = BandLocalZoomWindow(root, parsed_band_data(), settings(), Path(tmp))
            output = Path(tmp) / "local-band.svg"
            try:
                with patch(
                    "lyra.local_zoom_window.filedialog.asksaveasfilename",
                    return_value=str(output),
                ):
                    saved = window.save_current_view()

                self.assertEqual(saved, output)
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 100)
            finally:
                window.close()
                root.destroy()


if __name__ == "__main__":
    unittest.main()

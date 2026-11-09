from __future__ import annotations

import colorsys
import unittest

from lyra.theme import COLORS


def _relative_luminance(hex_color: str) -> float:
    channels = tuple(int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5))

    def linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((_relative_luminance(foreground), _relative_luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _hue_family(hex_color: str) -> int | None:
    red, green, blue = (int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5))
    hue, saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
    return round(hue * 12) % 12 if saturation >= 0.25 else None


class ThemeTokensTests(unittest.TestCase):
    def test_semantic_material_tokens_include_white_plot_and_preview_backgrounds(self) -> None:
        self.assertTrue(
            {
                "background",
                "shell",
                "panel",
                "panel_alt",
                "text",
                "muted",
                "accent",
                "warning",
                "caution",
                "success",
                "plot_background",
                "preview_background",
                "export_background",
            }.issubset(COLORS)
        )
        self.assertEqual(COLORS["plot_background"], "#ffffff")
        self.assertEqual(COLORS["preview_background"], "#ffffff")
        self.assertEqual(COLORS["export_background"], "#ffffff")

    def test_text_meets_wcag_aa_contrast_on_each_material_panel(self) -> None:
        for panel_name in ("shell", "panel", "panel_alt", "input"):
            with self.subTest(panel=panel_name):
                self.assertGreaterEqual(_contrast_ratio(COLORS["text"], COLORS[panel_name]), 4.5)

    def test_status_label_pairing_meets_wcag_aa_contrast(self) -> None:
        self.assertGreaterEqual(
            _contrast_ratio(COLORS["accent"], COLORS["accent_soft"]),
            4.5,
        )

    def test_secondary_text_meets_wcag_aa_on_material_surfaces(self) -> None:
        for foreground_name in ("muted", "subtle"):
            for panel_name in ("background", "shell", "panel", "panel_alt", "input"):
                with self.subTest(foreground=foreground_name, panel=panel_name):
                    self.assertGreaterEqual(
                        _contrast_ratio(COLORS[foreground_name], COLORS[panel_name]),
                        4.5,
                    )

    def test_semantic_accents_span_multiple_hue_families(self) -> None:
        hues = {
            _hue_family(COLORS[name])
            for name in ("accent", "warning", "caution", "success")
        }
        hues.discard(None)
        self.assertGreaterEqual(len(hues), 3)

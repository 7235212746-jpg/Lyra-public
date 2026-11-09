from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from lyra.models import ExportSettings
from lyra.plotting.style import publication_rc
from lyra.theme import COLORS


class PublicationStyleTests(unittest.TestCase):
    def test_publication_rc_consumes_plot_and_export_background_tokens(self) -> None:
        rc = publication_rc(ExportSettings(basename="style"))

        self.assertEqual(rc.get("figure.facecolor"), COLORS["plot_background"])
        self.assertEqual(rc.get("axes.facecolor"), COLORS["plot_background"])
        self.assertEqual(rc.get("savefig.facecolor"), COLORS["export_background"])
        self.assertFalse(rc["savefig.transparent"])

    def test_publication_rc_applies_white_plot_surfaces(self) -> None:
        with plt.rc_context(publication_rc(ExportSettings(basename="style"))):
            figure, axes = plt.subplots()
        try:
            self.assertEqual(to_hex(figure.get_facecolor()), COLORS["plot_background"])
            self.assertEqual(to_hex(axes.get_facecolor()), COLORS["plot_background"])
        finally:
            plt.close(figure)

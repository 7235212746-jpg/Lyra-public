from pathlib import Path
import tkinter as tk
import unittest
from unittest.mock import Mock

from lyra.cache import CacheIdentity, CacheSource
from lyra.models import BandStateSelection
from lyra.paired_band import PairedBandSession, assess_index_compatibility
from lyra.paired_band_window import PairedBandInspector
from tests.test_paired_band import make_index


def identity(name):
    return CacheIdentity(1, "test", "lite", "band", (CacheSource(name, 1, 1),))


def session():
    left = make_index()
    right = make_index(
        energies=((-0.95, 0.15), (-0.75, 0.25), (-0.55, 0.35)),
    )
    return PairedBandSession(
        left=left,
        right=right,
        left_identity=identity("left"),
        right_identity=identity("right"),
        left_folder=Path("left"),
        right_folder=Path("right"),
        left_mode="lite",
        right_mode="lite",
        compatibility=assess_index_compatibility(left, right),
    )


class PairedBandWindowTests(unittest.TestCase):
    def test_selecting_left_state_populates_ranked_right_candidates(self):
        root = tk.Tk()
        root.withdraw()
        query = Mock(return_value=None)
        window = PairedBandInspector(root, session(), query_detail=query)
        try:
            window.update()
            window.select_state("left", BandStateSelection("total", 1, 0))

            self.assertEqual(window.left_selection.band_index, 0)
            self.assertIsNotNone(window.right_selection)
            self.assertGreater(len(window.candidate_tree.get_children()), 0)
            self.assertIn("候选", window.scientific_status_var.get())
        finally:
            window.close()
            root.destroy()

    def test_titles_show_each_folder_and_reference_energy(self):
        root = tk.Tk()
        root.withdraw()
        window = PairedBandInspector(root, session(), query_detail=lambda *_: None)
        try:
            window.update()
            self.assertIn("left", window.left_axes.get_title())
            self.assertIn("E-fermi", window.left_axes.get_title())
            self.assertIn("right", window.right_axes.get_title())
        finally:
            window.close()
            root.destroy()


if __name__ == "__main__":
    unittest.main()

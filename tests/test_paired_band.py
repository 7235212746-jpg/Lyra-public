from pathlib import Path
import unittest
from dataclasses import replace

import numpy as np

from lyra.models import BandElementIndex, BandStateSelection, ParsedStructure
from lyra.paired_band import (
    assess_index_compatibility,
    nearest_periodic_kpoint,
    normalise_label,
    rank_projection_candidates,
)


def make_index(*, kpoints=None, labels=("G", "X"), energies=None, percentages=None):
    kpoints = np.asarray(
        kpoints if kpoints is not None else ((0, 0, 0), (0.5, 0, 0), (1, 0, 0)),
        dtype=float,
    )
    energies = np.asarray(
        energies if energies is not None else ((-1.0, 0.1), (-0.8, 0.2), (-0.6, 0.3)),
        dtype=float,
    )
    percentages = np.asarray(
        percentages
        if percentages is not None
        else (
            ((90, 80), (85, 75), (80, 70)),
            ((10, 20), (15, 25), (20, 30)),
        ),
        dtype=float,
    )
    raw_elements = percentages[None] / 100.0
    return BandElementIndex(
        spin_names=("total",),
        elements=("Ta", "Ga"),
        orbitals=("s", "d"),
        kpoints=kpoints,
        distances=np.linspace(0.0, 1.0, len(kpoints)),
        segment_ranges=((0, len(kpoints)),),
        tick_positions=(0.0, 1.0),
        tick_labels=labels,
        energies={"total": energies},
        reference_energy=5.0,
        raw_element_totals=raw_elements,
        raw_projection_totals=np.ones((1, len(kpoints), energies.shape[1])),
        relative_element_percentages=percentages[None],
    )


class PairedBandModelTests(unittest.TestCase):
    def test_normalises_gamma_aliases(self):
        self.assertEqual(normalise_label("Γ"), normalise_label("Gamma"))
        self.assertEqual(normalise_label("\\Gamma"), "G")

    def test_exact_path_enables_synchronised_selection(self):
        result = assess_index_compatibility(make_index(), make_index())

        self.assertTrue(result.path_compatible)
        self.assertTrue(result.sync_enabled)
        self.assertEqual(result.mapping, (0, 1, 2))

    def test_periodic_nearest_kpoint_maps_one_to_zero(self):
        target = np.array(((0.0, 0, 0), (0.4, 0, 0)))

        index, distance = nearest_periodic_kpoint(np.array((1.0, 0, 0)), target)

        self.assertEqual(index, 0)
        self.assertAlmostEqual(distance, 0.0)

    def test_different_sampling_remains_usable_as_approximate_mapping(self):
        left = make_index()
        right = make_index(
            kpoints=((0, 0, 0), (0.25, 0, 0), (0.5, 0, 0), (0.75, 0, 0)),
            energies=((-1, 0), (-0.9, 0.1), (-0.8, 0.2), (-0.7, 0.3)),
            percentages=(
                ((90, 80), (88, 78), (85, 75), (82, 72)),
                ((10, 20), (12, 22), (15, 25), (18, 28)),
            ),
        )

        result = assess_index_compatibility(left, right)

        self.assertFalse(result.path_compatible)
        self.assertFalse(result.sync_enabled)
        self.assertEqual(result.mapping[1], 2)

    def test_structure_mismatch_disables_exact_synchronisation(self):
        left_structure = ParsedStructure(
            np.eye(3), ("Ta",), np.array(((0.0, 0.0, 0.0),)), "left"
        )
        right_structure = ParsedStructure(
            np.eye(3), ("Ta",), np.array(((0.1, 0.0, 0.0),)), "right"
        )

        result = assess_index_compatibility(
            replace(make_index(), structure=left_structure),
            replace(make_index(), structure=right_structure),
        )

        self.assertFalse(result.structure_compatible)
        self.assertFalse(result.sync_enabled)

    def test_candidate_ranking_uses_projection_similarity_not_band_number(self):
        source = make_index(
            energies=((-1, 0.10), (-0.8, 0.20), (-0.6, 0.30)),
            percentages=(
                ((95, 10), (95, 10), (95, 10)),
                ((5, 90), (5, 90), (5, 90)),
            ),
        )
        target = make_index(
            energies=((-1, -0.99), (-0.8, -0.79), (-0.6, -0.59)),
            percentages=(
                ((10, 94), (10, 94), (10, 94)),
                ((90, 6), (90, 6), (90, 6)),
            ),
        )
        selection = BandStateSelection("total", 1, 0)

        candidates = rank_projection_candidates(source, target, selection, "total")

        self.assertEqual(candidates[0].band_index, 1)
        self.assertGreater(candidates[0].composition_similarity, 0.99)
        self.assertIn("candidate", candidates[0].evidence.lower())


if __name__ == "__main__":
    unittest.main()

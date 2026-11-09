from dataclasses import replace
import unittest

import numpy as np

from lyra.band_inspector import (
    band_state_detail,
    build_element_index,
    near_degenerate_bands,
    whole_band_detail,
)
from lyra.models import BandStateSelection, ParsedBandData, SpinKind
from lyra.projection_store import BandProjectionStore
from tests.test_projection_store import parsed_band_data


class BandInspectorTests(unittest.TestCase):
    def setUp(self):
        self.data = parsed_band_data()
        self.store = BandProjectionStore.from_parsed(self.data)

    def test_element_index_uses_all_elements_and_retains_raw_total(self):
        index = build_element_index(self.data, self.store)

        self.assertEqual(index.elements, ("A", "B"))
        self.assertEqual(index.raw_element_totals.shape, (2, 2, 2, 3))
        np.testing.assert_allclose(index.raw_element_totals[0, :, 0, 0], [15.0, 150.0])
        self.assertAlmostEqual(index.raw_projection_totals[0, 0, 0], 165.0)
        np.testing.assert_allclose(
            index.relative_element_percentages[0, :, 0, 0],
            [100.0 / 11.0, 1000.0 / 11.0],
        )

    def test_zero_projection_total_is_unavailable_not_fabricated_zero(self):
        zero_store = replace(
            self.store,
            tensor=np.zeros_like(self.store.tensor),
        )

        index = build_element_index(self.data, zero_store)

        self.assertTrue(np.isnan(index.relative_element_percentages).all())
        self.assertTrue((index.raw_projection_totals == 0.0).all())

    def test_state_detail_keeps_raw_and_relative_orbital_evidence(self):
        detail = band_state_detail(
            self.data,
            self.store,
            BandStateSelection("up", 0, 0),
        )

        self.assertEqual(detail.elements, ("A", "B"))
        self.assertEqual(detail.orbitals, self.store.orbitals)
        self.assertAlmostEqual(detail.raw_projection_total, 165.0)
        np.testing.assert_allclose(detail.raw_orbital_weights[0], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(float(np.nansum(detail.relative_orbital_percentages)), 100.0)
        self.assertIsNone(detail.spin_components)

    def test_soc_state_detail_exposes_signed_components_without_charge_mixing(self):
        charge = np.ascontiguousarray(self.store.tensor[:1])
        components = np.ascontiguousarray(
            np.stack([charge[0] * 0.1, charge[0] * -0.2, charge[0] * 0.3])
        )
        soc_store = BandProjectionStore(
            ("total",),
            self.store.elements,
            self.store.orbitals,
            charge,
            spin_components=components,
        )
        soc_data = replace(
            self.data,
            energies={"total": self.data.energies["up"]},
            spin_kind=SpinKind.SOC,
        )

        detail = band_state_detail(
            soc_data,
            soc_store,
            BandStateSelection("total", 0, 0),
        )

        np.testing.assert_allclose(
            detail.spin_components[:, 0, 0],
            [0.1, -0.2, 0.3],
        )
        self.assertAlmostEqual(detail.raw_projection_total, 165.0)

    def test_whole_band_summary_weights_segments_without_bridging_jump(self):
        data = ParsedBandData(
            folder=self.data.folder,
            kpoints=np.zeros((4, 3)),
            distances=np.array([0.0, 1.0, 1.0, 3.0]),
            segment_ranges=((0, 2), (2, 4)),
            tick_positions=(0.0, 1.0, 3.0),
            tick_labels=("A", "B|C", "D"),
            energies={"total": np.zeros((4, 1))},
            efermi=0.0,
            spin_kind=SpinKind.NON_SPIN,
            elements=("X",),
            orbitals=("s",),
        )
        tensor = np.ascontiguousarray(
            np.array([1.0, 3.0, 100.0, 200.0], dtype=np.float32).reshape(
                1, 1, 1, 4, 1
            )
        )
        store = BandProjectionStore(("total",), ("X",), ("s",), tensor)

        detail = whole_band_detail(
            data,
            store,
            BandStateSelection("total", 0, 0),
        )

        self.assertAlmostEqual(detail.raw_orbital_weights[0, 0], 302.0 / 3.0)
        self.assertAlmostEqual(detail.relative_orbital_percentages[0, 0], 100.0)

    def test_near_degenerate_bands_uses_energy_at_selected_kpoint(self):
        energies = self.data.energies["up"].copy()
        energies[0] = [0.0, 0.003, 0.020]
        data = replace(self.data, energies={**self.data.energies, "up": energies})
        index = build_element_index(data, self.store)

        members = near_degenerate_bands(
            index,
            BandStateSelection("up", 0, 1),
        )

        self.assertEqual(members, (0, 1))


if __name__ == "__main__":
    unittest.main()

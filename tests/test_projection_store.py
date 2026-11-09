from __future__ import annotations

from pathlib import Path
import unittest
from dataclasses import replace

import numpy as np

from lyra.models import ParsedBandData, ParsedDosData, ProjectionChannel, SpinKind
from lyra.projection_store import BandProjectionStore, DosProjectionStore


def parsed_band_data() -> ParsedBandData:
    shape = (2, 3)
    factors = {"up": 1.0, "down": 2.0}
    values = {
        "A:s": 1.0,
        "A:px": 2.0,
        "A:py": 3.0,
        "A:dxy": 4.0,
        "A:dz2": 5.0,
        "B:s": 10.0,
        "B:px": 20.0,
        "B:py": 30.0,
        "B:dxy": 40.0,
        "B:dz2": 50.0,
    }
    projections = {
        key: {
            spin: np.full(shape, value * factors[spin])
            for spin in factors
        }
        for key, value in values.items()
    }
    return ParsedBandData(
        folder=Path("calculation"),
        kpoints=np.zeros((2, 3)),
        distances=np.array([0.0, 1.0]),
        segment_ranges=((0, 2),),
        tick_positions=(0.0, 1.0),
        tick_labels=("G", "X"),
        energies={spin: np.zeros(shape) for spin in factors},
        efermi=5.0,
        spin_kind=SpinKind.COLLINEAR,
        elements=("A", "B"),
        projections=projections,
        orbitals=("s", "px", "py", "dxy", "dz2"),
    )


def parsed_dos_data() -> ParsedDosData:
    factors = {"up": 1.0, "down": 2.0}
    values = {
        "A:s": 1.0,
        "A:px": 2.0,
        "A:py": 3.0,
        "A:dxy": 4.0,
        "A:dz2": 5.0,
        "B:s": 10.0,
        "B:px": 20.0,
        "B:py": 30.0,
        "B:dxy": 40.0,
        "B:dz2": 50.0,
    }
    projections = {
        key: {
            spin: np.full(4, value * factors[spin])
            for spin in factors
        }
        for key, value in values.items()
    }
    return ParsedDosData(
        folder=Path("calculation"),
        energies=np.linspace(3.0, 6.0, 4),
        densities={spin: np.full(4, factor) for spin, factor in factors.items()},
        efermi=5.0,
        spin_kind=SpinKind.COLLINEAR,
        elements=("A", "B"),
        projections=projections,
        orbitals=("s", "px", "py", "dxy", "dz2"),
    )


class BandProjectionStoreTests(unittest.TestCase):
    def test_materialize_derives_element_grouped_and_resolved_channels(self):
        base_data = parsed_band_data()
        store = BandProjectionStore.from_parsed(base_data)
        tensor_before = store.tensor.copy()

        materialized = store.materialize(
            base_data,
            (
                ProjectionChannel("A"),
                ProjectionChannel("A", "p"),
                ProjectionChannel("A", "d"),
                ProjectionChannel("A", "px"),
                ProjectionChannel("B", "dxy"),
            ),
        )

        self.assertEqual(store.spin_names, ("up", "down"))
        self.assertEqual(store.elements, ("A", "B"))
        self.assertEqual(store.orbitals, ("s", "px", "py", "dxy", "dz2"))
        self.assertEqual(store.tensor.dtype, np.float32)
        self.assertTrue(store.tensor.flags.c_contiguous)
        np.testing.assert_array_equal(materialized.projections["A"]["up"], np.full((2, 3), 15.0))
        np.testing.assert_array_equal(materialized.projections["A"]["down"], np.full((2, 3), 30.0))
        np.testing.assert_array_equal(materialized.projections["A:p"]["up"], np.full((2, 3), 5.0))
        np.testing.assert_array_equal(materialized.projections["A:d"]["down"], np.full((2, 3), 18.0))
        np.testing.assert_array_equal(materialized.projections["A:px"]["up"], np.full((2, 3), 2.0))
        np.testing.assert_array_equal(materialized.projections["B:dxy"]["down"], np.full((2, 3), 80.0))
        np.testing.assert_array_equal(
            materialized.projection_total["up"], np.full((2, 3), 165.0)
        )
        np.testing.assert_array_equal(
            materialized.projection_total["down"], np.full((2, 3), 330.0)
        )
        np.testing.assert_array_equal(store.tensor, tensor_before)
        self.assertEqual(base_data.projections["A:s"]["up"].dtype, np.float64)

    def test_materialize_accepts_dx2_y2_alias_for_pymatgen_dx2_axis(self):
        data = parsed_band_data()
        projections = dict(data.projections)
        for element in data.elements:
            projections[f"{element}:dx2"] = projections.pop(f"{element}:dz2")
        data = replace(
            data,
            projections=projections,
            orbitals=("s", "px", "py", "dxy", "dx2"),
        )
        store = BandProjectionStore.from_parsed(data)

        materialized = store.materialize(
            replace(data, projections={}),
            (ProjectionChannel("A", "dx2-y2"),),
        )

        np.testing.assert_array_equal(
            materialized.projections["A:dx2-y2"]["up"],
            np.full((2, 3), 5.0),
        )


class DosProjectionStoreTests(unittest.TestCase):
    def test_materialize_derives_selected_dos_channels_without_mutating_store(self):
        base_data = parsed_dos_data()
        store = DosProjectionStore.from_parsed(base_data)
        tensor_before = store.tensor.copy()

        materialized = store.materialize(
            base_data,
            (
                ProjectionChannel("A"),
                ProjectionChannel("A", "p"),
                ProjectionChannel("A", "px"),
                ProjectionChannel("B", "dxy"),
            ),
        )

        self.assertEqual(store.spin_names, ("up", "down"))
        self.assertEqual(store.tensor.dtype, np.float32)
        self.assertTrue(store.tensor.flags.c_contiguous)
        self.assertEqual(set(materialized.projections), {"A", "A:p", "A:px", "B:dxy"})
        np.testing.assert_array_equal(materialized.projections["A"]["up"], np.full(4, 15.0))
        np.testing.assert_array_equal(materialized.projections["A:p"]["down"], np.full(4, 10.0))
        np.testing.assert_array_equal(materialized.projections["B:dxy"]["up"], np.full(4, 40.0))
        np.testing.assert_array_equal(store.tensor, tensor_before)
        self.assertEqual(base_data.projections["A:s"]["up"].dtype, np.float64)

    def test_materialize_accepts_dx2_y2_alias_for_full_dos_axis(self):
        data = parsed_dos_data()
        projections = dict(data.projections)
        for element in data.elements:
            projections[f"{element}:dx2"] = projections.pop(f"{element}:dz2")
        data = replace(
            data,
            projections=projections,
            orbitals=("s", "px", "py", "dxy", "dx2"),
        )
        store = DosProjectionStore.from_parsed(data)

        materialized = store.materialize(
            replace(data, projections={}),
            (ProjectionChannel("A", "dx2-y2"),),
        )

        np.testing.assert_array_equal(
            materialized.projections["A:dx2-y2"]["up"],
            np.full(4, 5.0),
        )


if __name__ == "__main__":
    unittest.main()

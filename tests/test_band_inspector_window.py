import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from matplotlib.collections import LineCollection

from lyra.band_inspector_window import (
    BandProjectionInspector,
    element_percentage_rows,
    nearest_band_state,
    state_detail_rows,
)
from lyra.models import (
    BandElementIndex,
    BandPlotSettings,
    BandStateDetail,
    BandStateSelection,
    ParsedBandData,
    PlotRange,
    SpinKind,
)


def make_index() -> BandElementIndex:
    energies = np.array(
        [
            [-1.0, 0.2, 1.0],
            [-0.8, 0.1, 1.2],
            [-0.6, 0.0, 1.4],
        ]
    )
    raw_elements = np.array(
        [
            [
                [[0.8, 0.3, 0.1], [0.6, 0.4, 0.2], [0.4, 0.5, 0.3]],
                [[0.2, 0.7, 0.9], [0.4, 0.6, 0.8], [0.6, 0.5, 0.7]],
            ]
        ],
        dtype=float,
    )
    raw_totals = raw_elements.sum(axis=1)
    relative = raw_elements * 100.0 / raw_totals[:, None, :, :]
    return BandElementIndex(
        spin_names=("total",),
        elements=("Ta", "Ga"),
        orbitals=("s", "p"),
        kpoints=np.zeros((3, 3)),
        distances=np.array([0.0, 0.5, 1.0]),
        segment_ranges=((0, 3),),
        tick_positions=(0.0, 1.0),
        tick_labels=("G", "X"),
        energies={"total": energies},
        reference_energy=5.0,
        raw_element_totals=raw_elements,
        raw_projection_totals=raw_totals,
        relative_element_percentages=relative,
    )


class BandInspectorPresentationTests(unittest.TestCase):
    def test_nearest_state_uses_actual_kpoint_and_energy(self):
        index = make_index()

        selection = nearest_band_state(index, 0.48, 0.15, "total")

        self.assertEqual(selection, BandStateSelection("total", 1, 1))

    def test_nearest_state_resolves_duplicate_path_distance_by_energy(self):
        index = make_index()
        index.distances[2] = index.distances[1]

        selection = nearest_band_state(index, 0.5, 1.35, "total")

        self.assertEqual(selection, BandStateSelection("total", 2, 2))

    def test_overview_rows_keep_poscar_element_order(self):
        rows = element_percentage_rows(make_index(), "total", 1, (-0.9, 0.3))

        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][2], ("60.0", "40.0"))
        self.assertEqual(rows[1][0], 2)
        self.assertEqual(rows[1][2], ("40.0", "60.0"))

    def test_detail_rows_label_resolved_percentages_and_signed_soc(self):
        detail = BandStateDetail(
            selection=BandStateSelection("total", 1, 1),
            elements=("Ta", "Ga"),
            orbitals=("s", "p"),
            band_indices=(1,),
            raw_orbital_weights=np.array([[0.2, 0.3], [0.1, 0.4]]),
            relative_orbital_percentages=np.array([[20.0, 30.0], [10.0, 40.0]]),
            raw_projection_total=1.0,
            spin_components=np.array(
                [
                    [[0.01, 0.02], [-0.01, 0.00]],
                    [[0.03, 0.00], [0.01, -0.02]],
                    [[0.04, 0.01], [-0.02, -0.01]],
                ]
            ),
        )

        rows = state_detail_rows(detail)

        self.assertEqual(rows[0], ("元素合计", ("50.0", "50.0")))
        self.assertIn(("Mx / 投影总量", ("+3.0", "-1.0")), rows)
        self.assertIn(("Mz / 投影总量", ("+5.0", "-3.0")), rows)


class BandInspectorWindowTests(unittest.TestCase):
    def test_window_requests_selected_band_and_exposes_element_columns(self):
        root = tk.Tk()
        root.withdraw()
        request_detail = Mock()
        window = BandProjectionInspector(root, make_index(), request_detail)
        try:
            window.update()
            self.assertEqual(
                tuple(window.overview.heading(column, "text") for column in ("Ta", "Ga")),
                ("Ta", "Ga"),
            )

            window.select_state(BandStateSelection("total", 1, 1))

            request_detail.assert_called_once()
            selection, whole_band, band_indices = request_detail.call_args.args
            self.assertEqual(selection, BandStateSelection("total", 1, 1))
            self.assertFalse(whole_band)
            self.assertEqual(band_indices, (1,))
        finally:
            window.destroy()
            root.destroy()

    def test_window_uses_current_band_style_and_highlights_clicked_state(self):
        index = make_index()
        data = ParsedBandData(
            folder=Path("."),
            kpoints=index.kpoints,
            distances=index.distances,
            segment_ranges=index.segment_ranges,
            tick_positions=index.tick_positions,
            tick_labels=index.tick_labels,
            energies={"total": index.energies["total"] + index.reference_energy},
            efermi=index.reference_energy,
            spin_kind=SpinKind.NON_SPIN,
            elements=index.elements,
        )
        settings = BandPlotSettings(
            energy_range=PlotRange(-1.5, 1.5),
            complete_band_colour="#123456",
        )
        root = tk.Tk()
        root.withdraw()
        window = BandProjectionInspector(
            root,
            index,
            Mock(),
            display_data=data,
            plot_settings=settings,
        )
        try:
            window.select_state(BandStateSelection("total", 1, 1))

            self.assertEqual(window.axes.collections[0]._lyra_layer, "complete-band")
            self.assertGreaterEqual(len(window._selection_artists), 2)
            self.assertEqual(window.navigation.winfo_manager(), "pack")
        finally:
            window.destroy()
            root.destroy()

    def test_spin_selector_draws_only_the_spin_being_analysed(self):
        base = make_index()
        index = replace(
            base,
            spin_names=("up", "down"),
            energies={
                "up": base.energies["total"],
                "down": base.energies["total"] + 0.4,
            },
            raw_element_totals=np.repeat(base.raw_element_totals, 2, axis=0),
            raw_projection_totals=np.repeat(base.raw_projection_totals, 2, axis=0),
            relative_element_percentages=np.repeat(
                base.relative_element_percentages, 2, axis=0
            ),
        )
        data = ParsedBandData(
            folder=Path("."),
            kpoints=index.kpoints,
            distances=index.distances,
            segment_ranges=index.segment_ranges,
            tick_positions=index.tick_positions,
            tick_labels=index.tick_labels,
            energies={
                "up": index.energies["up"] + index.reference_energy,
                "down": index.energies["down"] + index.reference_energy,
            },
            efermi=index.reference_energy,
            spin_kind=SpinKind.COLLINEAR,
            elements=index.elements,
        )
        root = tk.Tk()
        root.withdraw()
        window = BandProjectionInspector(
            root,
            index,
            Mock(),
            display_data=data,
            plot_settings=BandPlotSettings(energy_range=PlotRange(-1.5, 2.0)),
        )
        try:
            collections = [
                item for item in window.axes.collections
                if isinstance(item, LineCollection)
            ]
            self.assertEqual(len(collections), 1)
            self.assertEqual(window.spin_var.get(), "up")

            window.spin_var.set("down")
            window._spin_changed()
            collections = [
                item for item in window.axes.collections
                if isinstance(item, LineCollection)
            ]
            self.assertEqual(len(collections), 1)
        finally:
            window.destroy()
            root.destroy()


if __name__ == "__main__":
    unittest.main()

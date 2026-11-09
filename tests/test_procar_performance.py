from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lyra.adapters.eigenval import EigenvalAdapter
from lyra.adapters.procar import ProcarAdapter
from tests.fixtures import (
    write_band_fixture,
    write_eigenval,
    write_kpoints,
    write_outcar,
    write_poscar,
    write_procar,
)


class ProcarPerformanceTests(unittest.TestCase):
    def test_load_store_uses_one_bulk_ion_conversion_per_projection_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder, elements=("A", "B"), counts=(1, 2))
            write_outcar(folder)
            write_kpoints(folder)
            write_eigenval(folder)
            write_procar(folder, nions=3)
            band_data = EigenvalAdapter.load(folder)

            with patch("lyra.adapters.procar.np.fromstring", wraps=np.fromstring) as fromstring:
                ProcarAdapter.load_store(folder, band_data)

        self.assertEqual(len(fromstring.call_args_list), 36)
        self.assertEqual(
            [len(call.args[0].split()) for call in fromstring.call_args_list],
            [3 * (9 + 1), 9 + 1] * 18,
        )
        self.assertTrue(
            all(call.kwargs == {"sep": " ", "dtype": np.float32} for call in fromstring.call_args_list)
        )

    def test_load_store_streams_without_path_read_text_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            band_data = EigenvalAdapter.load(folder)
            events = []

            with patch.object(Path, "read_text", side_effect=AssertionError("read_text used")):
                ProcarAdapter.load_store(folder, band_data, progress=events.append)

        self.assertTrue(events)
        self.assertTrue(all(set(event) == {"phase", "current", "total", "detail"} for event in events))
        parse_events = [event for event in events if event["phase"] == "parse"]
        self.assertTrue(parse_events)
        self.assertEqual(parse_events[0]["current"], 0)
        self.assertEqual(parse_events[0]["total"], 18)
        self.assertEqual(parse_events[-1]["current"], 18)
        self.assertEqual(parse_events[-1]["total"], 18)
        self.assertIn("k-point 9/9", parse_events[-1]["detail"])
        self.assertEqual(events[-1]["phase"], "complete")
        self.assertEqual(events[-1]["current"], 18)
        self.assertEqual(events[-1]["total"], 18)

    def test_progress_throttling_caps_callbacks_for_309_600_tables(self):
        total = 309_600

        stride = ProcarAdapter.progress_stride(total)

        self.assertGreaterEqual(stride, 1)
        self.assertLessEqual((total + stride - 1) // stride + 3, 103)

    def test_progress_callback_uses_stride_for_a_manageable_table_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder)
            write_eigenval(folder, nbands=12)
            write_procar(folder, nbands=12)
            events = []

            ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder), events.append)

        parse_events = [event for event in events if event["phase"] == "parse"]
        currents = [event["current"] for event in parse_events]
        self.assertEqual(currents[0], 0)
        self.assertEqual(currents[-1], 108)
        self.assertEqual(set(event["total"] for event in parse_events), {108})
        self.assertEqual(currents[1:], list(range(2, 109, 2)))


if __name__ == "__main__":
    unittest.main()

import base64
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from lyra.adapters.eigenval import EigenvalAdapter
from lyra.adapters.procar import ProcarAdapter
from tests.fixtures import (
    band_fixture_energy,
    band_fixture_kpoints,
    write_band_fixture,
    write_eigenval,
    write_kpoints,
    write_outcar,
    write_poscar,
    write_procar,
)


def write_component_procar(
    folder: Path,
    *,
    component_count: int,
    nkpts: int = 9,
    nbands: int = 2,
    nions: int = 1,
    header_energy_overrides: dict[tuple[int, int, int], float] | None = None,
) -> None:
    orbitals = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2")
    header_energy_overrides = header_energy_overrides or {}
    lines = [
        "PROCAR lm decomposed",
        f" # of k-points: {nkpts}         # of bands: {nbands}         # of ions: {nions}",
        "",
    ]
    for component_index in range(component_count):
        lines.extend([f" spin component {component_index + 1}", ""])
        for kpoint_index in range(nkpts):
            point = band_fixture_kpoints(nkpts)[kpoint_index]
            lines.extend([
                f" k-point {kpoint_index + 1} : {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} weight = 0.111111",
                "",
            ])
            for band_index in range(nbands):
                energy = header_energy_overrides.get(
                    (component_index, kpoint_index, band_index),
                    band_fixture_energy(kpoint_index, band_index),
                )
                lines.extend([
                    f" band {band_index + 1} # energy {energy:.6f} # occ. 1.0",
                    "",
                    " ion " + " ".join(orbitals) + " tot",
                ])
                totals = [0.0] * len(orbitals)
                for ion_index in range(nions):
                    base = 0.01 * (component_index + 1) * (band_index + 1) * (ion_index + 1)
                    values = [base * (orbital_index + 1) for orbital_index in range(len(orbitals))]
                    totals = [left + right for left, right in zip(totals, values)]
                    lines.append(
                        f" {ion_index + 1} "
                        + " ".join(f"{value:.6f}" for value in values)
                        + f" {sum(values):.6f}"
                    )
                lines.append(
                    " tot "
                    + " ".join(f"{value:.6f}" for value in totals)
                    + f" {sum(totals):.6f}"
                )
                lines.append("")
    (folder / "PROCAR").write_text("\n".join(lines), encoding="utf-8")


def write_interleaved_soc_procar(
    folder: Path,
    *,
    nkpts: int = 9,
    nbands: int = 2,
    nions: int = 1,
    header_energy_overrides: dict[tuple[int, int], float] | None = None,
) -> None:
    """Write the noncollinear layout documented by the VASP PROCAR reference."""
    orbitals = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2")
    header_energy_overrides = header_energy_overrides or {}
    lines = [
        "PROCAR lm decomposed",
        f" # of k-points: {nkpts}         # of bands: {nbands}         # of ions: {nions}",
        "",
    ]
    for kpoint_index in range(nkpts):
        point = band_fixture_kpoints(nkpts)[kpoint_index]
        lines.extend(
            [
                f" k-point {kpoint_index + 1} : {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} weight = 0.111111",
                "",
            ]
        )
        for band_index in range(nbands):
            energy = header_energy_overrides.get(
                (kpoint_index, band_index),
                band_fixture_energy(kpoint_index, band_index),
            )
            lines.extend(
                [
                    f" band {band_index + 1} # energy {energy:.6f} # occ. 1.0",
                    "",
                    " ion " + " ".join(orbitals) + " tot",
                ]
            )
            for component_index in range(4):
                totals = [0.0] * len(orbitals)
                for ion_index in range(nions):
                    base = (
                        0.01
                        * (component_index + 1)
                        * (band_index + 1)
                        * (ion_index + 1)
                    )
                    values = [
                        base * (orbital_index + 1)
                        for orbital_index in range(len(orbitals))
                    ]
                    totals = [
                        left + right for left, right in zip(totals, values)
                    ]
                    lines.append(
                        f" {ion_index + 1} "
                        + " ".join(f"{value:.6f}" for value in values)
                        + f" {sum(values):.6f}"
                    )
                lines.append(
                    " tot "
                    + " ".join(f"{value:.6f}" for value in totals)
                    + f" {sum(totals):.6f}"
                )
            lines.append("")
    (folder / "PROCAR").write_text("\n".join(lines), encoding="utf-8")


def remove_line_after(lines: list[str], prefix: str, target: str) -> None:
    start = next(index for index, line in enumerate(lines) if line.startswith(prefix))
    target_index = next(
        index for index, line in enumerate(lines[start + 1 :], start + 1) if line.startswith(target)
    )
    del lines[target_index]


class ProcarAdapterTests(unittest.TestCase):
    _BASELINE_TENSOR = (
        "c$}rrAr`_w3<S{C7&ID%jzMgO3)Ix3w<o}JgVkHxBVf?gspbZ#wfIJa6O73FQS)Z{_|`sN(MdNObTd=d!)"
        "Nh3gSD;?t?N%&FZb$q1gm%YZzElD{FJ^e$3{xe%8^Mql`p1JZmcOiky4^HGnfsub8DE}iB{wV5ux=?h;F2&@l*OX#70Wb(#WLT0gq)x+5"
    )

    @classmethod
    def _baseline_tensor_bytes(cls) -> bytes:
        return zlib.decompress(base64.b85decode(cls._BASELINE_TENSOR))

    @staticmethod
    def _write_multi_element_fixture(folder: Path) -> None:
        write_poscar(folder, elements=("A", "B"), counts=(1, 2))
        write_outcar(folder)
        write_kpoints(folder)
        write_eigenval(folder)
        write_procar(folder, nions=3)

    @staticmethod
    def _write_soc_multi_element_fixture(folder: Path) -> None:
        write_poscar(folder, elements=("A", "B"), counts=(1, 2))
        write_outcar(folder, lsorbit=True, noncollinear=True)
        write_kpoints(folder)
        write_eigenval(folder)
        write_component_procar(folder, component_count=4, nions=3)

    @staticmethod
    def _expected_component_events(total: int, component_total: int) -> list[dict]:
        tables_per_component = total // component_total
        return [
            {"phase": "metadata", "current": 0, "total": 1, "detail": "Reading PROCAR metadata."},
            {
                "phase": "metadata",
                "current": 1,
                "total": 1,
                "detail": f"Prepared {total} PROCAR projection tables for 9 orbitals.",
            },
            {"phase": "parse", "current": 0, "total": total, "detail": "Parsing PROCAR projection tables."},
            *[
                {
                    "phase": "parse",
                    "current": current,
                    "total": total,
                    "detail": (
                        f"Parsed {current}/{total} projection tables (component "
                        f"{(current - 1) // tables_per_component + 1}/{component_total}, "
                        f"k-point {((current - 1) % tables_per_component) // 2 + 1}/9, "
                        f"band {(current - 1) % 2 + 1}/2)."
                    ),
                }
                for current in range(1, total + 1)
            ],
            {
                "phase": "complete",
                "current": total,
                "total": total,
                "detail": "Finished parsing PROCAR projection tables.",
            },
        ]

    def test_load_store_preserves_captured_multi_element_tensor_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_multi_element_fixture(folder)
            events = []
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder), events.append)

        self.assertEqual(store.tensor.dtype, np.dtype(np.float32))
        self.assertEqual(store.tensor.tobytes(), self._baseline_tensor_bytes())
        self.assertEqual(
            events,
            [
                {"phase": "metadata", "current": 0, "total": 1, "detail": "Reading PROCAR metadata."},
                {
                    "phase": "metadata",
                    "current": 1,
                    "total": 1,
                    "detail": "Prepared 18 PROCAR projection tables for 9 orbitals.",
                },
                {"phase": "parse", "current": 0, "total": 18, "detail": "Parsing PROCAR projection tables."},
                *[
                    {
                        "phase": "parse",
                        "current": current,
                        "total": 18,
                        "detail": (
                            f"Parsed {current}/18 projection tables (component 1/1, "
                            f"k-point {(current - 1) // 2 + 1}/9, band {(current - 1) % 2 + 1}/2)."
                        ),
                    }
                    for current in range(1, 19)
                ],
                {
                    "phase": "complete",
                    "current": 18,
                    "total": 18,
                    "detail": "Finished parsing PROCAR projection tables.",
                },
            ],
        )

    def test_load_store_preserves_out_of_order_ion_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_multi_element_fixture(folder)
            procar = folder / "PROCAR"
            lines = procar.read_text(encoding="utf-8").splitlines()
            header = next(index for index, line in enumerate(lines) if line.startswith(" ion "))
            rows = lines[header + 1 : header + 4]
            lines[header + 1 : header + 4] = (rows[2], rows[0], rows[1])
            procar.write_text("\n".join(lines), encoding="utf-8")
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        self.assertEqual(store.tensor.tobytes(), self._baseline_tensor_bytes())

    def test_load_store_preserves_captured_soc_multi_element_tensor_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_soc_multi_element_fixture(folder)
            events = []
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder), events.append)

        self.assertEqual(store.tensor.shape, (1, 2, 9, 9, 2))
        self.assertEqual(store.tensor.tobytes(), self._baseline_tensor_bytes())
        self.assertEqual(events, self._expected_component_events(total=72, component_total=4))

    def test_read_ion_table_preserves_legacy_nan_and_inf_values(self):
        for literal, check in (("nan", np.isnan), ("inf", np.isposinf)):
            with self.subTest(literal=literal):
                values = ProcarAdapter._read_ion_table(
                    iter([f"1 {literal} 0", f"tot {literal} 0"]), nions=1, orbital_count=1
                )

                self.assertTrue(check(values[0, 0]))

    def test_read_ion_table_preserves_legacy_partial_orbital_token_error(self):
        with self.assertRaisesRegex(RuntimeError, "Malformed ion projection row") as raised:
            ProcarAdapter._read_ion_table(
                iter(["1 1.0tail 1.0", "tot 1.0 1.0"]), nions=1, orbital_count=1
            )

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_read_ion_table_preserves_legacy_partial_final_total_behavior(self):
        values = ProcarAdapter._read_ion_table(
            iter(["1 1.0 1.0tail", "tot 1.0 1.0"]), nions=1, orbital_count=1
        )

        np.testing.assert_array_equal(values, np.array([[1.0]], dtype=np.float32))

    def test_read_ion_table_preserves_legacy_mixed_error_precedence_and_chaining(self):
        cases = (
            ("numeric before later duplicate", ["1 bad 0", "1 1.0 1.0", "tot 1.0 1.0"], ValueError),
            ("numeric after duplicate ID", ["1 1.0 1.0", "1 bad 0", "tot 1.0 1.0"], ValueError),
            ("numeric before later out of range", ["1 bad 0", "3 1.0 1.0", "tot 1.0 1.0"], ValueError),
            ("out of range before numeric", ["3 1.0 1.0", "1 bad 0", "tot 1.0 1.0"], None),
        )
        for name, lines, cause_type in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "Malformed ion projection row") as raised:
                    ProcarAdapter._read_ion_table(iter(lines), nions=2, orbital_count=1)

                if cause_type is None:
                    self.assertIsNone(raised.exception.__cause__)
                else:
                    self.assertIsInstance(raised.exception.__cause__, cause_type)

    def test_load_store_rejects_duplicate_ion_ids_with_existing_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_multi_element_fixture(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(
                    "\n 2 0.020000", "\n 1 0.020000", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Malformed ion projection row"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_out_of_range_ion_ids_with_existing_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write_multi_element_fixture(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(
                    "\n 3 0.030000", "\n 4 0.030000", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Malformed ion projection row"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_preserves_legacy_partial_ion_row_total_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(
                    " 0.450000\n tot", " 0.450000tail\n tot", 1
                ),
                encoding="utf-8",
            )

            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        self.assertAlmostEqual(store.tensor[0, 0, store.orbitals.index("px"), 0, 0], 0.04)

    def test_load_store_aggregates_resolved_orbitals_into_float32_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        self.assertEqual(store.spin_names, ("total",))
        self.assertEqual(store.elements, ("X",))
        self.assertEqual(
            store.orbitals,
            ("dx2-y2", "dxy", "dxz", "dyz", "dz2", "px", "py", "pz", "s"),
        )
        self.assertEqual(store.tensor.dtype, np.dtype(np.float32))
        self.assertTrue(store.tensor.flags.c_contiguous)
        self.assertAlmostEqual(store.tensor[0, 0, store.orbitals.index("px"), 0, 0], 0.04)
        self.assertAlmostEqual(
            np.sum(store.tensor[0, 0, [store.orbitals.index(name) for name in ("px", "py", "pz")], 0, 0]),
            0.09,
        )
        self.assertAlmostEqual(
            np.sum(
                store.tensor[
                    0,
                    0,
                    [store.orbitals.index(name) for name in ("dxy", "dyz", "dz2", "dxz", "dx2-y2")],
                    0,
                    0,
                ]
            ),
            0.35,
        )

    def test_aggregates_resolved_and_grouped_orbitals_by_element(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            data = ProcarAdapter.attach(folder, EigenvalAdapter.load(folder))

        self.assertIn("X", data.projections)
        self.assertIn("X:p", data.projections)
        self.assertIn("X:px", data.projections)
        self.assertIn("X:d", data.projections)
        self.assertIn("X:dxy", data.projections)
        self.assertIn("X:dx2-y2", data.projections)
        self.assertEqual(
            data.orbitals,
            ("dx2-y2", "dxy", "dxz", "dyz", "dz2", "px", "py", "pz", "s"),
        )
        self.assertAlmostEqual(data.projections["X:px"]["total"][0, 0], 0.04)
        self.assertAlmostEqual(data.projections["X:p"]["total"][0, 0], 0.09)
        self.assertAlmostEqual(data.projections["X:d"]["total"][0, 0], 0.35)

    def test_preserves_two_spin_projection_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)
            write_procar(folder, ispin=2)
            data = ProcarAdapter.attach(folder, EigenvalAdapter.load(folder))

        self.assertEqual(tuple(data.projections["X:p"]), ("up", "down"))
        self.assertAlmostEqual(
            data.projections["X:p"]["down"][0, 0],
            2 * data.projections["X:p"]["up"][0, 0],
        )

    def test_load_store_preserves_two_spin_projection_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)
            write_procar(folder, ispin=2)
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        p_orbitals = [store.orbitals.index(name) for name in ("px", "py", "pz")]
        self.assertEqual(store.spin_names, ("up", "down"))
        self.assertAlmostEqual(
            np.sum(store.tensor[1, 0, p_orbitals, 0, 0]),
            2 * np.sum(store.tensor[0, 0, p_orbitals, 0, 0]),
        )

    def test_rejects_procar_dimensions_that_do_not_match_eigenval(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder, nkpts=8)

            with self.assertRaisesRegex(RuntimeError, "PROCAR dimensions"):
                ProcarAdapter.attach(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_kpoint_coordinates_that_do_not_match_eigenval(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder, kpoint_overrides={1: (0.0, 0.0, 0.75)})

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 2.*PROCAR \(0.0, 0.0, 0.75\).*EIGENVAL \(0.0, 0.0, 0.5\)",
            ):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_accepts_periodically_equivalent_kpoint_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            eigenval = folder / "EIGENVAL"
            eigenval.write_text(
                eigenval.read_text(encoding="utf-8").replace(
                    "0.0 0.0 1.0 0.0",
                    "0.0 0.0 0.0 0.0",
                    1,
                ),
                encoding="utf-8",
            )
            write_procar(folder)

            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        self.assertEqual(store.tensor.shape[-2:], (9, 2))

    def test_load_store_rejects_band_energy_that_does_not_match_eigenval(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder, energy_overrides={(0, 1, 0): 4.5})

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 2, band 1.*PROCAR 4.500000 eV.*EIGENVAL 4.050000 eV",
            ):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_down_channel_energy_with_channel_and_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)
            write_procar(
                folder,
                ispin=2,
                energy_overrides={(1, 1, 0): 4.5},
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 2, band 1.*spin channel down.*PROCAR component 2 \(down\)"
                r".*PROCAR 4.500000 eV.*EIGENVAL 4.250000 eV",
            ):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_malformed_ion_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(
                    " 1 0.010000 0.020000 0.030000",
                    " 1 malformed",
                    1,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Malformed ion projection row"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_missing_spin_channel_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)
            write_procar(folder, ispin=2)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace("spin component 2", "spin component 1"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "projection table"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_accepts_soc_components_and_keeps_charge_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True, noncollinear=True)
            write_kpoints(folder)
            write_eigenval(folder)
            write_component_procar(folder, component_count=4)
            events = []
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder), events.append)

        self.assertEqual(store.spin_names, ("total",))
        self.assertAlmostEqual(store.tensor[0, 0, store.orbitals.index("px"), 0, 0], 0.04)
        self.assertEqual(store.spin_components.shape, (3, 1, 9, 9, 2))
        np.testing.assert_allclose(
            store.spin_components[:, 0, store.orbitals.index("px"), 0, 0],
            [0.08, 0.12, 0.16],
        )
        self.assertEqual(events[-1]["current"], 72)
        self.assertEqual(events[-1]["total"], 72)

    def test_load_store_rejects_explicit_soc_non_total_component_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True, noncollinear=True)
            write_kpoints(folder)
            write_eigenval(folder)
            write_component_procar(
                folder,
                component_count=4,
                header_energy_overrides={(1, 1, 0): 4.5},
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 2, band 1.*spin channel total.*PROCAR component 2 \(m_x\)"
                r".*PROCAR 4.500000 eV.*EIGENVAL 4.050000 eV",
            ):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_accepts_official_interleaved_soc_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True, noncollinear=True)
            write_kpoints(folder)
            write_eigenval(folder)
            write_interleaved_soc_procar(folder)
            events = []

            store = ProcarAdapter.load_store(
                folder,
                EigenvalAdapter.load(folder),
                events.append,
            )

        px_index = store.orbitals.index("px")
        self.assertAlmostEqual(store.tensor[0, 0, px_index, 0, 0], 0.04)
        np.testing.assert_allclose(
            store.spin_components[:, 0, px_index, 0, 0],
            [0.08, 0.12, 0.16],
        )
        self.assertEqual(events[-1]["current"], 72)
        self.assertEqual(events[-1]["total"], 72)

    def test_load_store_rejects_interleaved_soc_energy_shared_with_non_total_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True, noncollinear=True)
            write_kpoints(folder)
            write_eigenval(folder)
            # Official interleaved SOC shares this header across total, m_x, m_y, and m_z.
            write_interleaved_soc_procar(
                folder,
                header_energy_overrides={(1, 0): 4.5},
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 2, band 1.*spin channel total.*PROCAR components 1-4 "
                r"\(total/m_x/m_y/m_z, interleaved\).*PROCAR 4.500000 eV"
                r".*EIGENVAL 4.050000 eV",
            ):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_missing_soc_spin_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True)
            write_kpoints(folder)
            write_eigenval(folder)
            write_component_procar(folder, component_count=3)

            with self.assertRaisesRegex(RuntimeError, "spin component"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_reordered_soc_spin_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, lsorbit=True)
            write_kpoints(folder)
            write_eigenval(folder)
            write_component_procar(folder, component_count=4)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace("spin component 4", "spin component 3"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "spin component"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_missing_band_header_after_new_kpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            lines = procar.read_text(encoding="utf-8").splitlines()
            remove_line_after(lines, " k-point 2", " band 1")
            procar.write_text("\n".join(lines), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "band header"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_reordered_band_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(" band 2 # energy", " band 1 # energy", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "band header"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_malformed_tot_summary_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            procar.write_text(
                procar.read_text(encoding="utf-8").replace(
                    "\n tot 0.010000", "\n tot malformed", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "summary row"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_rejects_missing_tot_summary_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            procar = folder / "PROCAR"
            lines = procar.read_text(encoding="utf-8").splitlines()
            summary_index = next(index for index, line in enumerate(lines) if line.startswith(" tot "))
            del lines[summary_index]
            procar.write_text("\n".join(lines), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "summary row"):
                ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

    def test_load_store_aggregates_multiple_ions_by_element(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder, elements=("A", "B"), counts=(1, 2))
            write_outcar(folder)
            write_kpoints(folder)
            write_eigenval(folder)
            write_procar(folder, nions=3)
            store = ProcarAdapter.load_store(folder, EigenvalAdapter.load(folder))

        px_index = store.orbitals.index("px")
        self.assertEqual(store.elements, ("A", "B"))
        self.assertAlmostEqual(store.tensor[0, 0, px_index, 0, 0], 0.04)
        self.assertAlmostEqual(store.tensor[0, 1, px_index, 0, 0], 0.20)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lyra.adapters.common import read_poscar
from lyra.adapters.eigenval import EigenvalAdapter
from lyra.models import SpinKind
from tests.fixtures import write_band_fixture, write_eigenval, write_kpoints, write_outcar, write_poscar


def interpolated_points(segments, points_per_segment):
    points = []
    for (start, _start_label), (end, _end_label) in segments:
        points.extend(
            np.linspace(start, end, points_per_segment, endpoint=True).tolist()
        )
    return points


class EigenvalAdapterTests(unittest.TestCase):
    def test_lite_band_preserves_spin_resolved_occupations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)

            data = EigenvalAdapter.load(folder)

        self.assertEqual(set(data.occupations), {"up", "down"})
        np.testing.assert_array_equal(data.occupations["up"], np.ones((9, 2)))
        np.testing.assert_array_equal(data.occupations["down"], np.ones((9, 2)))

    def test_noncollinear_ignores_ispin_and_reads_one_spinor_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(
                folder,
                ispin=2,
                lsorbit=True,
                noncollinear=True,
            )
            write_kpoints(folder)
            write_eigenval(folder, ispin=1)

            data = EigenvalAdapter.load(folder)

        self.assertEqual(tuple(data.energies), ("total",))
        self.assertEqual(data.spin_kind, SpinKind.SOC)
        self.assertAlmostEqual(data.energies["total"][0, 0], 4.0)

    def test_lite_band_retains_poscar_structure_and_atom_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_poscar(
                folder,
                elements=("Fe", "O"),
                counts=(1, 2),
                lattice=((2, 0, 0), (0, 3, 0), (0, 0, 4)),
                positions=((0.25, 0, 0), (0, 0.5, 0), (1.25, -0.25, 2)),
            )

            data = EigenvalAdapter.load(folder)

        self.assertIsNotNone(data.structure)
        assert data.structure is not None
        np.testing.assert_allclose(data.structure.lattice, [[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        self.assertEqual(data.structure.atom_elements, ("Fe", "O", "O"))
        np.testing.assert_allclose(data.structure.fractional_positions, [[0.25, 0, 0], [0, 0.5, 0], [0.25, 0.75, 0]])

    def test_poscar_negative_scale_is_interpreted_as_target_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "POSCAR").write_text(
                "\n".join(
                    [
                        "Target volume",
                        "-8.0",
                        "1 0 0",
                        "0 1 0",
                        "0 0 1",
                        "X",
                        "1",
                        "Direct",
                        "0 0 0",
                    ]
                ),
                encoding="utf-8",
            )

            reciprocal, _, _ = read_poscar(folder / "POSCAR")

        np.testing.assert_allclose(reciprocal, np.eye(3) * np.pi)

    def test_poscar_three_scale_factors_apply_to_cartesian_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "POSCAR").write_text(
                "\n".join(
                    [
                        "Anisotropic scale",
                        "2.0 3.0 4.0",
                        "1 0 0",
                        "0 1 0",
                        "0 0 1",
                        "X",
                        "1",
                        "Direct",
                        "0 0 0",
                    ]
                ),
                encoding="utf-8",
            )

            reciprocal, _, _ = read_poscar(folder / "POSCAR")

        np.testing.assert_allclose(
            reciprocal,
            np.diag([np.pi, 2.0 * np.pi / 3.0, np.pi / 2.0]),
        )

    def test_loads_discontinuous_path_without_adding_jump_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)

            data = EigenvalAdapter.load(folder)

        self.assertEqual(data.tick_labels, ("H", "A|L", "M|H", "K"))
        self.assertEqual(data.segment_ranges, ((0, 3), (3, 6), (6, 9)))
        self.assertEqual(data.distances[2], data.distances[3])
        self.assertEqual(data.distances[5], data.distances[6])
        np.testing.assert_allclose(data.tick_positions, [0.0, 2 * np.pi, 4 * np.pi, 6 * np.pi])

    def test_reads_two_spin_energy_channels_and_outcar_fermi(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder, ispin=2)

            data = EigenvalAdapter.load(folder)

        self.assertEqual(data.spin_kind, SpinKind.COLLINEAR)
        self.assertEqual(tuple(data.energies), ("up", "down"))
        self.assertEqual(data.energies["up"].shape, (9, 2))
        self.assertAlmostEqual(data.energies["down"][0, 0] - data.energies["up"][0, 0], 0.2)
        self.assertEqual(data.efermi, 5.0)

    def test_detects_soc_without_inventing_two_spin_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder, efermi=4.2, ispin=1, lsorbit=True)
            write_kpoints(folder)
            write_eigenval(folder)

            data = EigenvalAdapter.load(folder)

        self.assertEqual(data.spin_kind, SpinKind.SOC)
        self.assertEqual(tuple(data.energies), ("total",))

    def test_rejects_kpoint_count_inconsistent_with_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder, n_per_segment=3)
            write_eigenval(folder, nkpts=8)

            with self.assertRaisesRegex(RuntimeError, "expected 9 k-points"):
                EigenvalAdapter.load(folder)

    def test_rejects_same_count_stale_kpoints_at_first_wrong_endpoint(self):
        segments = (
            (((0, 0, 0), "G"), ((0.5, 0, 0), "X")),
            (((0.25, 0.25, 0), "L"), ((0.25, 0.25, 0.5), "W")),
        )
        stale_segments = (
            segments[0],
            (segments[1][0], ((0.25, 0.25, 0.4), "K")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder, segments=stale_segments)
            write_eigenval(
                folder,
                nkpts=6,
                kpoints=interpolated_points(segments, 3),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r"KPOINTS.*EIGENVAL.*segment 2.*point 2",
            ):
                EigenvalAdapter.load(folder)

    def test_rejects_reordered_segments_and_wrong_interior_interpolation(self):
        segments = (
            (((0, 0, 0), "G"), ((0.5, 0, 0), "X")),
            (((0.25, 0.25, 0), "L"), ((0.25, 0.25, 0.5), "W")),
        )
        expected = interpolated_points(segments, 3)
        cases = {
            "reordered": (segments[::-1], expected, r"segment 1.*point 1"),
            "interior": (
                segments,
                [expected[0], (0.2, 0, 0), *expected[2:]],
                r"segment 1.*point 2",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            for label, (kpoint_segments, eigenval_points, context) in cases.items():
                with self.subTest(label=label):
                    write_kpoints(folder, segments=kpoint_segments)
                    write_eigenval(folder, nkpts=6, kpoints=eigenval_points)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"KPOINTS.*EIGENVAL.*{context}",
                    ):
                        EigenvalAdapter.load(folder)

    def test_accepts_reciprocal_periodic_equivalents_with_valid_distance(self):
        segments = (
            (((0.5, 0, 0), "A"), ((1.0, 0, 0), "B")),
        )
        eigenval_points = interpolated_points(segments, 3)
        eigenval_points[-1] = (0.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder, segments=segments)
            write_eigenval(folder, nkpts=3, kpoints=eigenval_points)

            data = EigenvalAdapter.load(folder)

        np.testing.assert_allclose(data.distances, [0.0, 0.5 * np.pi, np.pi])
        self.assertEqual(data.tick_labels, ("A", "B"))

    def test_reciprocal_and_cartesian_line_modes_map_to_the_same_path(self):
        scale = 2.0
        lattice = np.asarray(
            ((1, 0.2, 0), (0, 2, 0.3), (0.1, 0, 3)),
            dtype=float,
        )
        reciprocal = 2.0 * np.pi * np.linalg.inv(scale * lattice).T
        reciprocal_segments = (
            (((0, 0, 0), "G"), ((0.5, 0.25, 0), "X")),
            (((0.25, 0.25, 0.25), "L"), ((0.25, 0.25, 0.5), "W")),
        )

        def cartesian_endpoint(endpoint):
            coordinates, label = endpoint
            physical = np.asarray(coordinates) @ reciprocal
            return (tuple(physical * scale / (2.0 * np.pi)), label)

        cartesian_segments = tuple(
            tuple(cartesian_endpoint(endpoint) for endpoint in segment)
            for segment in reciprocal_segments
        )
        eigenval_points = interpolated_points(reciprocal_segments, 3)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder, scale=scale, lattice=lattice)
            write_outcar(folder)
            write_eigenval(folder, nkpts=6, kpoints=eigenval_points)
            write_kpoints(folder, segments=reciprocal_segments)
            reciprocal_data = EigenvalAdapter.load(folder)
            write_kpoints(
                folder,
                coordinate_mode="Cartesian",
                segments=cartesian_segments,
            )

            cartesian_data = EigenvalAdapter.load(folder)

        np.testing.assert_allclose(cartesian_data.distances, reciprocal_data.distances)
        np.testing.assert_allclose(cartesian_data.tick_positions, reciprocal_data.tick_positions)
        self.assertEqual(cartesian_data.tick_labels, reciprocal_data.tick_labels)

    def test_rejects_cartesian_line_mode_with_three_poscar_scale_factors(self):
        segments = ((((0, 0, 0), "G"), ((0.5, 0, 0), "X")),)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder, scale=(1.0, 2.0, 3.0))
            write_outcar(folder)
            write_kpoints(folder, coordinate_mode="Cartesian", segments=segments)
            write_eigenval(
                folder,
                nkpts=3,
                kpoints=interpolated_points(segments, 3),
            )

            with self.assertRaisesRegex(RuntimeError, r"Cartesian.*KPOINTS.*POSCAR"):
                EigenvalAdapter.load(folder)

    def test_rejects_invalid_band_id_sequences_with_context(self):
        cases = {
            "duplicate": ((1, 1), r"k-point 1.*band row 2.*expected ID 2.*found 1"),
            "reordered": ((2, 1), r"k-point 1.*band row 1.*expected ID 1.*found 2"),
            "noninteger": (("1.5", 2), r"k-point 1.*band row 1.*integer.*1.5"),
            "out of range": ((3, 2), r"k-point 1.*band row 1.*expected ID 1.*found 3"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder)
            for label, (ids, message) in cases.items():
                with self.subTest(label=label):
                    write_eigenval(folder, band_ids={0: ids})
                    with self.assertRaisesRegex(RuntimeError, message):
                        EigenvalAdapter.load(folder)

    def test_rejects_missing_band_row_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(folder)
            write_outcar(folder)
            write_kpoints(folder)
            write_eigenval(folder, omitted_band_rows={(0, 1)})

            with self.assertRaisesRegex(
                RuntimeError,
                r"k-point 1.*band row 2.*missing",
            ):
                EigenvalAdapter.load(folder)


if __name__ == "__main__":
    unittest.main()

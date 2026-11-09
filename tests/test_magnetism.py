import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lyra.magnetism import (
    MagneticMomentData,
    read_magnetic_moments,
    spinor_to_cartesian,
    validate_magnetic_cif,
    write_vesta_magnetic_files,
)
from tests.fixtures import (
    write_magnetic_outcar,
    write_magnetic_poscar,
    write_truncated_magnetic_outcar,
)


class MagneticMomentParserTests(unittest.TestCase):
    def test_collinear_final_complete_block_preserves_atom_order_and_uses_z_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_magnetic_outcar(folder, scalar_blocks=((9.0, 8.0), (1.5, -2.5)))

            data = read_magnetic_moments(folder)

        self.assertEqual(data.atom_elements, ("Fe", "O"))
        self.assertFalse(data.is_noncollinear)
        np.testing.assert_allclose(data.fractional_positions, [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]])
        np.testing.assert_allclose(data.spinor_vectors, [[0.0, 0.0, 1.5], [0.0, 0.0, -2.5]])
        np.testing.assert_allclose(data.cartesian_vectors, data.spinor_vectors)
        np.testing.assert_allclose(data.lattice, [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

    def test_cartesian_poscar_coordinates_are_scaled_and_converted_to_fractional(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(
                folder,
                scale=2.0,
                coordinate_mode="Cartesian",
                positions=((0.5, 0.0, 0.0), (0.0, 0.5, 0.0)),
            )
            write_magnetic_outcar(folder, scalar_blocks=((1.0, 2.0),))

            data = read_magnetic_moments(folder)

        np.testing.assert_allclose(data.lattice, [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]])
        np.testing.assert_allclose(data.fractional_positions, [[0.25, 0.0, 0.0], [0.0, 1.0 / 6.0, 0.0]])

    def test_noncollinear_components_are_assembled_and_rotated_from_saxis(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_magnetic_outcar(
                folder,
                vector_blocks=(((0.0, 0.0), (0.0, 0.0), (2.0, -3.0)),),
                saxis=(1.0, 0.0, 0.0),
            )

            data = read_magnetic_moments(folder)

        self.assertTrue(data.is_noncollinear)
        np.testing.assert_allclose(data.saxis, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(data.spinor_vectors, [[0.0, 0.0, 2.0], [0.0, 0.0, -3.0]])
        np.testing.assert_allclose(data.cartesian_vectors, [[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], atol=1e-12)

    def test_noncollinear_uses_last_complete_group_when_trailing_group_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_magnetic_outcar(
                folder,
                component_blocks=(
                    ("x", (1.0, 2.0), True),
                    ("y", (3.0, 4.0), True),
                    ("z", (5.0, 6.0), True),
                    ("x", (7.0, 8.0), True),
                    ("y", (9.0, 10.0), True),
                    ("z", (11.0, 12.0), False),
                ),
            )

            data = read_magnetic_moments(folder)

        np.testing.assert_allclose(data.spinor_vectors, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])

    def test_noncollinear_does_not_mix_components_across_x_group_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_magnetic_outcar(
                folder,
                component_blocks=(
                    ("x", (1.0, 2.0), True),
                    ("y", (3.0, 4.0), True),
                    ("x", (5.0, 6.0), True),
                    ("z", (7.0, 8.0), True),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "complete magnetization"):
                read_magnetic_moments(folder)

    def test_spinor_rotation_uses_rz_alpha_then_ry_beta(self):
        vectors = np.array([[1.0, 2.0, 3.0]])

        cartesian = spinor_to_cartesian(vectors, np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(cartesian, [[-2.0, 3.0, -1.0]], atol=1e-12)

    def test_rejects_final_incomplete_magnetization_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_truncated_magnetic_outcar(folder, moments=(1.0, 2.0))

            with self.assertRaisesRegex(RuntimeError, "complete magnetization"):
                read_magnetic_moments(folder)

    def test_rejects_complete_block_with_wrong_atom_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            write_magnetic_outcar(folder, scalar_blocks=((1.0,),))

            with self.assertRaisesRegex(RuntimeError, "atom count"):
                read_magnetic_moments(folder)

    def test_reads_selective_dynamics_direct_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder)
            poscar = (folder / "POSCAR").read_text(encoding="utf-8")
            (folder / "POSCAR").write_text(
                poscar.replace("Direct\n", "Selective Dynamics\nDirect\n").replace(
                    "0.25 0.0 0.0\n0.0 0.5 0.0",
                    "0.25 0.0 0.0 T F T\n0.0 0.5 0.0 F T F",
                ),
                encoding="utf-8",
            )
            write_magnetic_outcar(folder, scalar_blocks=((1.0, 2.0),))

            data = read_magnetic_moments(folder)

        np.testing.assert_allclose(data.fractional_positions, [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]])

    def test_negative_scale_sets_target_volume_without_scaling_direct_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(folder, scale=-192.0)
            write_magnetic_outcar(folder, scalar_blocks=((1.0, 2.0),))

            data = read_magnetic_moments(folder)

        self.assertAlmostEqual(abs(float(np.linalg.det(data.lattice))), 192.0)
        np.testing.assert_allclose(data.lattice, [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]])
        np.testing.assert_allclose(data.fractional_positions, [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0]])

    def test_three_cartesian_scales_apply_by_coordinate_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_magnetic_poscar(
                folder,
                scale="2.0 3.0 4.0",
                coordinate_mode="Cartesian",
                positions=((0.5, 0.5, 0.0), (0.0, 0.5, 0.5)),
                lattice=((1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
            )
            write_magnetic_outcar(folder, scalar_blocks=((1.0, 2.0),))

            data = read_magnetic_moments(folder)

        np.testing.assert_allclose(data.lattice, [[2.0, 3.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 4.0]])
        np.testing.assert_allclose(
            data.fractional_positions,
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            atol=1e-12,
        )


class MagneticCifExportTests(unittest.TestCase):
    def _data(self) -> MagneticMomentData:
        lattice = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        direct_axes = lattice / np.linalg.norm(lattice, axis=1)[:, None]
        crystal_moments = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        cartesian_moments = crystal_moments @ direct_axes
        return MagneticMomentData(
            lattice=lattice,
            atom_elements=("Fe", "O"),
            fractional_positions=np.array([[0.125, 0.25, 0.375], [0.5, 0.625, 0.75]]),
            spinor_vectors=cartesian_moments,
            cartesian_vectors=cartesian_moments,
            saxis=np.array([0.0, 0.0, 1.0]),
            is_noncollinear=True,
        )

    def test_writes_vesta_magnetic_cif_with_unit_direct_axis_moments_and_evidence(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            magnetic_cif, summary = write_vesta_magnetic_files(data, tmp)
            cif_text = magnetic_cif.read_text(encoding="utf-8")
            summary_text = summary.read_text(encoding="utf-8")

            self.assertEqual(magnetic_cif.suffix, ".mcif")
            self.assertEqual(summary.suffix, ".txt")
            self.assertIn("_space_group_name_H-M_alt 'P 1'", cif_text)
            self.assertIn("loop_\n_space_group_symop_operation_xyz\nx,y,z", cif_text)
            self.assertIn("_space_group_magn.name_BNS 'P 1'", cif_text)
            self.assertIn("_space_group_magn.number_BNS 1.1", cif_text)
            self.assertIn(
                "loop_\n"
                "_space_group_symop_magn_operation.id\n"
                "_space_group_symop_magn_operation.xyz\n"
                "1 x,y,z,+1",
                cif_text,
            )
            self.assertIn(
                "loop_\n"
                "_space_group_symop_magn_centering.id\n"
                "_space_group_symop_magn_centering.xyz\n"
                "1 x,y,z,+1",
                cif_text,
            )
            self.assertIn("_cell_length_a 2.00000000", cif_text)
            self.assertIn("_cell_length_b 3.16227766", cif_text)
            self.assertIn(f"_cell_angle_gamma {math.degrees(math.acos(1.0 / math.sqrt(10.0))):.8f}", cif_text)
            self.assertIn("_atom_site_label", cif_text)
            self.assertIn("Fe1 Fe 0.12500000 0.25000000 0.37500000", cif_text)
            self.assertIn("O1 O 0.50000000 0.62500000 0.75000000", cif_text)
            self.assertIn("_atom_site_moment_crystalaxis_x", cif_text)
            self.assertIn("_atom_site_moment_crystalaxis_y", cif_text)
            self.assertIn("_atom_site_moment_crystalaxis_z", cif_text)
            self.assertIn("Fe1 1.00000000 2.00000000 3.00000000", cif_text)
            moment_loop = cif_text.split("_atom_site_moment_crystalaxis_z", maxsplit=1)[1]
            self.assertNotIn("O1", moment_loop)
            self.assertIn("units: mu_B", summary_text)
            self.assertIn("OUTCAR on-site projected moments", summary_text)
            self.assertIn("Fe1", summary_text)
            self.assertIn("O1", summary_text)
            validate_magnetic_cif(magnetic_cif, ("Fe1", "O1"))

    def test_validator_rejects_missing_labels_nonfinite_components_and_incomplete_moment_rows(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            magnetic_cif, _ = write_vesta_magnetic_files(data, tmp)
            valid_text = magnetic_cif.read_text(encoding="utf-8")
            cases = (
                ("missing labels", valid_text.replace("Fe1 Fe", "? Fe", 1)),
                ("finite", valid_text.replace("Fe1 1.00000000 2.00000000 3.00000000", "Fe1 nan 2.00000000 3.00000000", 1)),
                ("complete", valid_text.replace("Fe1 1.00000000 2.00000000 3.00000000", "Fe1 1.00000000 2.00000000", 1)),
            )
            for expected_error, malformed_text in cases:
                with self.subTest(expected_error=expected_error):
                    malformed = Path(tmp) / f"malformed-{expected_error}.mcif"
                    malformed.write_text(malformed_text, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected_error):
                        validate_magnetic_cif(malformed, ("Fe1", "O1"))

    def test_validator_requires_complete_finite_atom_site_rows(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            magnetic_cif, _ = write_vesta_magnetic_files(data, tmp)
            valid_text = magnetic_cif.read_text(encoding="utf-8")
            cases = (
                (
                    "complete atom-site headers",
                    valid_text.replace("_atom_site_type_symbol", "_atom_site_occupancy", 1),
                ),
                (
                    "finite fractional coordinates",
                    valid_text.replace(
                        "Fe1 Fe 0.12500000 0.25000000 0.37500000",
                        "Fe1 Fe nan 0.25000000 0.37500000",
                        1,
                    ),
                ),
                (
                    "finite fractional coordinates",
                    valid_text.replace(
                        "Fe1 Fe 0.12500000 0.25000000 0.37500000",
                        "Fe1 Fe 0.12500000 inf 0.37500000",
                        1,
                    ),
                ),
                (
                    "complete atom-site row",
                    valid_text.replace(
                        "Fe1 Fe 0.12500000 0.25000000 0.37500000",
                        "Fe1 Fe 0.12500000 0.25000000",
                        1,
                    ),
                ),
            )
            for expected_error, malformed_text in cases:
                with self.subTest(expected_error=expected_error):
                    malformed = Path(tmp) / f"malformed-{expected_error}.mcif"
                    malformed.write_text(malformed_text, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected_error):
                        validate_magnetic_cif(malformed, ("Fe1", "O1"))

    def test_validation_failure_preserves_existing_output_pair(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            magnetic_cif = destination / "vesta_magnetic_moments.mcif"
            summary = destination / "magnetic_moment_evidence.txt"
            magnetic_cif.write_text("old magnetic cif\n", encoding="utf-8")
            summary.write_text("old evidence\n", encoding="utf-8")

            with mock.patch(
                "lyra.magnetism.validate_magnetic_cif",
                side_effect=ValueError("injected validation failure"),
            ):
                with self.assertRaisesRegex(ValueError, "injected validation failure"):
                    write_vesta_magnetic_files(data, destination)

            self.assertEqual(magnetic_cif.read_text(encoding="utf-8"), "old magnetic cif\n")
            self.assertEqual(summary.read_text(encoding="utf-8"), "old evidence\n")

    def test_fsync_failure_removes_temporary_file_and_preserves_existing_output_pair(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            magnetic_cif = destination / "vesta_magnetic_moments.mcif"
            summary = destination / "magnetic_moment_evidence.txt"
            magnetic_cif.write_text("old magnetic cif\n", encoding="utf-8")
            summary.write_text("old evidence\n", encoding="utf-8")

            with mock.patch(
                "lyra.magnetism.os.fsync",
                side_effect=OSError("injected fsync failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected fsync failure"):
                    write_vesta_magnetic_files(data, destination)

            self.assertEqual(tuple(destination.glob(".*.tmp")), ())
            self.assertEqual(magnetic_cif.read_text(encoding="utf-8"), "old magnetic cif\n")
            self.assertEqual(summary.read_text(encoding="utf-8"), "old evidence\n")

    def test_second_target_replace_failure_rolls_back_existing_output_pair(self):
        data = self._data()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            magnetic_cif = destination / "vesta_magnetic_moments.mcif"
            summary = destination / "magnetic_moment_evidence.txt"
            magnetic_cif.write_text("old magnetic cif\n", encoding="utf-8")
            summary.write_text("old evidence\n", encoding="utf-8")
            original_replace = Path.replace
            failure_injected = False

            def fail_new_summary_once(source: Path, target: str | Path) -> Path:
                nonlocal failure_injected
                target_path = Path(target)
                if (
                    not failure_injected
                    and target_path == summary
                    and source.name.startswith(f".{summary.name}.")
                    and source.suffix == ".tmp"
                ):
                    failure_injected = True
                    raise OSError("injected second replacement failure")
                return original_replace(source, target)

            with mock.patch.object(Path, "replace", new=fail_new_summary_once):
                with self.assertRaisesRegex(OSError, "injected second replacement failure"):
                    write_vesta_magnetic_files(data, destination)

            self.assertTrue(failure_injected)
            self.assertEqual(magnetic_cif.read_text(encoding="utf-8"), "old magnetic cif\n")
            self.assertEqual(summary.read_text(encoding="utf-8"), "old evidence\n")


if __name__ == "__main__":
    unittest.main()

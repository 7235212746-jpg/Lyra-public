import tempfile
import unittest
from pathlib import Path

import numpy as np

from lyra.adapters.common import (
    parse_line_mode_kpoints,
    read_poscar,
    read_poscar_structure,
)
from lyra.models import ParsedStructure
from tests.fixtures import write_poscar


class PoscarStructureTests(unittest.TestCase):

    def test_read_poscar_keeps_header_only_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "POSCAR"
            path.write_text(
                "Synthetic\n1.0\n1 0 0\n0 1 0\n0 0 1\nTa Ga\n1 2\n",
                encoding="utf-8",
            )

            reciprocal, elements, counts = read_poscar(path)

        np.testing.assert_allclose(reciprocal, 2.0 * np.pi * np.eye(3))
        self.assertEqual(elements, ("Ta", "Ga"))
        self.assertEqual(counts, (1, 2))

    def test_parsed_structure_owns_atom_element_tuple(self):
        elements = ["Ta"]
        structure = ParsedStructure(
            lattice=np.eye(3),
            atom_elements=elements,
            fractional_positions=np.zeros((1, 3)),
            source="synthetic",
        )

        elements.append("Ga")

        self.assertEqual(structure.atom_elements, ("Ta",))
    def test_reads_direct_selective_dynamics_and_wraps_fractional_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(
                folder,
                elements=("Fe", "O"),
                counts=(1, 2),
                lattice=((2, 0, 0), (0, 3, 0), (0, 0, 4)),
                selective_dynamics=True,
                positions=((0.25, 0, 0, "T", "T", "T"), (0, 0.5, 0, "F", "T", "F"), (1.25, -0.25, 2, "T", "F", "F")),
            )

            structure = read_poscar_structure(folder / "POSCAR")

        np.testing.assert_allclose(structure.lattice, [[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        self.assertEqual(structure.atom_elements, ("Fe", "O", "O"))
        np.testing.assert_allclose(structure.fractional_positions, [[0.25, 0, 0], [0, 0.5, 0], [0.25, 0.75, 0]])
        self.assertEqual(structure.source, str(folder / "POSCAR"))

    def test_scales_cartesian_positions_before_converting_them_to_fractional(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(
                folder,
                scale=2.0,
                lattice=((2, 0, 0), (0, 3, 0), (0, 0, 4)),
                coordinate_mode="Cartesian",
                positions=((0.5, 0, 0), (0, 1, 0)),
                elements=("Fe", "O"),
                counts=(1, 1),
            )

            structure = read_poscar_structure(folder / "POSCAR")

        np.testing.assert_allclose(structure.lattice, [[4, 0, 0], [0, 6, 0], [0, 0, 8]])
        np.testing.assert_allclose(structure.fractional_positions, [[0.25, 0, 0], [0, 1 / 3, 0]])

    def test_interprets_negative_scale_as_target_volume_and_three_scales_as_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_poscar(
                folder,
                scale=-192.0,
                lattice=((1, 0, 0), (0, 2, 0), (0, 0, 4)),
                positions=((1.25, -0.25, 2.0),),
            )
            target_volume = read_poscar_structure(folder / "POSCAR")
            write_poscar(
                folder,
                scale=(2.0, 3.0, 4.0),
                lattice=((1, 1, 0), (0, 1, 1), (0, 0, 1)),
                coordinate_mode="K-point",
                positions=((0.25, 0.5, 0.25),),
            )
            component_scaled = read_poscar_structure(folder / "POSCAR")
            reciprocal, _, _ = read_poscar(folder / "POSCAR")

        self.assertAlmostEqual(abs(float(np.linalg.det(target_volume.lattice))), 192.0)
        np.testing.assert_allclose(target_volume.fractional_positions, [[0.25, 0.75, 0]])
        np.testing.assert_allclose(component_scaled.lattice, [[2, 3, 0], [0, 3, 4], [0, 0, 4]])
        np.testing.assert_allclose(component_scaled.fractional_positions, [[0.25, 0.25, 0]])
        np.testing.assert_allclose(reciprocal, 2.0 * np.pi * np.linalg.inv(component_scaled.lattice).T)

    def test_rejects_invalid_or_incomplete_structure_input(self):
        cases = {
            "zero scale": "Synthetic\n0\n1 0 0\n0 1 0\n0 0 1\nX\n1\nDirect\n0 0 0",
            "singular lattice": "Synthetic\n1\n1 0 0\n2 0 0\n0 0 1\nX\n1\nDirect\n0 0 0",
            "negative count": "Synthetic\n1\n1 0 0\n0 1 0\n0 0 1\nX\n-1\nDirect",
            "missing coordinate mode": "Synthetic\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1",
            "incomplete coordinates": "Synthetic\n1\n1 0 0\n0 1 0\n0 0 1\nX\n2\nDirect\n0 0 0",
            "nonfinite position": "Synthetic\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\nDirect\nnan 0 0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            poscar = folder / "POSCAR"
            for label, text in cases.items():
                with self.subTest(label=label):
                    poscar.write_text(text, encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        read_poscar_structure(poscar)

    def test_compatibility_wrapper_preserves_poscar_element_and_count_lines(self):
        text = "\n".join(
            [
                "Repeated species",
                "1",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "X Y X",
                "1 0 1",
                "Direct",
                "0 0 0",
                "0.5 0.5 0.5",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            poscar = Path(tmp) / "POSCAR"
            poscar.write_text(text, encoding="utf-8")

            reciprocal, elements, counts = read_poscar(poscar)

        np.testing.assert_allclose(reciprocal, 2.0 * np.pi * np.eye(3))
        self.assertEqual(elements, ("X", "Y", "X"))
        self.assertEqual(counts, (1, 0, 1))


class LineModeKpointsTests(unittest.TestCase):
    def test_parser_retains_mode_coordinates_labels_and_segment_order(self):
        text = "\n".join(
            [
                "Ordered path",
                "3 ! points per segment",
                "Line-mode",
                "Reciprocal",
                "0 0 0 ! Gamma",
                "0.5 0 0 ! X",
                "",
                "0.25 0.25 0 L",
                "0.25 0.25 0.5 W",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "KPOINTS"
            path.write_text(text, encoding="utf-8")

            definition = parse_line_mode_kpoints(path)

        self.assertEqual(definition.points_per_segment, 3)
        self.assertEqual(definition.coordinate_mode, "reciprocal")
        self.assertEqual(definition.segments[0].start.coordinates, (0.0, 0.0, 0.0))
        self.assertEqual(definition.segments[0].start.label, "G")
        self.assertEqual(definition.segments[1].end.coordinates, (0.25, 0.25, 0.5))
        self.assertEqual(definition.segments[1].end.label, "W")

    def test_parser_rejects_non_line_mode_and_malformed_endpoint_pairs(self):
        cases = {
            "wrong mode": "Path\n3\nGamma\nReciprocal\n0 0 0\n0.5 0 0",
            "one point per segment": "Path\n1\nLine-mode\nReciprocal\n0 0 0\n0.5 0 0",
            "odd endpoints": "Path\n3\nLine-mode\nReciprocal\n0 0 0",
            "malformed endpoint": "Path\n3\nLine-mode\nReciprocal\n0 0 0\nnot-a-point",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "KPOINTS"
            for label, text in cases.items():
                with self.subTest(label=label):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "KPOINTS"):
                        parse_line_mode_kpoints(path)


if __name__ == "__main__":
    unittest.main()

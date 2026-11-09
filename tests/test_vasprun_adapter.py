import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lyra.adapters.vasprun import (
    VasprunAdapter,
    _band_projections,
    _dos_projections,
    _read_xml_efermi,
)
from tests.fixtures import write_kpoints


class _FakeSpin:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)


UP = _FakeSpin("up")
DOWN = _FakeSpin("down")
ORBITALS = ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2")
MINIMAL_VASPRUN = '<modeling><dos><i name="efermi">4.25</i></dos></modeling>'


def _projection_tensor(scale):
    band_kpoint_grid = np.arange(18, dtype=float).reshape(2, 9) + 1.0
    projections = np.zeros((2, 9, len(ORBITALS), 3))
    projections[:, :, 0, 0] = scale * band_kpoint_grid
    projections[:, :, 0, 1] = scale * 2.0 * band_kpoint_grid
    projections[:, :, 1, 0] = scale * 10.0 * band_kpoint_grid
    projections[:, :, 1, 1] = scale * 20.0 * band_kpoint_grid
    projections[:, :, 0, 2] = scale * 1000.0
    return projections


class _FakeSite:
    def __init__(self, specie):
        self.specie = specie


class _FakeVasprun:
    calls = []

    def __init__(self, filename, **kwargs):
        self.calls.append((Path(filename), kwargs))
        self.efermi = 4.25
        self.final_structure = SimpleNamespace(
            sites=[_FakeSite("X")]
        )
        self.complete_dos = self._dos()

    def get_band_structure(self, **kwargs):
        projections = np.zeros((2, 9, len(ORBITALS), 1))
        projections[:, :, 0, 0] = 0.2
        projections[:, :, 1:4, 0] = 0.1
        return SimpleNamespace(
            bands={UP: np.arange(18, dtype=float).reshape(2, 9)},
            kpoints=[SimpleNamespace(frac_coords=(0, 0, index)) for index in range(9)],
            projections={UP: projections},
            structure=self.final_structure,
            efermi=4.25,
        )

    def _dos(self):
        energies = np.array([3.0, 4.0, 5.0])
        orbital_dos = {}
        for index, orbital in enumerate(ORBITALS):
            orbital_dos[orbital] = {UP: np.full(3, index + 1.0)}
        return SimpleNamespace(
            energies=energies,
            efermi=4.25,
            densities={UP: np.array([1.0, 2.0, 3.0]), DOWN: np.array([2.0, 3.0, 4.0])},
            structure=self.final_structure,
            pdos={self.final_structure.sites[0]: orbital_dos},
        )


class _MultiIonFakeVasprun(_FakeVasprun):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.final_structure = SimpleNamespace(
            sites=[_FakeSite("X"), _FakeSite("X"), _FakeSite("Y")]
        )

    def get_band_structure(self, **kwargs):
        return SimpleNamespace(
            bands={
                UP: np.arange(18, dtype=float).reshape(2, 9),
                DOWN: np.arange(18, dtype=float).reshape(2, 9) + 100.0,
            },
            kpoints=[SimpleNamespace(frac_coords=(0, 0, index)) for index in range(9)],
            projections={UP: _projection_tensor(1.0), DOWN: _projection_tensor(10.0)},
            structure=self.final_structure,
            efermi=4.25,
        )


class _PymatgenLikeVasprun(_FakeVasprun):
    """Match pymatgen: parse_dos=False leaves self.efermi unset."""

    received_efermi = None

    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.efermi = None

    def get_band_structure(self, *, efermi=None, **kwargs):
        self.__class__.received_efermi = efermi
        if efermi is None:
            raise ValueError("e_fermi is None.")
        result = super().get_band_structure(**kwargs)
        result.efermi = efermi
        return result


class _OccupationFakeVasprun(_FakeVasprun):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        energies = np.arange(18, dtype=float).reshape(9, 2)
        occupations = np.column_stack((np.full(9, 2.0), np.zeros(9)))
        self.eigenvalues = {
            UP: np.stack((energies, occupations), axis=-1),
        }


class VasprunAdapterTests(unittest.TestCase):
    def setUp(self):
        _FakeVasprun.calls.clear()
        _PymatgenLikeVasprun.received_efermi = None

    def test_full_band_opens_xml_only_on_explicit_load_and_converts_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text(MINIMAL_VASPRUN, encoding="utf-8")
            write_kpoints(folder)

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(_FakeVasprun, ORBITALS),
            ):
                data = VasprunAdapter.load_band(folder)

        self.assertEqual(_FakeVasprun.calls[0][0].name, "vasprun.xml")
        self.assertTrue(_FakeVasprun.calls[0][1]["parse_projected_eigen"])
        self.assertEqual(data.energies["total"].shape, (9, 2))
        self.assertIn("X:p", data.projections)
        self.assertEqual(data.tick_labels, ("H", "A|L", "M|H", "K"))

    def test_full_band_reads_xml_fermi_before_pymatgen_projection_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text(
                "<modeling><dos><i name=\"efermi\">6.75</i></dos></modeling>",
                encoding="utf-8",
            )
            write_kpoints(folder)

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(_PymatgenLikeVasprun, ORBITALS),
            ):
                data = VasprunAdapter.load_band(folder)

        self.assertEqual(_PymatgenLikeVasprun.received_efermi, 6.75)
        self.assertEqual(data.efermi, 6.75)

    def test_full_band_preserves_vasprun_occupations_in_plot_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text(MINIMAL_VASPRUN, encoding="utf-8")
            write_kpoints(folder)

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(_OccupationFakeVasprun, ORBITALS),
            ):
                data = VasprunAdapter.load_band(folder)

        self.assertEqual(set(data.occupations), {"total"})
        self.assertEqual(data.occupations["total"].shape, (9, 2))
        np.testing.assert_array_equal(data.occupations["total"][:, 0], np.full(9, 2.0))

    def test_xml_fermi_uses_main_scf_dos_not_kpoints_opt_dos(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vasprun.xml"
            path.write_text(
                "<modeling>"
                '<dos comment="kpoints_opt"><i name="efermi">99.0</i></dos>'
                '<dos><i name="efermi">5.25</i></dos>'
                "</modeling>",
                encoding="utf-8",
            )

            value = _read_xml_efermi(path)

        self.assertEqual(value, 5.25)

    def test_xml_fermi_rejects_non_finite_values_before_full_parse(self):
        for raw in ("NaN", "Inf", "-Inf"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "vasprun.xml"
                path.write_text(
                    f'<modeling><dos><i name="efermi">{raw}</i></dos></modeling>',
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(RuntimeError, "invalid E-fermi"):
                    _read_xml_efermi(path)

    def test_full_band_uses_pymatgen_orbital_then_ion_projection_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text(MINIMAL_VASPRUN, encoding="utf-8")
            write_kpoints(folder)

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(_MultiIonFakeVasprun, ORBITALS),
            ):
                data = VasprunAdapter.load_band(folder)

        band_kpoint_grid = np.arange(18, dtype=float).reshape(2, 9) + 1.0
        self.assertEqual(set(data.projections["X:py"]), {"up", "down"})
        self.assertEqual(data.projections["X:py"]["up"].shape, (9, 2))
        np.testing.assert_allclose(
            data.projections["X:py"]["up"], 30.0 * band_kpoint_grid.T
        )
        np.testing.assert_allclose(
            data.projections["X:py"]["down"], 300.0 * band_kpoint_grid.T
        )
        np.testing.assert_allclose(
            data.projections["X"]["up"], 33.0 * band_kpoint_grid.T
        )
        np.testing.assert_allclose(
            data.projections["Y"]["up"], np.full((9, 2), 1000.0)
        )
        np.testing.assert_allclose(
            data.projection_total["up"],
            33.0 * band_kpoint_grid.T + 1000.0,
        )
        np.testing.assert_allclose(
            data.projection_total["down"],
            330.0 * band_kpoint_grid.T + 10000.0,
        )

    def test_full_band_s_projection_is_accumulated_once(self):
        tensor = np.zeros((1, 1, 1, 1))
        tensor[0, 0, 0, 0] = 0.25
        band_structure = SimpleNamespace(
            projections={UP: tensor},
            structure=SimpleNamespace(sites=[_FakeSite("X")]),
        )

        projections, orbitals = _band_projections(
            band_structure, ("s",), {UP: "total"}
        )

        self.assertEqual(orbitals, ("s",))
        np.testing.assert_allclose(projections["X:s"]["total"], [[0.25]])

    def test_full_dos_s_projection_is_accumulated_once(self):
        site = _FakeSite("X")
        complete_dos = SimpleNamespace(
            pdos={site: {"s": {UP: np.array([0.25])}}}
        )

        projections, orbitals = _dos_projections(complete_dos, {UP: "total"})

        self.assertEqual(orbitals, ("s",))
        np.testing.assert_allclose(projections["X:s"]["total"], [0.25])

    def test_full_band_rejects_projection_tensor_with_too_few_orbitals(self):
        band_structure = SimpleNamespace(
            projections={UP: np.zeros((1, 1, 1, 1))},
            structure=SimpleNamespace(sites=[_FakeSite("X")]),
        )

        with self.assertRaisesRegex(RuntimeError, "orbital dimension"):
            _band_projections(band_structure, ("s", "px"), {UP: "total"})

    def test_full_band_accepts_spd_tensor_from_complete_pymatgen_orbital_enum(self):
        pymatgen_orbitals = (
            "s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2",
            "f_3", "f_2", "f_1", "f0", "f1", "f2", "f3",
        )
        tensor = np.zeros((1, 1, 9, 1))
        tensor[0, 0, 8, 0] = 0.75
        band_structure = SimpleNamespace(
            projections={UP: tensor},
            structure=SimpleNamespace(sites=[_FakeSite("X")]),
        )

        projections, orbitals = _band_projections(
            band_structure, pymatgen_orbitals, {UP: "total"}
        )

        self.assertEqual(orbitals, (*pymatgen_orbitals[:8], "dx2-y2"))
        np.testing.assert_allclose(projections["X:dx2-y2"]["total"], [[0.75]])

    def test_full_band_rejects_projection_tensor_with_too_few_ions(self):
        band_structure = SimpleNamespace(
            projections={UP: np.zeros((1, 1, 1, 1))},
            structure=SimpleNamespace(sites=[_FakeSite("X"), _FakeSite("Y")]),
        )

        with self.assertRaisesRegex(RuntimeError, "ion dimension"):
            _band_projections(band_structure, ("s",), {UP: "total"})

    def test_full_dos_prefers_gz_and_converts_site_orbital_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml.gz").write_bytes(b"xml")

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                return_value=(_FakeVasprun, ORBITALS),
            ):
                data = VasprunAdapter.load_dos(folder)

        self.assertEqual(_FakeVasprun.calls[0][0].name, "vasprun.xml.gz")
        self.assertEqual(set(data.densities), {"up", "down"})
        self.assertIn("X:d", data.projections)
        self.assertIn("X:dxy", data.projections)

    def test_missing_optional_dependency_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text("xml", encoding="utf-8")

            with patch(
                "lyra.adapters.vasprun._import_pymatgen",
                side_effect=RuntimeError("Full Mode requires pymatgen"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Full Mode requires pymatgen"):
                    VasprunAdapter.load_dos(folder)

    def test_missing_xml_is_reported_before_dependency_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("lyra.adapters.vasprun._import_pymatgen") as dependency:
                with self.assertRaisesRegex(FileNotFoundError, "vasprun.xml"):
                    VasprunAdapter.load_dos(Path(tmp))
            dependency.assert_not_called()


if __name__ == "__main__":
    unittest.main()

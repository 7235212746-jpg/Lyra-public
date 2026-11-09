from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lyra.cache import CacheIdentity, CacheSource, ProjectionCache
from lyra.data_service import ProjectionDataService
from lyra.models import (
    BandPlotSettings,
    BandStateSelection,
    PlotRange,
    ProjectionChannel,
    ReadMode,
)
from tests.fixtures import (
    write_band_fixture,
    write_dos_fixture,
    write_fermi_fixture,
    write_procar,
)
from tests.test_projection_store import parsed_band_data, parsed_dos_data


class ProjectionDataServiceTests(unittest.TestCase):
    def test_hot_memory_accounting_counts_shared_numpy_array_once(self):
        shared = np.zeros(128, dtype=np.float32)
        payload = {"first": shared, "second": [shared]}

        measured = ProjectionDataService._estimate_hot_bytes(payload)

        self.assertGreaterEqual(measured, shared.nbytes)
        self.assertLess(measured, shared.nbytes * 2)

    def test_hot_memory_budget_evicts_least_recently_used_without_touching_disk(self):
        folder = self.root / "budget"
        marker = folder / ".lyra-cache" / "marker"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"disk-cache")
        first_key = (folder.resolve(), "first")
        second_key = (folder.resolve(), "second")
        first = np.zeros(64, dtype=np.uint8)
        second = np.ones(64, dtype=np.uint8)
        service = ProjectionDataService(hot_memory_budget=100)

        service._remember(service._fermi_memory, first_key, first)
        service._touch_hot_entry(service._fermi_memory, first_key)
        service._remember(service._fermi_memory, second_key, second)

        self.assertNotIn(first_key, service._fermi_memory)
        self.assertIn(second_key, service._fermi_memory)
        self.assertEqual(marker.read_bytes(), b"disk-cache")

    def test_hot_memory_oversize_item_is_returnable_but_not_resident(self):
        service = ProjectionDataService(hot_memory_budget=16)
        key = (self.root.resolve(), "oversize")
        item = np.zeros(32, dtype=np.uint8)

        service._remember(service._fermi_memory, key, item)

        self.assertNotIn(key, service._fermi_memory)

    def test_explicit_clear_removes_hot_order_references_and_accounting(self):
        folder = self.root / "clear-hot"
        folder.mkdir()
        service = ProjectionDataService(hot_memory_budget=2048)
        key = (folder.resolve(), "fermi-test")
        item = np.zeros(1024, dtype=np.uint8)
        service._remember(service._fermi_memory, key, item)
        self.assertEqual(service._resident_hot_bytes(), 1024)

        service.clear(folder, "fermi")

        self.assertEqual(service._fermi_memory, {})
        self.assertEqual(service._hot_order, {})
        self.assertEqual(service._resident_hot_bytes(), 0)

    def test_cancelled_rebuild_never_replaces_prior_hot_memory(self):
        folder = self.root / "rebuild-memory"
        folder.mkdir()
        old_identity = CacheIdentity(
            1, "old", "lite", "band", (CacheSource("PROCAR", 1, 1),)
        )
        new_identity = CacheIdentity(
            1, "new", "lite", "band", (CacheSource("PROCAR", 2, 2),)
        )
        old_bundle = (
            replace(parsed_band_data(), folder=folder.resolve(), projections={}),
            None,
        )
        new_bundle = (
            replace(
                parsed_band_data(),
                folder=folder.resolve(),
                projections={},
                efermi=9.5,
            ),
            None,
        )
        service = ProjectionDataService()
        old_key = (folder.resolve(), old_identity)
        new_key = (folder.resolve(), new_identity)
        service._remember(service._band_memory, old_key, old_bundle)
        token = service.begin_rebuild(folder, "band")

        loaded = service._resolve_band(
            folder.resolve(),
            new_identity,
            lambda: new_bundle,
            lambda _event: None,
            rebuild=True,
        )

        self.assertIs(loaded, new_bundle)
        self.assertIs(service._band_memory[old_key], old_bundle)
        self.assertNotIn(new_key, service._band_memory)

        service.finish_rebuild(folder, "band", token, success=False)

        self.assertIs(service._band_memory[old_key], old_bundle)
        self.assertNotIn(new_key, service._band_memory)

    def test_cache_commit_failure_never_replaces_prior_hot_memory(self):
        folder = self.root / "failed-memory-commit"
        folder.mkdir()
        old_identity = CacheIdentity(
            1, "old", "lite", "band", (CacheSource("PROCAR", 1, 1),)
        )
        new_identity = CacheIdentity(
            1, "new", "lite", "band", (CacheSource("PROCAR", 2, 2),)
        )
        old_bundle = (
            replace(parsed_band_data(), folder=folder.resolve(), projections={}),
            None,
        )
        new_bundle = (
            replace(
                parsed_band_data(),
                folder=folder.resolve(),
                projections={},
                efermi=9.5,
            ),
            None,
        )
        service = ProjectionDataService()
        old_key = (folder.resolve(), old_identity)
        new_key = (folder.resolve(), new_identity)
        service._remember(service._band_memory, old_key, old_bundle)
        service.begin_rebuild(folder, "band")
        service._stage_rebuild_memory(service._band_memory, new_key, new_bundle)

        with patch.object(
            service.cache,
            "finish_rebuild",
            side_effect=OSError("cache commit failed"),
        ):
            with self.assertRaisesRegex(OSError, "cache commit failed"):
                service.finish_rebuild(folder, "band", None, success=True)

        self.assertIs(service._band_memory[old_key], old_bundle)
        self.assertNotIn(new_key, service._band_memory)

    def test_lite_dos_service_rejects_fermi_inconsistent_doscar_before_caching(self):
        folder = self.root / "dos"
        folder.mkdir()
        write_dos_fixture(folder, projected=False)
        lines = (folder / "DOSCAR").read_text(encoding="utf-8").splitlines()
        header = lines[5].split()
        header[3] = "5.001"
        lines[5] = " ".join(header)
        (folder / "DOSCAR").write_text("\n".join(lines), encoding="utf-8")
        service = ProjectionDataService()

        with self.assertRaisesRegex(RuntimeError, "E-fermi"):
            service.load_dos(folder, "lite")

        self.assertEqual(service._dos_memory, {})

    def test_fermi_data_reuses_memory_and_tracks_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_fermi_fixture(folder)
            calls = []

            def loader(source):
                calls.append(Path(source).resolve())
                from lyra.adapters.fermi import FermiAdapter

                return FermiAdapter.load_lite(source)

            service = ProjectionDataService(fermi_lite_loader=loader)
            identity = service.fermi_identity(folder, ReadMode.LITE)

            first = service.load_fermi(folder, ReadMode.LITE, identity=identity)
            second = service.load_fermi(folder, ReadMode.LITE, identity=identity)
            matches = ProjectionDataService.identity_matches(folder, identity)

        self.assertIs(first, second)
        self.assertEqual(calls, [folder.resolve()])
        self.assertEqual(identity.data_kind, "fermi")
        self.assertTrue(matches)

    def test_scientific_parser_changes_invalidate_lite_and_full_band_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lite = root / "lite"
            full = root / "full"
            lite.mkdir()
            full.mkdir()
            write_band_fixture(lite)
            (full / "vasprun.xml").write_text("xml", encoding="utf-8")
            (full / "KPOINTS").write_text("fixture", encoding="utf-8")

            service = ProjectionDataService()
            lite_identity = service.band_identity(lite, ReadMode.LITE, ())
            full_identity = service.band_identity(full, ReadMode.FULL, ())

        self.assertEqual(lite_identity.parser_version, "0.2.2")
        self.assertEqual(full_identity.parser_version, "0.1.5")

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_cached_band_point_query_uses_complete_cache_without_hot_residency(self):
        folder = self.root / "point-query"
        folder.mkdir()
        write_band_fixture(folder)
        write_procar(folder)
        builder = ProjectionDataService()
        index, identity, _display = builder.prepare_band_inspector(folder, "lite")
        service = ProjectionDataService()
        selection = BandStateSelection(index.spin_names[0], 0, 0)

        detail = service.query_cached_band_state(folder, identity, selection)

        self.assertEqual(detail.selection, selection)
        self.assertEqual(
            detail.raw_orbital_weights.shape,
            (len(index.elements), len(index.orbitals)),
        )
        self.assertEqual(service._band_memory, {})

    def test_prepare_band_zoom_reuses_exact_projection_cache_identity(self):
        folder = self.root / "zoom-cache"
        folder.mkdir()
        write_band_fixture(folder)
        write_procar(folder)
        builder = ProjectionDataService()
        _index, identity, _display = builder.prepare_band_inspector(folder, "lite")
        settings = BandPlotSettings(
            energy_range=PlotRange(-2, 2),
            mode="fat",
            projection_channels=(ProjectionChannel("X", "dx2-y2"),),
        )
        service = ProjectionDataService(
            eigenval_loader=lambda _folder: (_ for _ in ()).throw(
                AssertionError("source loader must not run")
            ),
            procar_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("PROCAR loader must not run")
            ),
        )

        prepared = service.prepare_band_zoom(folder, "lite", settings, identity)

        self.assertEqual(prepared["identity"], identity)
        self.assertIn("X:dx2-y2", prepared["data"].projections)
        self.assertIs(prepared["settings"], settings)

    def test_prepare_paired_band_builds_independent_complete_caches(self):
        left = self.root / "paired-left"
        right = self.root / "paired-right"
        left.mkdir()
        right.mkdir()
        write_band_fixture(left)
        write_procar(left)
        write_band_fixture(right)
        write_procar(right)
        service = ProjectionDataService()

        session = service.prepare_paired_band(left, "lite", right, "lite")

        self.assertEqual(session.left_folder, left.resolve())
        self.assertEqual(session.right_folder, right.resolve())
        self.assertNotEqual(session.left_identity.sources, ())
        self.assertNotEqual(session.right_identity.sources, ())
        self.assertTrue(session.compatibility.sync_enabled)
        left_detail = ProjectionDataService().query_cached_band_state(
            left,
            session.left_identity,
            BandStateSelection(session.left.spin_names[0], 0, 0),
        )
        right_detail = ProjectionDataService().query_cached_band_state(
            right,
            session.right_identity,
            BandStateSelection(session.right.spin_names[0], 0, 0),
        )
        self.assertGreater(left_detail.raw_projection_total, 0.0)
        self.assertGreater(right_detail.raw_projection_total, 0.0)

    def test_prepare_paired_band_keeps_standard_side_without_procar(self):
        left = self.root / "paired-standard"
        right = self.root / "paired-projected"
        left.mkdir()
        right.mkdir()
        write_band_fixture(left)
        write_band_fixture(right)
        write_procar(right)

        session = ProjectionDataService().prepare_paired_band(
            left, "lite", right, "lite"
        )

        self.assertFalse(session.left_detail_available)
        self.assertTrue(session.right_detail_available)
        self.assertEqual(session.left.raw_element_totals.shape[-2:], (9, 2))

    @staticmethod
    def _write_sources(folder: Path, *names: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / name).write_text(f"source:{name}", encoding="utf-8")

    def _service(self, calls: list[tuple[str, Path]]) -> ProjectionDataService:
        def eigenval(folder):
            folder = Path(folder).resolve()
            calls.append(("eigenval", folder))
            return replace(parsed_band_data(), folder=folder)

        def procar(folder, band_data, progress=None):
            from lyra.projection_store import BandProjectionStore

            folder = Path(folder).resolve()
            calls.append(("procar", folder))
            if progress is not None:
                progress({"phase": "parse", "current": 1, "total": 1, "detail": "PROCAR"})
            return BandProjectionStore.from_parsed(replace(parsed_band_data(), folder=folder))

        def vasprun_band(folder):
            folder = Path(folder).resolve()
            calls.append(("vasprun-band", folder))
            return replace(parsed_band_data(), folder=folder)

        def vasprun_dos(folder):
            folder = Path(folder).resolve()
            calls.append(("vasprun-dos", folder))
            return replace(parsed_dos_data(), folder=folder)

        def doscar(folder):
            folder = Path(folder).resolve()
            calls.append(("doscar", folder))
            return replace(parsed_dos_data(), folder=folder)

        return ProjectionDataService(
            cache=ProjectionCache(),
            eigenval_loader=eigenval,
            procar_loader=procar,
            vasprun_band_loader=vasprun_band,
            vasprun_dos_loader=vasprun_dos,
            doscar_loader=doscar,
        )

    def test_lite_projected_lookup_order_is_memory_then_folder_cache_then_source(self):
        folder = self.root / "band"
        self._write_sources(folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        calls: list[tuple[str, Path]] = []
        events: list[dict] = []
        service = self._service(calls)

        first = service.load_band(
            folder, ReadMode.LITE, (ProjectionChannel("A", "p"),), events.append
        )
        ProjectionCache().clear(folder)
        second = service.load_band(
            folder, "lite", (ProjectionChannel("B", "d"),), events.append
        )

        self.assertEqual(calls, [("eigenval", folder.resolve()), ("procar", folder.resolve())])
        self.assertEqual(set(first.projections), {"A:p"})
        self.assertEqual(set(second.projections), {"B:d"})
        self.assertIn("memory-hit", [event["phase"] for event in events])

        service = self._service(calls)
        service.load_band(folder, "lite", (ProjectionChannel("A", "s"),), events.append)
        self.assertEqual(calls[-2:], [("eigenval", folder.resolve()), ("procar", folder.resolve())])

        service.load_band(folder, "lite", (ProjectionChannel("A", "s"),), events.append)
        self.assertEqual([event["phase"] for event in events][-2:], ["validate", "memory-hit"])

    def test_folder_cache_hit_avoids_source_loaders_and_materializes_new_channels(self):
        folder = self.root / "band"
        self._write_sources(folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        calls: list[tuple[str, Path]] = []
        self._service(calls).load_band(
            folder, "lite", (ProjectionChannel("A", "p"),), lambda _event: None
        )
        events: list[dict] = []

        loaded = self._service(calls).load_band(
            folder, "lite", (ProjectionChannel("B", "dxy"),), events.append
        )

        self.assertEqual(calls, [("eigenval", folder.resolve()), ("procar", folder.resolve())])
        self.assertEqual(set(loaded.projections), {"B:dxy"})
        self.assertEqual([event["phase"] for event in events], ["validate", "cache-hit"])

    def test_touching_procar_invalidates_lite_projected_memory_and_folder_cache(self):
        folder = self.root / "band"
        self._write_sources(folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)
        channel = (ProjectionChannel("A", "s"),)
        service.load_band(folder, "lite", channel, lambda _event: None)

        with (folder / "PROCAR").open("a", encoding="utf-8") as handle:
            handle.write("changed")
        service.load_band(folder, "lite", channel, lambda _event: None)

        self.assertEqual([name for name, _folder in calls], ["eigenval", "procar", "eigenval", "procar"])

    def test_lite_standard_band_and_dos_reuse_previewed_memory_without_disk_cache(self):
        band_folder = self.root / "band"
        dos_folder = self.root / "dos"
        self._write_sources(band_folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        self._write_sources(dos_folder, "DOSCAR", "POSCAR", "OUTCAR")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)

        events: list[dict] = []
        band = service.load_band(band_folder, "lite", (), events.append)
        dos = service.load_dos(dos_folder, "lite", (), events.append)
        service.load_band(band_folder, "lite", (), events.append)
        service.load_dos(dos_folder, "lite", (), events.append)

        self.assertEqual([name for name, _folder in calls], ["eigenval", "doscar"])
        self.assertEqual(band.folder, band_folder.resolve())
        self.assertEqual(dos.folder, dos_folder.resolve())
        self.assertEqual([event["phase"] for event in events].count("memory-hit"), 2)
        self.assertIn("cache-not-used", [event["phase"] for event in events])
        self.assertFalse((band_folder / ".lyra-cache").exists())
        self.assertFalse((dos_folder / ".lyra-cache").exists())

    def test_cache_write_failure_is_reported_as_a_distinct_memory_only_event(self):
        folder = self.root / "band"
        self._write_sources(folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        calls: list[tuple[str, Path]] = []

        class FailingCache(ProjectionCache):
            def save_band(self, *args, **kwargs):
                raise OSError("disk is read-only")

        service = self._service(calls)
        service.cache = FailingCache()
        events: list[dict] = []

        service.load_band(folder, "lite", (ProjectionChannel("A", "s"),), events.append)

        self.assertIn("cache-write-failed", [event["phase"] for event in events])

    def test_full_band_and_dos_persist_normalized_data_in_separate_namespaces(self):
        folder = self.root / "full"
        self._write_sources(folder, "vasprun.xml", "KPOINTS")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)

        band = service.load_band(
            folder, "full", (ProjectionChannel("A", "px"),), lambda _event: None
        )
        dos = service.load_dos(
            folder, "full", (ProjectionChannel("B", "d"),), lambda _event: None
        )

        self.assertEqual([name for name, _folder in calls], ["vasprun-band", "vasprun-dos"])
        self.assertEqual(set(band.projections), {"A:px"})
        self.assertEqual(set(dos.projections), {"B:d"})
        self.assertTrue((folder / ".lyra-cache" / "band-projection-v1" / "index.json").is_file())
        self.assertTrue((folder / ".lyra-cache" / "dos-projection-v1" / "index.json").is_file())

        disk_events: list[dict] = []
        reloaded = self._service(calls)
        reloaded.load_band(folder, "full", (ProjectionChannel("B"),), disk_events.append)
        reloaded.load_dos(folder, "full", (ProjectionChannel("A", "s"),), disk_events.append)
        self.assertEqual([name for name, _folder in calls], ["vasprun-band", "vasprun-dos"])
        self.assertEqual([event["phase"] for event in disk_events], [
            "validate", "cache-hit", "validate", "cache-hit"
        ])

    def test_full_identities_invalidate_on_vasprun_and_band_kpoints_changes(self):
        folder = self.root / "full"
        self._write_sources(folder, "vasprun.xml", "KPOINTS")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)
        channel = (ProjectionChannel("A", "s"),)
        service.load_band(folder, "full", channel, lambda _event: None)
        service.load_dos(folder, "full", channel, lambda _event: None)

        with (folder / "KPOINTS").open("a", encoding="utf-8") as handle:
            handle.write("changed")
        service.load_band(folder, "full", channel, lambda _event: None)
        self.assertEqual([name for name, _folder in calls].count("vasprun-band"), 2)
        self.assertEqual([name for name, _folder in calls].count("vasprun-dos"), 1)

        with (folder / "vasprun.xml").open("a", encoding="utf-8") as handle:
            handle.write("changed")
        service.load_band(folder, "full", channel, lambda _event: None)
        service.load_dos(folder, "full", channel, lambda _event: None)
        self.assertEqual([name for name, _folder in calls].count("vasprun-band"), 3)
        self.assertEqual([name for name, _folder in calls].count("vasprun-dos"), 2)

    def test_combined_sources_resolve_band_and_dos_folders_independently(self):
        band_folder = self.root / "band-full"
        dos_folder = self.root / "dos-full"
        self._write_sources(band_folder, "vasprun.xml", "KPOINTS")
        self._write_sources(dos_folder, "vasprun.xml")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)

        band = service.load_band(band_folder, "full", (), lambda _event: None)
        dos = service.load_dos(dos_folder, "full", (), lambda _event: None)

        self.assertEqual(calls, [
            ("vasprun-band", band_folder.resolve()),
            ("vasprun-dos", dos_folder.resolve()),
        ])
        self.assertEqual(band.folder, band_folder.resolve())
        self.assertEqual(dos.folder, dos_folder.resolve())
        self.assertEqual(dos.efermi, 5.0)

    def test_clear_removes_memory_and_folder_entries(self):
        folder = self.root / "band"
        self._write_sources(folder, "EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR")
        calls: list[tuple[str, Path]] = []
        service = self._service(calls)
        channel = (ProjectionChannel("A", "s"),)
        service.load_band(folder, "lite", channel, lambda _event: None)

        service.clear(folder)
        self.assertFalse((folder / ".lyra-cache").exists())
        service.load_band(folder, "lite", channel, lambda _event: None)

        self.assertEqual([name for name, _folder in calls].count("procar"), 2)

    def test_full_base_only_band_and_dos_persist_across_restart(self):
        band_folder = self.root / "base-band"
        dos_folder = self.root / "base-dos"
        self._write_sources(band_folder, "vasprun.xml", "KPOINTS")
        self._write_sources(dos_folder, "vasprun.xml")
        calls: list[tuple[str, Path]] = []

        def load_band(folder):
            folder = Path(folder).resolve()
            calls.append(("band", folder))
            return replace(
                parsed_band_data(),
                folder=folder,
                projections={},
                orbitals=(),
            )

        def load_dos(folder):
            folder = Path(folder).resolve()
            calls.append(("dos", folder))
            return replace(
                parsed_dos_data(),
                folder=folder,
                projections={},
                orbitals=(),
            )

        def service():
            return ProjectionDataService(
                cache=ProjectionCache(),
                vasprun_band_loader=load_band,
                vasprun_dos_loader=load_dos,
            )

        first = service()
        band = first.load_band(band_folder, "full", (), lambda _event: None)
        dos = first.load_dos(dos_folder, "full", (), lambda _event: None)

        self.assertEqual(band.projections, {})
        self.assertEqual(dos.projections, {})
        self.assertEqual(calls, [
            ("band", band_folder.resolve()),
            ("dos", dos_folder.resolve()),
        ])

        restarted = service()
        disk_events: list[dict] = []
        restarted.load_band(band_folder, "full", (), disk_events.append)
        restarted.load_dos(dos_folder, "full", (), disk_events.append)

        self.assertEqual(calls, [
            ("band", band_folder.resolve()),
            ("dos", dos_folder.resolve()),
        ])
        self.assertEqual(
            [event["phase"] for event in disk_events],
            ["validate", "cache-hit", "validate", "cache-hit"],
        )

        with self.assertRaisesRegex(ValueError, "projection.*unavailable"):
            restarted.load_band(
                band_folder,
                "full",
                (ProjectionChannel("A", "s"),),
                lambda _event: None,
            )
        with self.assertRaisesRegex(ValueError, "projection.*unavailable"):
            restarted.load_dos(
                dos_folder,
                "full",
                (ProjectionChannel("A", "s"),),
                lambda _event: None,
            )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

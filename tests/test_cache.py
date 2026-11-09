from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lyra.cache import ProjectionCache
from lyra.projection_store import BandProjectionStore, DosProjectionStore
from lyra.models import ParsedStructure
from tests.test_projection_store import parsed_band_data, parsed_dos_data


def cache_identity(timestamp: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "parser_version": "0.1.0",
        "read_mode": "lite",
        "data_kind": "band",
        "sources": [
            {"filename": "EIGENVAL", "size": 101, "mtime_ns": timestamp},
            {"filename": "PROCAR", "size": 202, "mtime_ns": timestamp},
        ],
    }


def dos_cache_identity(timestamp: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "parser_version": "0.1.0",
        "read_mode": "full",
        "data_kind": "dos",
        "sources": [
            {"filename": "vasprun.xml", "size": 303, "mtime_ns": timestamp},
        ],
    }


class ProjectionCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_directory.name) / "calculation"
        self.folder.mkdir()
        self.cache = ProjectionCache()
        self.base_data = parsed_band_data()
        self.store = BandProjectionStore.from_parsed(self.base_data)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def active_generation_directory(self) -> Path:
        namespace = self.folder / ".lyra-cache" / "band-projection-v1"
        index = json.loads((namespace / "index.json").read_text(encoding="utf-8"))
        return namespace / "generations" / index["generation"]

    def store_with_offset(self, offset: float) -> BandProjectionStore:
        return BandProjectionStore(
            self.store.spin_names,
            self.store.elements,
            self.store.orbitals,
            np.ascontiguousarray(self.store.tensor + offset),
        )

    def test_save_load_round_trip_uses_folder_local_numpy_and_json_files(self):
        identity = cache_identity()
        occupations = {
            "up": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "down": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        }
        base_data = replace(self.base_data, occupations=occupations)
        self.cache.save_band(self.folder, identity, base_data, self.store)

        cache_directory = self.active_generation_directory()
        namespace = cache_directory.parent.parent
        self.assertTrue((namespace / "index.json").is_file())
        self.assertTrue((cache_directory / "manifest.json").is_file())
        self.assertTrue((cache_directory / "projections.npy").is_file())
        self.assertTrue((cache_directory / "energies.npz").is_file())
        loaded = self.cache.load_band(self.folder, identity)
        self.assertIsNotNone(loaded)
        loaded_data, loaded_store = loaded
        self.assertNotIsInstance(loaded_store.tensor, np.memmap)
        self.assertTrue(loaded_store.tensor.flags.c_contiguous)
        self.assertEqual(loaded_data.folder, self.folder)
        self.assertEqual(loaded_data.segment_ranges, base_data.segment_ranges)
        self.assertEqual(loaded_data.tick_labels, base_data.tick_labels)
        self.assertEqual(loaded_data.spin_kind, base_data.spin_kind)
        np.testing.assert_array_equal(loaded_data.kpoints, self.base_data.kpoints)
        np.testing.assert_array_equal(loaded_data.distances, base_data.distances)
        np.testing.assert_array_equal(loaded_data.energies["up"], base_data.energies["up"])
        np.testing.assert_array_equal(loaded_data.occupations["down"], occupations["down"])
        np.testing.assert_array_equal(loaded_store.tensor, self.store.tensor)

    def test_external_root_keeps_cache_out_of_calculation_tree(self):
        external = Path(self.temporary_directory.name) / "remote-cache"
        cache = ProjectionCache(root=external)

        cache.save_band(self.folder, cache_identity(), self.base_data, self.store)

        self.assertFalse((self.folder / ".lyra-cache").exists())
        self.assertTrue((external / "band-projection-v1" / "index.json").is_file())
        loaded = cache.load_band(self.folder, cache_identity())
        self.assertIsNotNone(loaded)

    def test_band_cache_round_trip_preserves_structure_evidence(self):
        structure = ParsedStructure(
            np.eye(3),
            ("A", "B"),
            np.array(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))),
            "POSCAR",
        )
        data = replace(self.base_data, structure=structure)

        self.cache.save_band(self.folder, cache_identity(), data, self.store)
        loaded, _store = self.cache.load_band(self.folder, cache_identity())

        self.assertEqual(loaded.structure.atom_elements, ("A", "B"))
        np.testing.assert_array_equal(loaded.structure.lattice, structure.lattice)
        np.testing.assert_array_equal(
            loaded.structure.fractional_positions,
            structure.fractional_positions,
        )

    def test_band_cache_round_trip_preserves_soc_spin_components(self):
        components = np.ascontiguousarray(
            np.stack(
                [
                    self.store.tensor[0] * 0.1,
                    self.store.tensor[0] * 0.2,
                    self.store.tensor[0] * 0.3,
                ]
            ).astype(np.float32)
        )
        store = BandProjectionStore(
            self.store.spin_names,
            self.store.elements,
            self.store.orbitals,
            self.store.tensor,
            spin_components=components,
        )

        self.cache.save_band(self.folder, cache_identity(), self.base_data, store)
        loaded = self.cache.load_band(self.folder, cache_identity())

        self.assertIsNotNone(loaded)
        loaded_store = loaded[1]
        np.testing.assert_array_equal(loaded_store.spin_components, components)

    def test_source_timestamp_change_is_a_cache_miss(self):
        self.cache.save_band(self.folder, cache_identity(), self.base_data, self.store)

        self.assertIsNone(self.cache.load_band(self.folder, cache_identity(timestamp=2)))

    def test_invalid_identity_is_rejected_before_loading_or_saving(self):
        invalid_identity = {}

        with self.assertRaisesRegex(ValueError, "schema_version"):
            self.cache.load_band(self.folder, invalid_identity)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            self.cache.save_band(self.folder, invalid_identity, self.base_data, self.store)
        self.assertFalse((self.folder / ".lyra-cache").exists())

    def test_identity_rejects_malformed_source_records_with_value_error(self):
        invalid_identity = cache_identity()
        invalid_identity["sources"] = [{"filename": 3, "size": 1, "mtime_ns": 2}]

        with self.assertRaisesRegex(ValueError, "filename"):
            self.cache.load_band(self.folder, invalid_identity)
        self.assertFalse((self.folder / ".lyra-cache").exists())

    def test_identity_rejects_non_mapping_source_records_with_value_error(self):
        invalid_identity = cache_identity()
        invalid_identity["sources"] = [3]

        with self.assertRaisesRegex(ValueError, "source records"):
            self.cache.load_band(self.folder, invalid_identity)
        self.assertFalse((self.folder / ".lyra-cache").exists())

    def test_object_dtype_payload_is_rejected_before_numpy_writes(self):
        object_metadata = replace(
            self.base_data,
            kpoints=np.array([["not", "numeric", "metadata"]], dtype=object),
        )

        with (
            patch("lyra.cache.np.save") as save,
            patch("lyra.cache.np.savez") as savez,
            self.assertRaisesRegex(ValueError, "object dtype"),
        ):
            self.cache.save_band(self.folder, cache_identity(), object_metadata, self.store)

        save.assert_not_called()
        savez.assert_not_called()

    def test_schema_change_is_a_cache_miss(self):
        self.cache.save_band(self.folder, cache_identity(), self.base_data, self.store)
        incompatible_identity = copy.deepcopy(cache_identity())
        incompatible_identity["schema_version"] = 2

        self.assertIsNone(self.cache.load_band(self.folder, incompatible_identity))

    def test_corrupt_manifest_recovers_as_a_cache_miss(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        manifest = self.active_generation_directory() / "manifest.json"
        manifest.write_text("{not valid json", encoding="utf-8")

        self.assertIsNone(self.cache.load_band(self.folder, identity))

    def test_band_status_distinguishes_missing_and_stale_identity(self):
        identity = cache_identity()

        self.assertEqual(self.cache.band_status(self.folder, identity), "missing")

        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        self.assertEqual(self.cache.band_status(self.folder, identity), "valid")

        stale_identity = cache_identity(timestamp=999)
        self.assertEqual(self.cache.band_status(self.folder, stale_identity), "stale")

    def test_clearing_band_namespace_preserves_dos_cache(self):
        band_identity = cache_identity()
        dos_identity = dos_cache_identity()
        dos_data = parsed_dos_data()
        dos_store = DosProjectionStore.from_parsed(dos_data)
        self.cache.save_band(self.folder, band_identity, self.base_data, self.store)
        self.cache.save_dos(self.folder, dos_identity, dos_data, dos_store)

        self.cache.clear(self.folder, "band")

        self.assertEqual(self.cache.band_status(self.folder, band_identity), "missing")
        self.assertEqual(self.cache.dos_status(self.folder, dos_identity), "valid")

    def test_clearing_a_namespace_also_removes_disposable_previews(self):
        band_identity = cache_identity()
        dos_identity = dos_cache_identity()
        dos_data = parsed_dos_data()
        dos_store = DosProjectionStore.from_parsed(dos_data)
        self.cache.save_band(self.folder, band_identity, self.base_data, self.store)
        self.cache.save_dos(self.folder, dos_identity, dos_data, dos_store)
        preview = self.folder / ".lyra-cache" / "previews" / "lyra_preview_band.png"
        preview.parent.mkdir(parents=True)
        preview.write_bytes(b"preview")

        self.cache.clear(self.folder, "band")

        self.assertFalse(preview.parent.exists())
        self.assertEqual(self.cache.dos_status(self.folder, dos_identity), "valid")

    def test_corrupt_metadata_recovers_as_a_cache_miss(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        metadata = self.active_generation_directory() / "energies.npz"
        metadata.write_bytes(b"PK\x03\x04")

        self.assertIsNone(self.cache.load_band(self.folder, identity))

    def test_save_cleans_interrupted_temporary_directories(self):
        cache_root = self.folder / ".lyra-cache" / "band-projection-v1" / "generations"
        interrupted = cache_root / ".tmp-interrupted"
        interrupted.mkdir(parents=True)
        (interrupted / "projections.npy").write_bytes(b"partial")

        self.cache.save_band(self.folder, cache_identity(), self.base_data, self.store)

        self.assertFalse(interrupted.exists())
        self.assertEqual(list(cache_root.glob(".tmp-*")), [])
        self.assertIsNotNone(self.cache.load_band(self.folder, cache_identity()))

    def test_missing_or_malformed_index_is_a_cache_miss_and_can_be_rebuilt(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        index = self.folder / ".lyra-cache" / "band-projection-v1" / "index.json"
        index.unlink()

        self.assertIsNone(self.cache.load_band(self.folder, identity))
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        index.write_text("not json", encoding="utf-8")

        self.assertIsNone(self.cache.load_band(self.folder, identity))

    def test_successful_rebuild_cleans_unreferenced_generations(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        self.cache.save_band(self.folder, identity, self.base_data, self.store_with_offset(1.0))
        generations = self.folder / ".lyra-cache" / "band-projection-v1" / "generations"

        self.assertEqual(
            [path.name for path in generations.iterdir() if path.is_dir()],
            [self.active_generation_directory().name],
        )

    def test_interruption_before_index_promotion_keeps_old_cache_loadable(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        original_replace = os.replace

        def interrupt_generation_promotion(source, destination):
            if Path(destination).parent.name == "generations":
                raise OSError("interrupted before index promotion")
            return original_replace(source, destination)

        with patch("lyra.cache.os.replace", side_effect=interrupt_generation_promotion):
            with self.assertRaisesRegex(OSError, "before index promotion"):
                self.cache.save_band(self.folder, identity, self.base_data, self.store_with_offset(1.0))

        loaded = self.cache.load_band(self.folder, identity)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[1].tensor, self.store.tensor)

    def test_interruption_during_index_promotion_keeps_old_cache_loadable(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        original_replace = os.replace

        def interrupt_index_promotion(source, destination):
            if Path(destination).name == "index.json":
                raise OSError("interrupted during index promotion")
            return original_replace(source, destination)

        with patch("lyra.cache.os.replace", side_effect=interrupt_index_promotion):
            with self.assertRaisesRegex(OSError, "during index promotion"):
                self.cache.save_band(self.folder, identity, self.base_data, self.store_with_offset(1.0))

        loaded = self.cache.load_band(self.folder, identity)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[1].tensor, self.store.tensor)

    def test_clear_removes_the_folder_local_cache(self):
        self.cache.save_band(self.folder, cache_identity(), self.base_data, self.store)

        self.cache.clear(self.folder)

        self.assertFalse((self.folder / ".lyra-cache").exists())
        self.assertIsNone(self.cache.load_band(self.folder, cache_identity()))

    def test_loaded_cache_can_be_cleared_and_replaced_immediately(self):
        identity = cache_identity()
        self.cache.save_band(self.folder, identity, self.base_data, self.store)
        loaded = self.cache.load_band(self.folder, identity)

        self.assertIsNotNone(loaded)
        self.assertNotIsInstance(loaded[1].tensor, np.memmap)
        self.cache.clear(self.folder)
        self.cache.save_band(self.folder, identity, self.base_data, self.store)

        self.assertIsNotNone(self.cache.load_band(self.folder, identity))

    def test_dos_round_trip_uses_a_separate_normalized_namespace(self):
        data = parsed_dos_data()
        store = DosProjectionStore.from_parsed(data)
        identity = dos_cache_identity()

        self.cache.save_dos(self.folder, identity, data, store)
        loaded = self.cache.load_dos(self.folder, identity)

        self.assertIsNotNone(loaded)
        loaded_data, loaded_store = loaded
        self.assertTrue(
            (self.folder / ".lyra-cache" / "dos-projection-v1" / "index.json").is_file()
        )
        self.assertFalse(
            (self.folder / ".lyra-cache" / "band-projection-v1" / "index.json").exists()
        )
        self.assertEqual(loaded_data.spin_kind, data.spin_kind)
        self.assertEqual(loaded_data.elements, data.elements)
        np.testing.assert_array_equal(loaded_data.energies, data.energies)
        np.testing.assert_array_equal(loaded_data.densities["down"], data.densities["down"])
        np.testing.assert_array_equal(loaded_store.tensor, store.tensor)

    def test_cache_methods_reject_an_identity_for_the_other_data_kind(self):
        dos_data = parsed_dos_data()
        dos_store = DosProjectionStore.from_parsed(dos_data)

        with self.assertRaisesRegex(ValueError, "data_kind"):
            self.cache.load_band(self.folder, dos_cache_identity())
        with self.assertRaisesRegex(ValueError, "data_kind"):
            self.cache.save_dos(
                self.folder, cache_identity(), dos_data, dos_store
            )

    def test_base_only_band_round_trip_omits_projection_payload(self):
        identity = cache_identity()
        data = replace(self.base_data, projections={}, orbitals=())

        self.cache.save_band(self.folder, identity, data, None)
        loaded = self.cache.load_band(self.folder, identity)

        self.assertIsNotNone(loaded)
        loaded_data, loaded_store = loaded
        self.assertIsNone(loaded_store)
        self.assertEqual(loaded_data.elements, data.elements)
        self.assertEqual(loaded_data.orbitals, ())
        np.testing.assert_array_equal(loaded_data.energies["up"], data.energies["up"])
        self.assertFalse((self.active_generation_directory() / "projections.npy").exists())

    def test_base_only_dos_round_trip_omits_projection_payload(self):
        identity = dos_cache_identity()
        data = replace(parsed_dos_data(), projections={}, orbitals=())

        self.cache.save_dos(self.folder, identity, data, None)
        loaded = self.cache.load_dos(self.folder, identity)

        self.assertIsNotNone(loaded)
        loaded_data, loaded_store = loaded
        self.assertIsNone(loaded_store)
        self.assertEqual(loaded_data.elements, data.elements)
        self.assertEqual(loaded_data.orbitals, ())
        np.testing.assert_array_equal(loaded_data.densities["down"], data.densities["down"])
        namespace = self.folder / ".lyra-cache" / "dos-projection-v1"
        index = json.loads((namespace / "index.json").read_text(encoding="utf-8"))
        generation = namespace / "generations" / index["generation"]
        self.assertFalse((generation / "projections.npy").exists())


if __name__ == "__main__":
    unittest.main()

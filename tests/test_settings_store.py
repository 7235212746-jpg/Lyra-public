import json
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from queue import Empty
from unittest.mock import patch

import lyra.settings_store as settings_store_module
from lyra.settings_store import SettingsStore


def _load_namespace(store: SettingsStore, name: str) -> dict:
    method = getattr(store, "load_namespace", None)
    if method is None:
        raise AssertionError("SettingsStore.load_namespace is required")
    return method(name)


def _save_namespace(
    store: SettingsStore, name: str, settings: dict[str, object]
) -> None:
    method = getattr(store, "save_namespace", None)
    if method is None:
        raise AssertionError("SettingsStore.save_namespace is required")
    method(name, settings)


def _spawn_namespace_writer(path, namespace, settings, start, ready, results):
    try:
        store = SettingsStore(Path(path))
        ready.put(namespace)
        if not start.wait(5):
            raise TimeoutError("writer start was not released")
        _save_namespace(store, namespace, settings)
    except BaseException as exc:
        results.put((namespace, f"{type(exc).__name__}: {exc}"))
    else:
        results.put((namespace, None))


def _spawn_os_lock_holder(path, acquired, release, results):
    try:
        with settings_store_module._interprocess_lock(Path(path)):
            acquired.set()
            if not release.wait(10):
                raise TimeoutError("lock holder was not released")
    except BaseException as exc:
        results.put(f"{type(exc).__name__}: {exc}")
    else:
        results.put(None)


class SettingsStoreTests(unittest.TestCase):
    def _assert_spawn_writer_waits_for_os_lock(
        self, holder_path: Path, writer_path: Path
    ) -> None:
        context = multiprocessing.get_context("spawn")
        acquired = context.Event()
        release = context.Event()
        start = context.Event()
        ready = context.Queue()
        holder_results = context.Queue()
        writer_results = context.Queue()
        holder = context.Process(
            target=_spawn_os_lock_holder,
            args=(str(holder_path), acquired, release, holder_results),
        )
        writer = context.Process(
            target=_spawn_namespace_writer,
            args=(
                str(writer_path),
                "qt",
                {"window": {"width": 1600}},
                start,
                ready,
                writer_results,
            ),
        )
        try:
            holder.start()
            self.assertTrue(acquired.wait(10), "holder did not acquire OS lock")
            writer.start()
            self.assertEqual(ready.get(timeout=10), "qt")
            start.set()

            with self.assertRaises(Empty):
                writer_results.get(timeout=1)
            self.assertTrue(writer.is_alive())

            release.set()
            holder.join(timeout=10)
            writer.join(timeout=10)
            self.assertFalse(holder.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertEqual(holder_results.get(timeout=5), None)
            self.assertEqual(writer_results.get(timeout=5), ("qt", None))
        finally:
            start.set()
            release.set()
            for process in (holder, writer):
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            for queue in (ready, holder_results, writer_results):
                queue.close()

    def test_round_trip_preserves_unicode_paths_and_nested_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Lyra" / "settings.json"
            settings = {
                "band": {
                    "folder": r"C:\Users\demo\Desktop\实验数据",
                    "channels": [
                        {
                            "enabled": True,
                            "element": "Ta",
                            "orbital": "d",
                            "colour": "#3974d8",
                        }
                    ],
                }
            }

            store = SettingsStore(path)
            store.save(settings)

            self.assertEqual(store.load(), settings)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["namespaces"]["classic"], settings)
            self.assertEqual(document["namespaces"]["qt"], {})
            self.assertEqual(document["namespaces"]["shared"], {})

    def test_v1_loads_classic_and_empty_qt_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            original = (
                b'{"schema_version": 1, "settings": '
                b'{"theme": {"accent": "#3974d8"}, "rows": [1, 2]}}\r\n'
            )
            path.write_bytes(original)
            store = SettingsStore(path)

            classic = _load_namespace(store, "classic")
            qt = _load_namespace(store, "qt")
            classic["theme"]["accent"] = "changed"

            self.assertEqual(qt, {})
            self.assertEqual(
                _load_namespace(store, "classic")["theme"]["accent"],
                "#3974d8",
            )
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("settings.json.v1.*.bak")), [])

    def test_qt_first_save_migrates_complete_v1_settings_to_classic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            classic = {
                "band": {"folder": "C:/classic", "channels": [{"orbital": "d"}]},
                "export": {"formats": ["png", "pdf"]},
            }
            original = json.dumps(
                {
                    "schema_version": 1,
                    "settings": classic,
                    "top_level_extension": {"owner": "classic-plugin"},
                },
                ensure_ascii=False,
            ).encode("utf-8")
            path.write_bytes(original)

            _save_namespace(
                SettingsStore(path),
                "qt",
                {"window": {"width": 1280}},
            )

            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["namespaces"]["classic"], classic)
            self.assertEqual(document["namespaces"]["qt"], {"window": {"width": 1280}})
            self.assertEqual(document["namespaces"]["shared"], {})
            self.assertEqual(
                document["top_level_extension"], {"owner": "classic-plugin"}
            )
            backups = list(path.parent.glob("settings.json.v1.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_failed_v1_migration_retries_reuse_one_byte_identical_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            original = b'{"schema_version":1,"settings":{"classic":true}}\n'
            path.write_bytes(original)
            store = SettingsStore(path)

            with patch(
                "lyra.settings_store.os.replace",
                side_effect=OSError("replace blocked"),
            ):
                for _ in range(2):
                    with self.assertRaisesRegex(OSError, "replace blocked"):
                        _save_namespace(store, "qt", {"attempted": True})

            backups = list(path.parent.glob("settings.json.v1.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("settings.json.*.tmp")), [])

    def test_unrelated_historical_backup_does_not_block_v1_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            original = b'{"schema_version":1,"settings":{"folder":"current"}}\n'
            historical_bytes = (
                b'{"schema_version":1,"settings":{"folder":"historical"}}\n'
            )
            historical = path.with_name(
                "settings.json.v1.20000101T000000.000000Z.historical.bak"
            )
            path.write_bytes(original)
            historical.write_bytes(historical_bytes)

            SettingsStore(path).save_namespace("qt", {"window": {"width": 900}})

            backups = list(path.parent.glob("settings.json.v1.*.bak"))
            self.assertEqual(historical.read_bytes(), historical_bytes)
            self.assertEqual(
                sum(backup.read_bytes() == original for backup in backups),
                1,
            )
            self.assertEqual(
                SettingsStore(path).load_namespace("classic"),
                {"folder": "current"},
            )
            self.assertEqual(
                SettingsStore(path).load_namespace("qt"),
                {"window": {"width": 900}},
            )

    def test_alternating_stores_preserve_all_unowned_and_unknown_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "top_level_extension": {"enabled": True},
                        "namespaces": {
                            "classic": {
                                "band": {
                                    "folder": "C:/old",
                                    "extension": {"keep": 1, "replace": "old"},
                                },
                                "classic_only": True,
                            },
                            "qt": {"window": {"width": 900}},
                            "shared": {"cache": {"root": "C:/cache"}},
                            "plugin": {"opaque": {"value": 7}},
                        },
                    }
                ),
                encoding="utf-8",
            )

            _save_namespace(
                SettingsStore(path),
                "classic",
                {
                    "band": {
                        "folder": "C:/new",
                        "extension": {"replace": "new"},
                    }
                },
            )
            _save_namespace(
                SettingsStore(path),
                "qt",
                {"window": {"height": 700}, "recent": ["A", "B"]},
            )

            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["top_level_extension"], {"enabled": True})
            self.assertEqual(
                document["namespaces"]["classic"],
                {
                    "band": {
                        "folder": "C:/new",
                        "extension": {"keep": 1, "replace": "new"},
                    },
                    "classic_only": True,
                },
            )
            self.assertEqual(
                document["namespaces"]["qt"],
                {"window": {"width": 900, "height": 700}, "recent": ["A", "B"]},
            )
            self.assertEqual(
                document["namespaces"]["shared"], {"cache": {"root": "C:/cache"}}
            )
            self.assertEqual(
                document["namespaces"]["plugin"], {"opaque": {"value": 7}}
            )

    def test_spawn_writers_retain_classic_and_qt_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            ready = context.Queue()
            results = context.Queue()
            writers = [
                context.Process(
                    target=_spawn_namespace_writer,
                    args=(
                        str(path),
                        namespace,
                        settings,
                        start,
                        ready,
                        results,
                    ),
                )
                for namespace, settings in (
                    ("classic", {"band": {"folder": "C:/classic"}}),
                    ("qt", {"window": {"width": 1440}}),
                )
            ]
            try:
                for writer in writers:
                    writer.start()
                self.assertEqual(
                    {ready.get(timeout=10) for _ in writers},
                    {"classic", "qt"},
                )
                start.set()
                for writer in writers:
                    writer.join(timeout=10)
                self.assertTrue(all(not writer.is_alive() for writer in writers))
                outcomes = [results.get(timeout=5) for _ in writers]
            finally:
                start.set()
                for writer in writers:
                    if writer.is_alive():
                        writer.terminate()
                    writer.join(timeout=5)
                ready.close()
                results.close()

            self.assertEqual(dict(outcomes), {"classic": None, "qt": None})
            self.assertEqual(
                _load_namespace(SettingsStore(path), "classic"),
                {"band": {"folder": "C:/classic"}},
            )
            self.assertEqual(
                _load_namespace(SettingsStore(path), "qt"),
                {"window": {"width": 1440}},
            )

    def test_spawn_writer_waits_until_os_transaction_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"

            self._assert_spawn_writer_waits_for_os_lock(path, path)

            self.assertEqual(
                SettingsStore(path).load_namespace("qt"),
                {"window": {"width": 1600}},
            )

    def test_posix_sidecar_path_uses_canonical_settings_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "child"
            child.mkdir()
            canonical = root / "settings.json"
            alias = child / ".." / "settings.json"
            lock_path = settings_store_module._posix_lock_path(alias)

            self.assertEqual(
                lock_path,
                canonical.resolve().with_name("settings.json.lock"),
            )

    def test_transaction_io_uses_resolved_target_without_changing_public_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "alias" / "settings.json"
            canonical = root / "canonical" / "settings.json"
            canonical.parent.mkdir()
            original = b'{"schema_version":1,"settings":{"seed":true}}\n'
            canonical.write_bytes(original)
            resolve_path = settings_store_module._resolved_path

            def resolve_alias(path: Path) -> Path:
                if Path(path) == public:
                    return canonical
                return resolve_path(Path(path))

            store = SettingsStore(public)
            with patch(
                "lyra.settings_store._resolved_path",
                side_effect=resolve_alias,
            ):
                self.assertEqual(store.load_namespace("classic"), {"seed": True})
                store.save_namespace("qt", {"window": {"width": 1800}})

            self.assertEqual(store.path, public)
            self.assertFalse(public.parent.exists())
            self.assertEqual(
                SettingsStore(canonical).load_namespace("classic"),
                {"seed": True},
            )
            self.assertEqual(
                SettingsStore(canonical).load_namespace("qt"),
                {"window": {"width": 1800}},
            )
            backups = list(canonical.parent.glob("settings.json.v1.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    @unittest.skipUnless(os.name == "posix", "requires POSIX flock and symlinks")
    def test_posix_file_alias_writer_waits_for_canonical_os_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "settings.json"
            alias = root / "settings-alias.json"
            SettingsStore(canonical).save_namespace("classic", {"seed": True})
            alias.symlink_to(canonical)

            self._assert_spawn_writer_waits_for_os_lock(canonical, alias)

            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), canonical.resolve())
            for path in (canonical, alias):
                with self.subTest(path=path):
                    store = SettingsStore(path)
                    self.assertEqual(
                        store.load_namespace("classic"), {"seed": True}
                    )
                    self.assertEqual(
                        store.load_namespace("qt"),
                        {"window": {"width": 1600}},
                    )

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlinks")
    def test_posix_missing_file_under_parent_alias_uses_canonical_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_parent = root / "canonical"
            alias_parent = root / "alias"
            canonical_parent.mkdir()
            alias_parent.symlink_to(canonical_parent, target_is_directory=True)
            canonical = canonical_parent / "settings.json"
            alias = alias_parent / "settings.json"

            SettingsStore(alias).save_namespace("classic", {"seed": True})

            self.assertTrue(alias_parent.is_symlink())
            self.assertEqual(alias.resolve(), canonical.resolve())
            for path in (canonical, alias):
                with self.subTest(path=path):
                    self.assertEqual(
                        SettingsStore(path).load_namespace("classic"),
                        {"seed": True},
                    )

    def test_load_returns_defaults_for_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{broken", encoding="utf-8")

            self.assertEqual(SettingsStore(path).load(), {})

    def test_load_returns_defaults_for_unsupported_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps({"schema_version": 99, "settings": {"dpi": "300"}}),
                encoding="utf-8",
            )

            self.assertEqual(SettingsStore(path).load(), {})

    def test_malformed_and_future_schema_saves_refuse_without_changing_bytes(self):
        documents = (
            b"{broken",
            b'{"schema_version": 99, "namespaces": {}}\n',
            b'{"schema_version": true, "settings": {}}\n',
            (
                b'{"schema_version": 2.0, "namespaces": '
                b'{"classic": {}, "qt": {}, "shared": {}}}\n'
            ),
            b'{"schema_version": 2, "namespaces": []}\n',
        )
        for original in documents:
            with self.subTest(original=original):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "settings.json"
                    path.write_bytes(original)

                    with self.assertRaisesRegex(ValueError, "settings"):
                        _save_namespace(SettingsStore(path), "classic", {"dpi": "600"})

                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(
                        list(path.parent.glob("settings.json.*.tmp")), []
                    )

    def test_duplicate_json_keys_load_defaults_and_save_refuses_prior_bytes(self):
        documents = (
            (
                b'{"schema_version": 99, "schema_version": 2, "namespaces": '
                b'{"classic": {}, "qt": {}, "shared": {}}}\n'
            ),
            (
                b'{"schema_version": 2, "namespaces": {"classic": '
                b'{"dpi": 300, "dpi": 600}, "qt": {}, "shared": {}}}\n'
            ),
        )
        for original in documents:
            with self.subTest(original=original):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "settings.json"
                    path.write_bytes(original)
                    store = SettingsStore(path)

                    self.assertEqual(store.load_namespace("classic"), {})
                    with self.assertRaisesRegex(ValueError, "settings"):
                        store.save_namespace("classic", {"dpi": 900})

                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(
                        list(path.parent.glob("settings.json.*.tmp")), []
                    )

    def test_nonstandard_json_constants_load_defaults_and_refuse_save(self):
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(token=token):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "settings.json"
                    original = (
                        b'{"schema_version": 2, "namespaces": {"classic": '
                        b'{"value": '
                        + token
                        + b'}, "qt": {}, "shared": {}}}\n'
                    )
                    path.write_bytes(original)
                    store = SettingsStore(path)

                    self.assertEqual(store.load_namespace("classic"), {})
                    with self.assertRaisesRegex(ValueError, "settings"):
                        store.save_namespace("classic", {"value": 1.0})

                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(
                        list(path.parent.glob("settings.json.*.tmp")), []
                    )

    def test_save_rejects_incoming_nonfinite_numbers_without_rewriting(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "settings.json"
                    store = SettingsStore(path)
                    store.save_namespace("classic", {"dpi": 300})
                    original = path.read_bytes()

                    with self.assertRaises(ValueError):
                        store.save_namespace(
                            "classic", {"calculation": {"value": value}}
                        )

                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(path.parent.glob("settings.json.*.tmp")), [])

    def test_save_atomically_replaces_existing_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            store = SettingsStore(path)
            store.save({"dpi": "300"})

            with patch("lyra.settings_store.os.replace", wraps=os.replace) as replace:
                store.save({"dpi": "600"})

            replace.assert_called_once()
            self.assertEqual(store.load(), {"dpi": "600"})
            self.assertEqual(list(path.parent.glob("settings.json.*.tmp")), [])

    def test_replace_failure_preserves_original_bytes_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            store = SettingsStore(path)
            _save_namespace(store, "classic", {"dpi": "300"})
            original = path.read_bytes()

            with patch(
                "lyra.settings_store.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _save_namespace(store, "classic", {"dpi": "600"})

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("settings.json.*.tmp")), [])

    def test_default_path_honours_appdata(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"APPDATA": tmp}):
                store = SettingsStore.default()

            self.assertEqual(store.path, Path(tmp) / "Lyra" / "settings.json")


if __name__ == "__main__":
    unittest.main()

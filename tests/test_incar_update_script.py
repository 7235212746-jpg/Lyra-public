import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "update_incar_knowledge.py"


class IncarUpdateScriptTests(unittest.TestCase):
    def _module(self):
        spec = importlib.util.spec_from_file_location("update_incar_knowledge", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_fetch_index_follows_pagination_and_preserves_revision_metadata(self):
        module = self._module()
        responses = iter([
            {
                "continue": {"gcmcontinue": "next", "continue": "gcmcontinue||"},
                "query": {"pages": [{
                    "title": "ENCUT",
                    "canonicalurl": "https://vasp.at/wiki/ENCUT",
                    "revisions": [{"revid": 7, "timestamp": "2025-01-01T00:00:00Z"}],
                }]},
            },
            {
                "batchcomplete": True,
                "query": {"pages": [{
                    "title": "ML LMLFF",
                    "canonicalurl": "https://vasp.at/wiki/ML_LMLFF",
                    "revisions": [{"revid": 9, "timestamp": "2025-02-01T00:00:00Z"}],
                }]},
            },
        ])
        calls = []

        def fake_fetch(params):
            calls.append(dict(params))
            return next(responses)

        records = module.fetch_official_index(fake_fetch)

        self.assertEqual([item["tag"] for item in records], ["ENCUT", "ML_LMLFF"])
        self.assertEqual(records[0]["revision_id"], 7)
        self.assertNotIn("rvlimit", calls[0])
        self.assertEqual(calls[1]["gcmcontinue"], "next")

    def test_atomic_writer_rejects_implausibly_small_official_index(self):
        module = self._module()
        document = module.make_index_document(
            [{
                "tag": "ENCUT",
                "url": "https://vasp.at/wiki/ENCUT",
                "revision_id": 7,
                "revision_timestamp": "2025-01-01T00:00:00Z",
            }],
            fetched_at="2026-08-13T00:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "index.json"
            with self.assertRaisesRegex(ValueError, "600"):
                module.write_index_atomic(document, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

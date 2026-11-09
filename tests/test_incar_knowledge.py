import json
import tempfile
import unittest
from pathlib import Path

from lyra.incar import IncarKnowledgeBase


class IncarKnowledgeTests(unittest.TestCase):
    def test_bundled_dataset_includes_license_and_attribution(self):
        root = Path(__file__).parents[1] / "lyra" / "resources" / "incar"

        self.assertIn("GNU Free Documentation License", (root / "LICENSE-GFDL-1.2.txt").read_text(encoding="utf-8"))
        attribution = (root / "ATTRIBUTION.md").read_text(encoding="utf-8")
        self.assertIn("https://vasp.at/wiki/", attribution)
        self.assertIn("not an official VASP translation", attribution)

    def test_bundled_index_has_real_official_metadata_and_600_tags(self):
        knowledge = IncarKnowledgeBase.bundled()

        self.assertGreaterEqual(len(knowledge), 600)
        self.assertEqual(
            sum(record.has_knowledge_body for record in knowledge.records),
            len(knowledge),
        )
        self.assertEqual(knowledge.dataset.license_name, "GNU FDL 1.2")
        self.assertTrue(knowledge.dataset.fetched_at)
        self.assertIsNotNone(knowledge.get("AMIX_MAG"))
        self.assertIsNotNone(knowledge.get("EFIELD_PEAD"))
        self.assertIsNotNone(knowledge.get("KERNEL_TRUNCATION/FACTOR"))
        for record in knowledge.records:
            self.assertTrue(record.source.url.startswith("https://vasp.at/wiki/"))
            self.assertIsInstance(record.source.revision_id, int)
            self.assertGreater(record.source.revision_id, 0)
            self.assertTrue(record.source.revision_timestamp)
            self.assertEqual(record.source.license_name, "GNU FDL 1.2")
            self.assertIsNotNone(record.source.translation_basis_revision_id)
            self.assertTrue(record.source.translation_basis_revision_timestamp)

    def test_core_record_is_explicitly_nonofficial_chinese_content(self):
        record = IncarKnowledgeBase.bundled().get("encut")

        self.assertIsNotNone(record)
        self.assertTrue(record.has_knowledge_body)
        self.assertEqual(record.content_origin, "lyra_curated_translation")
        self.assertFalse(record.is_official_recommendation)
        self.assertIn("截断", record.summary_zh)
        self.assertIn("vasp.at/wiki/ENCUT", record.source.url)
        self.assertTrue(record.source.translation_curated_at)
        self.assertEqual(
            record.source.translation_basis_fetched_at,
            "2026-08-13T09:43:45Z",
        )

    def test_every_official_record_has_offline_chinese_body_and_source_page(self):
        knowledge = IncarKnowledgeBase.bundled()
        self.assertTrue(all(record.has_knowledge_body for record in knowledge.records))
        self.assertTrue(all(record.source.url for record in knowledge.records))

    def test_search_matches_tag_and_chinese_terms(self):
        knowledge = IncarKnowledgeBase.bundled()

        self.assertEqual(knowledge.search("ENCUT", limit=1)[0].tag, "ENCUT")
        self.assertIn(
            "ENCUT",
            {item.tag for item in knowledge.search("平面波截断", limit=10)},
        )

    def test_search_matches_enum_value_meanings(self):
        knowledge = IncarKnowledgeBase.bundled()

        self.assertIn(
            "ISIF",
            {item.tag for item in knowledge.search("只允许体积变化", limit=10)},
        )

    def test_scientific_limitations_from_official_pages_are_preserved(self):
        knowledge = IncarKnowledgeBase.bundled()

        self.assertIn("LDA 或 GGA", knowledge.require("GGA").summary_zh)
        self.assertIn("VASP 6.6.0", knowledge.require("ELPH_USERECIP").details_zh)
        self.assertIn("ISYM=-1", knowledge.require("KPUSE").details_zh)
        self.assertIn("LMAXMIX=6", knowledge.require("LDAU").details_zh)
        self.assertIn("软退出", knowledge.require("ML_LBASIS_DISCARD").details_zh)
        self.assertIn("没有简单的事后修正", knowledge.require("NELECT").details_zh)
        self.assertIn("IBRION=5-8", knowledge.require("VCA").details_zh)

    def test_isif_exposes_official_value_meanings_with_revision_anchor(self):
        record = IncarKnowledgeBase.bundled().require("ISIF")

        self.assertEqual(
            tuple(item.value for item in record.value_meanings),
            tuple(str(value) for value in range(9)),
        )
        self.assertIn("离子位置、晶胞形状和体积", record.meaning_for_value("3").summary_zh)
        self.assertIn("仅计算应力张量的迹", record.meaning_for_value("1").details_zh)
        self.assertIn("VASP 6.4.1", record.meaning_for_value("8").details_zh)
        self.assertEqual(record.source.revision_id, 34129)
        self.assertEqual(record.source.revision_timestamp, "2026-02-09T11:41:24Z")

    def test_loader_supports_extensible_value_meanings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {
                "schema_version": 1,
                "dataset": {
                    "source_url": "https://vasp.at/wiki/Category:INCAR_tag",
                    "fetched_at": "2026-08-13T00:00:00Z",
                    "license_name": "GNU FDL 1.2",
                    "license_url": "https://www.gnu.org/licenses/old-licenses/fdl-1.2.html",
                },
                "records": [{
                    "tag": "IBRION",
                    "url": "https://vasp.at/wiki/IBRION",
                    "revision_id": 456,
                    "revision_timestamp": "2026-06-01T00:00:00Z",
                }],
            }
            core = {
                "schema_version": 1,
                "curated_at": "2026-08-13T01:00:00Z",
                "based_on_index_fetched_at": "2026-08-13T00:00:00Z",
                "records": [{
                    "tag": "IBRION",
                    "summary_zh": "选择离子运动算法。",
                    "source_revision_id": 456,
                    "source_revision_timestamp": "2026-06-01T00:00:00Z",
                    "value_meanings": [
                        {"value": "-1", "summary_zh": "不更新离子。"},
                        {"value": "2", "summary_zh": "共轭梯度算法。"},
                    ],
                }],
            }
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            (root / "core_zh.json").write_text(json.dumps(core), encoding="utf-8")

            record = IncarKnowledgeBase.from_directory(root).require("IBRION")

        self.assertEqual(record.meaning_for_value("-1").summary_zh, "不更新离子。")
        self.assertEqual(record.meaning_for_value("2.0").summary_zh, "共轭梯度算法。")
        self.assertIsNone(record.meaning_for_value("99"))

    def test_loader_merges_core_content_without_losing_index_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {
                "schema_version": 1,
                "dataset": {
                    "source_url": "https://vasp.at/wiki/Category:INCAR_tag",
                    "fetched_at": "2026-08-13T00:00:00Z",
                    "license_name": "GNU FDL 1.2",
                    "license_url": "https://www.gnu.org/licenses/old-licenses/fdl-1.2.html",
                },
                "records": [{
                    "tag": "ENCUT",
                    "url": "https://vasp.at/wiki/ENCUT",
                    "revision_id": 123,
                    "revision_timestamp": "2026-01-01T00:00:00Z",
                }],
            }
            core = {
                "schema_version": 1,
                "curated_at": "2026-08-13T01:00:00Z",
                "based_on_index_fetched_at": "2026-08-13T00:00:00Z",
                "records": [{
                    "tag": "ENCUT",
                    "title_zh": "平面波截断能",
                    "summary_zh": "控制平面波基组的截断能。",
                    "details_zh": "Lyra 中文整理内容。",
                    "keywords_zh": ["截断能", "平面波"],
                }],
            }
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
            (root / "core_zh.json").write_text(json.dumps(core), encoding="utf-8")

            knowledge = IncarKnowledgeBase.from_directory(root)

        record = knowledge.require("ENCUT")
        self.assertEqual(record.source.revision_id, 123)
        self.assertEqual(record.title_zh, "平面波截断能")


if __name__ == "__main__":
    unittest.main()

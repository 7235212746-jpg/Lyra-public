import unittest

from lyra.incar import IncarAssistant, IncarKnowledgeBase


class IncarAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = IncarAssistant(IncarKnowledgeBase.bundled())

    def test_explains_every_assignment_and_reports_unknown_tags(self):
        analysis = self.assistant.analyse(
            "ENCUT = 520\nTOTALLY_UNKNOWN_TAG = 1\nENCUT = 600\n"
        )

        self.assertEqual(len(analysis.entries), 3)
        self.assertEqual(analysis.unknown_tags, ("TOTALLY_UNKNOWN_TAG",))
        self.assertTrue(analysis.entries[0].record.has_knowledge_body)
        self.assertIsNone(analysis.entries[1].record)
        self.assertFalse(analysis.entries[0].assignment.is_effective)
        self.assertTrue(analysis.entries[2].assignment.is_effective)
        self.assertIn("ENCUT", {item.tags[0] for item in analysis.findings})

    def test_reports_basic_conflicts_and_risks_as_lyra_rules(self):
        analysis = self.assistant.analyse(
            "ISPIN = 2\n"
            "LNONCOLLINEAR = .TRUE.\n"
            "LDAU = .TRUE.\n"
            "NSW = 10\n"
            "IBRION = -1\n"
            "ICHARG = 11\n"
        )
        codes = {item.code for item in analysis.findings}

        self.assertIn("noncollinear-ispin", codes)
        self.assertIn("ldau-parameters-missing", codes)
        self.assertIn("ionic-steps-disabled", codes)
        self.assertIn("fixed-charge-ionic-run", codes)
        self.assertTrue(all(item.origin == "lyra_auxiliary_rule" for item in analysis.findings))
        self.assertIn("不是 VASP 官方建议", analysis.disclaimer_zh)
        noncollinear = next(item for item in analysis.findings if item.code == "noncollinear-ispin")
        self.assertEqual(noncollinear.severity, "warning")
        self.assertIn("版本", noncollinear.message_zh)
        self.assertIn("MAGMOM", noncollinear.message_zh)

    def test_boolean_and_number_parsing_accepts_common_vasp_forms(self):
        analysis = self.assistant.analyse(
            "LDAU = T\nNSW = 0.0\nIBRION = -1\nICHARG = 11\n"
        )
        codes = {item.code for item in analysis.findings}

        self.assertIn("ldau-parameters-missing", codes)
        self.assertNotIn("ionic-steps-disabled", codes)
        self.assertNotIn("fixed-charge-ionic-run", codes)

    def test_analysis_resolves_the_meaning_of_the_assigned_enum_value(self):
        analysis = self.assistant.analyse("ISIF = 3\n")

        self.assertIsNotNone(analysis.entries[0].value_meaning)
        self.assertEqual(analysis.entries[0].value_meaning.value, "3")
        self.assertIn("晶胞形状", analysis.entries[0].value_meaning.summary_zh)


if __name__ == "__main__":
    unittest.main()

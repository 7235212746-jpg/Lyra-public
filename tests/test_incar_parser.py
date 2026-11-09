import unittest

from lyra.incar import parse_incar


class IncarParserTests(unittest.TestCase):
    def test_parses_comments_semicolons_quotes_and_case_insensitively(self):
        document = parse_incar(
            'system = "Fe # sample; phase"; encut = 520 ! trailing comment\n'
            'ISMEAR = -5 # tetrahedron\n'
        )

        self.assertEqual(
            [(item.tag, item.value) for item in document.assignments],
            [
                ("SYSTEM", '"Fe # sample; phase"'),
                ("ENCUT", "520"),
                ("ISMEAR", "-5"),
            ],
        )
        self.assertEqual(document.effective_value("encut"), "520")

    def test_supports_continuations_and_multiple_assignments(self):
        document = parse_incar(
            "MAGMOM = 2*5.0 \\\n"
            "         4*0.0\n"
            "LDAUL = 2 -1; LDAUU = 4.0 0.0; LDAUJ = 0 0\n"
        )

        self.assertEqual(document.effective_value("MAGMOM"), "2*5.0 4*0.0")
        self.assertEqual(document.effective_value("LDAUU"), "4.0 0.0")
        self.assertEqual(document.assignments[0].line_start, 1)
        self.assertEqual(document.assignments[0].line_end, 2)

    def test_supports_multiline_quoted_values_and_nested_tag_blocks(self):
        document = parse_incar(
            'WANNIER90_WIN = "begin projections\n'
            'Fe:d\n'
            'end projections"\n'
            'PLUGINS { STRUCTURE = T; INNER { MODE = fast } }\n'
        )

        self.assertIn("Fe:d", document.effective_value("WANNIER90_WIN"))
        self.assertEqual(document.effective_value("PLUGINS/STRUCTURE"), "T")
        self.assertEqual(document.effective_value("PLUGINS/INNER/MODE"), "fast")
        self.assertEqual(document.diagnostics, ())

    def test_ampersand_is_not_silently_treated_as_a_continuation(self):
        document = parse_incar("MAGMOM = 2*5.0 &\n4*0.0\n")

        self.assertEqual(document.effective_value("MAGMOM"), "2*5.0 &")
        self.assertGreaterEqual(len(document.diagnostics), 2)
        self.assertTrue(any("&" in item.message_zh for item in document.diagnostics))

    def test_preserves_duplicates_and_marks_last_occurrence_effective(self):
        document = parse_incar("ENCUT = 400\nencut = 520\nENCUT = 600\n")

        self.assertEqual(document.values("ENCUT"), ("400", "520", "600"))
        self.assertEqual(document.effective_value("ENCUT"), "600")
        self.assertEqual(document.duplicate_tags, ("ENCUT",))
        self.assertFalse(document.assignments[0].is_effective)
        self.assertTrue(document.assignments[-1].is_effective)

    def test_accepts_underscore_digit_tags_and_reports_malformed_fragments(self):
        document = parse_incar(
            "ML_LMLFF = .TRUE.; AMIN = 0.1\n"
            "KERNEL_TRUNCATION/FACTOR = 0.5\n"
            "PLUGINS/FORCE_AND_STRESS = enabled\n"
            "THIS IS NOT AN ASSIGNMENT\n"
            "EMPTY =\n"
        )

        self.assertEqual(document.effective_value("ML_LMLFF"), ".TRUE.")
        self.assertEqual(document.effective_value("AMIN"), "0.1")
        self.assertEqual(document.effective_value("KERNEL_TRUNCATION/FACTOR"), "0.5")
        self.assertEqual(document.effective_value("PLUGINS/FORCE_AND_STRESS"), "enabled")
        self.assertEqual(len(document.diagnostics), 2)
        self.assertEqual({item.line_start for item in document.diagnostics}, {4, 5})


if __name__ == "__main__":
    unittest.main()

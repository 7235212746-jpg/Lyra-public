import json
import tempfile
import unittest
from pathlib import Path

from lyra.check_cli import find_vasp_directories, main
from tests.fixtures import write_band_fixture, write_dos_fixture


class CheckCliTests(unittest.TestCase):
    def test_recursive_detection_uses_vasp_files_not_folder_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band = root / "anything" / "alpha"
            dos = root / "unrelated" / "beta"
            empty = root / "looks_like_VASP"
            band.mkdir(parents=True)
            dos.mkdir(parents=True)
            empty.mkdir()
            write_band_fixture(band)
            write_dos_fixture(dos)

            found = find_vasp_directories(root, recursive=True)

        self.assertEqual(found, (band.resolve(), dos.resolve()))

    def test_json_report_is_deterministic_and_cli_returns_problem_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calculation = root / "calc & one"
            calculation.mkdir()
            write_band_fixture(calculation)
            (calculation / "OUTCAR").write_text(
                "ZBRENT: fatal error in bracketing\n",
                encoding="utf-8",
            )
            report_path = root / "report.json"

            exit_code = main(
                [str(root), "--recursive", "--json", str(report_path)]
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["directories"][0]["path"], str(calculation.resolve()))
        self.assertEqual(
            payload["directories"][0]["issues"][0]["code"],
            "vasp-fatal-marker",
        )

    def test_unreadable_or_missing_root_returns_invocation_error(self):
        self.assertEqual(main(["Z:/definitely/missing/root"]), 2)


if __name__ == "__main__":
    unittest.main()

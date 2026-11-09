from pathlib import Path

import pytest

from lyra.incar import IncarKnowledgeBase
import lyra.incar_window as incar_window


resolve_incar_path = incar_window.resolve_incar_path


def test_resolve_incar_path_accepts_file_or_calculation_folder(tmp_path: Path) -> None:
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 520\n", encoding="utf-8")

    assert resolve_incar_path(incar) == incar.resolve()
    assert resolve_incar_path(tmp_path) == incar.resolve()


def test_resolve_incar_path_rejects_missing_and_oversized_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_incar_path(tmp_path)

    incar = tmp_path / "INCAR"
    incar.write_bytes(b"x" * 33)
    with pytest.raises(ValueError, match="too large"):
        resolve_incar_path(incar, max_bytes=32)


def test_record_detail_lists_all_enum_values_and_marks_the_selected_one() -> None:
    record = IncarKnowledgeBase.bundled().require("ISIF")

    assert hasattr(incar_window, "format_incar_record"), "枚举值详情格式化功能尚未实现"
    detail = incar_window.format_incar_record(record, selected_value="3")

    assert "具体取值（依据上述官方修订）" in detail
    assert "  ISIF = 2：" in detail
    assert "▶ ISIF = 3：" in detail
    assert "  ISIF = 8：" in detail
    assert "VASP 6.4.1" in detail


def test_record_detail_without_selection_still_lists_enum_values() -> None:
    record = IncarKnowledgeBase.bundled().require("ISIF")

    assert hasattr(incar_window, "format_incar_record"), "枚举值详情格式化功能尚未实现"
    detail = incar_window.format_incar_record(record)

    assert "ISIF = 0" in detail
    assert "ISIF = 8" in detail
    assert "▶" not in detail

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_incar_translations.py"


def module():
    spec = importlib.util.spec_from_file_location("merge_incar_translations", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def record(tag: str, revision: int = 7):
    return {
        "tag": tag,
        "title_zh": f"{tag} 标题",
        "summary_zh": f"{tag} 摘要",
        "details_zh": f"{tag} 详情",
        "category_zh": "测试",
        "keywords_zh": [tag],
        "source_revision_id": revision,
        "source_revision_timestamp": "2026-01-01T00:00:00Z",
    }


def test_merge_requires_complete_revision_anchored_coverage():
    index = {
        "dataset": {"fetched_at": "2026-01-02T00:00:00Z"},
        "records": [
            {"tag": tag, "revision_id": 7, "revision_timestamp": "2026-01-01T00:00:00Z"}
            for tag in ("A", "B")
        ],
    }
    core = {"records": [record("A")]}
    merged = module().merge_documents(
        index,
        core,
        [{"records": [record("B")]}],
        curated_at="2026-01-03T00:00:00Z",
    )

    assert [item["tag"] for item in merged["records"]] == ["A", "B"]
    assert merged["based_on_index_fetched_at"] == "2026-01-02T00:00:00Z"

    with pytest.raises(ValueError, match="incomplete"):
        module().merge_documents(index, core, [])


def test_merge_rejects_revision_mismatch():
    index = {
        "dataset": {"fetched_at": "2026-01-02T00:00:00Z"},
        "records": [{"tag": "A", "revision_id": 7, "revision_timestamp": "2026-01-01T00:00:00Z"}],
    }
    bad = record("A", revision=8)
    with pytest.raises(ValueError, match="revision mismatch"):
        module().merge_documents(index, {"records": [bad]}, [])

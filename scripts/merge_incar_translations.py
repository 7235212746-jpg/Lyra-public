"""Validate and merge reviewed INCAR translation batches into core_zh.json."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


TEXT_FIELDS = ("title_zh", "summary_zh", "details_zh", "category_zh")


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def merge_documents(
    index: dict,
    core: dict,
    batches: list[dict],
    *,
    require_complete: bool = True,
    curated_at: str | None = None,
) -> dict:
    index_records = index.get("records")
    if not isinstance(index_records, list):
        raise ValueError("INCAR index records are missing")
    official = {str(item["tag"]).upper(): item for item in index_records}
    if len(official) != len(index_records):
        raise ValueError("INCAR index contains duplicate tags")

    translated: dict[str, dict] = {}
    sources = [core, *batches]
    for source in sources:
        records = source.get("records")
        if not isinstance(records, list):
            raise ValueError("translation records are missing")
        for raw in records:
            if not isinstance(raw, dict):
                raise ValueError("translation record must be an object")
            tag = str(raw.get("tag", "")).strip().upper()
            reference = official.get(tag)
            if reference is None:
                raise ValueError(f"translation tag is absent from official index: {tag}")
            record = dict(raw)
            record["tag"] = tag
            revision_id = record.setdefault(
                "source_revision_id", reference["revision_id"]
            )
            revision_timestamp = record.setdefault(
                "source_revision_timestamp", reference["revision_timestamp"]
            )
            if (
                revision_id != reference["revision_id"]
                or revision_timestamp != reference["revision_timestamp"]
            ):
                raise ValueError(f"translation revision mismatch for {tag}")
            for field in TEXT_FIELDS:
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"translation field {field} is missing for {tag}")
                record[field] = record[field].strip()
            keywords = record.get("keywords_zh")
            if not isinstance(keywords, list) or not all(
                isinstance(value, str) and value.strip() for value in keywords
            ):
                raise ValueError(f"translation keywords are invalid for {tag}")
            record["keywords_zh"] = [value.strip() for value in keywords]
            if tag in translated:
                raise ValueError(f"duplicate translated INCAR tag: {tag}")
            translated[tag] = record

    missing = sorted(set(official).difference(translated))
    if require_complete and missing:
        raise ValueError(
            f"INCAR translation is incomplete: {len(missing)} tags remain; "
            f"first tags: {', '.join(missing[:10])}"
        )
    timestamp = curated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "content_origin": "lyra_curated_translation",
        "curated_at": timestamp,
        "based_on_index_fetched_at": index["dataset"]["fetched_at"],
        "translation_note_zh": core.get(
            "translation_note_zh",
            "中文内容由 Lyra 依据 VASP Wiki 整理，不代表官方翻译或参数建议。",
        ),
        "records": [translated[tag] for tag in sorted(translated)],
    }


def write_atomic(document: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1] / "lyra" / "resources" / "incar"
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=root / "index.json")
    parser.add_argument("--core", type=Path, default=root / "core_zh.json")
    parser.add_argument("--batches", type=Path, default=root / "translation_batches")
    parser.add_argument("--output", type=Path, default=root / "core_zh.json")
    args = parser.parse_args(argv)
    batch_paths = sorted(args.batches.glob("batch_*.json"))
    document = merge_documents(
        _read_object(args.index),
        _read_object(args.core),
        [_read_object(path) for path in batch_paths],
    )
    write_atomic(document, args.output)
    print(f"Merged {len(document['records'])} INCAR translations into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

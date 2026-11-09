from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import (
    DatasetMetadata,
    IncarKnowledgeRecord,
    IncarValueMeaning,
    SourceReference,
    _normalize_incar_value,
)


class IncarKnowledgeBase:
    def __init__(
        self,
        dataset: DatasetMetadata,
        records: Iterable[IncarKnowledgeRecord],
    ) -> None:
        self.dataset = dataset
        ordered = tuple(sorted(records, key=lambda item: item.tag))
        self.records = ordered
        self._by_tag = {item.tag: item for item in ordered}

    def __len__(self) -> int:
        return len(self.records)

    def get(self, tag: str) -> IncarKnowledgeRecord | None:
        return self._by_tag.get(tag.strip().upper())

    def require(self, tag: str) -> IncarKnowledgeRecord:
        record = self.get(tag)
        if record is None:
            raise KeyError(tag)
        return record

    def search(self, query: str, *, limit: int = 20) -> tuple[IncarKnowledgeRecord, ...]:
        needle = query.strip().casefold()
        if not needle or limit <= 0:
            return ()
        scored: list[tuple[int, str, IncarKnowledgeRecord]] = []
        for record in self.records:
            tag = record.tag.casefold()
            fields = (
                record.title_zh,
                record.summary_zh,
                record.details_zh,
                record.category_zh,
                *record.keywords_zh,
                *(item.summary_zh for item in record.value_meanings),
                *(item.details_zh for item in record.value_meanings),
            )
            haystack = " ".join(fields).casefold()
            if tag == needle:
                score = 1000
            elif tag.startswith(needle):
                score = 700
            elif needle in tag:
                score = 500
            elif needle in haystack:
                score = 300
            else:
                continue
            if record.has_knowledge_body:
                score += 20
            scored.append((-score, record.tag, record))
        scored.sort()
        return tuple(item[2] for item in scored[:limit])

    @classmethod
    def bundled(cls) -> "IncarKnowledgeBase":
        root = Path(__file__).resolve().parents[1] / "resources" / "incar"
        return cls.from_directory(root)

    @classmethod
    def from_directory(cls, root: str | Path) -> "IncarKnowledgeBase":
        directory = Path(root)
        index = _read_json(directory / "index.json")
        core = _read_json(directory / "core_zh.json")
        if index.get("schema_version") != 1 or core.get("schema_version") != 1:
            raise ValueError("Unsupported INCAR knowledge schema.")
        metadata_raw = index.get("dataset", {})
        dataset = DatasetMetadata(
            source_url=_required_string(metadata_raw, "source_url"),
            fetched_at=_required_string(metadata_raw, "fetched_at"),
            license_name=_required_string(metadata_raw, "license_name"),
            license_url=_required_string(metadata_raw, "license_url"),
        )
        core_by_tag = {
            _normalize_tag(item.get("tag")): item for item in core.get("records", [])
        }
        translation_curated_at = _required_string(core, "curated_at")
        translation_basis_fetched_at = _required_string(
            core, "based_on_index_fetched_at"
        )
        records: list[IncarKnowledgeRecord] = []
        seen: set[str] = set()
        for raw in index.get("records", []):
            tag = _normalize_tag(raw.get("tag"))
            if tag in seen:
                raise ValueError(f"Duplicate INCAR knowledge tag: {tag}")
            seen.add(tag)
            translated = core_by_tag.get(tag, {})
            translation_revision_id = translated.get("source_revision_id")
            translation_revision_timestamp = translated.get(
                "source_revision_timestamp", ""
            )
            if translation_revision_id is not None:
                translation_revision_id = _required_positive_int(
                    translated, "source_revision_id"
                )
                translation_revision_timestamp = _required_string(
                    translated, "source_revision_timestamp"
                )
            source = SourceReference(
                url=_required_string(raw, "url"),
                revision_id=_required_positive_int(raw, "revision_id"),
                revision_timestamp=_required_string(raw, "revision_timestamp"),
                fetched_at=dataset.fetched_at,
                license_name=dataset.license_name,
                license_url=dataset.license_url,
                translation_curated_at=(
                    translation_curated_at if translated else ""
                ),
                translation_basis_fetched_at=(
                    translation_basis_fetched_at if translated else ""
                ),
                translation_basis_revision_id=translation_revision_id,
                translation_basis_revision_timestamp=translation_revision_timestamp,
            )
            summary = _optional_string(translated, "summary_zh")
            details = _optional_string(translated, "details_zh")
            records.append(
                IncarKnowledgeRecord(
                    tag=tag,
                    source=source,
                    title_zh=_optional_string(translated, "title_zh"),
                    summary_zh=summary,
                    details_zh=details,
                    category_zh=_optional_string(translated, "category_zh"),
                    keywords_zh=tuple(
                        str(value).strip()
                        for value in translated.get("keywords_zh", [])
                        if str(value).strip()
                    ),
                    value_meanings=_load_value_meanings(translated),
                    content_origin=(
                        "lyra_curated_translation"
                        if summary or details
                        else "official_index_only"
                    ),
                    is_official_recommendation=False,
                )
            )
        if not records:
            raise ValueError("INCAR knowledge index is empty.")
        unknown_core = sorted(set(core_by_tag).difference(seen))
        if unknown_core:
            raise ValueError(f"Core records missing from official index: {unknown_core}")
        return cls(dataset, records)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load INCAR knowledge file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"INCAR knowledge file must contain an object: {path}")
    return value


def _normalize_tag(value: object) -> str:
    tag = str(value or "").strip().upper()
    if not tag:
        raise ValueError("INCAR knowledge record has an empty tag.")
    return tag


def _required_string(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing INCAR knowledge field: {key}")
    return value.strip()


def _optional_string(mapping: dict, key: str) -> str:
    value = mapping.get(key, "")
    return value.strip() if isinstance(value, str) else ""


def _required_positive_int(mapping: dict, key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid INCAR knowledge field: {key}")
    return value


def _load_value_meanings(mapping: dict) -> tuple[IncarValueMeaning, ...]:
    raw_items = mapping.get("value_meanings", [])
    if not isinstance(raw_items, list):
        raise ValueError("Invalid INCAR knowledge field: value_meanings")
    meanings: list[IncarValueMeaning] = []
    seen: set[tuple[str, object]] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Invalid INCAR value meaning entry.")
        value = _required_string(raw, "value")
        key = _normalize_incar_value(value)
        if key in seen:
            raise ValueError(f"Duplicate INCAR value meaning: {value}")
        seen.add(key)
        meanings.append(
            IncarValueMeaning(
                value=value,
                summary_zh=_required_string(raw, "summary_zh"),
                details_zh=_optional_string(raw, "details_zh"),
            )
        )
    return tuple(meanings)

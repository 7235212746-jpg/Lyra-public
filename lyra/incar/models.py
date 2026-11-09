from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class IncarAssignment:
    tag: str
    value: str
    line_start: int
    line_end: int
    ordinal: int
    raw: str
    is_effective: bool = True


@dataclass(frozen=True)
class ParseDiagnostic:
    line_start: int
    line_end: int
    fragment: str
    message_zh: str


@dataclass(frozen=True)
class IncarDocument:
    assignments: tuple[IncarAssignment, ...]
    diagnostics: tuple[ParseDiagnostic, ...] = ()

    def values(self, tag: str) -> tuple[str, ...]:
        normalized = tag.strip().upper()
        return tuple(item.value for item in self.assignments if item.tag == normalized)

    def effective_value(self, tag: str) -> str | None:
        values = self.values(tag)
        return values[-1] if values else None

    @property
    def duplicate_tags(self) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for item in self.assignments:
            counts[item.tag] = counts.get(item.tag, 0) + 1
        return tuple(sorted(tag for tag, count in counts.items() if count > 1))

    @property
    def effective_values(self) -> dict[str, str]:
        return {item.tag: item.value for item in self.assignments}


@dataclass(frozen=True)
class DatasetMetadata:
    source_url: str
    fetched_at: str
    license_name: str
    license_url: str
    schema_version: int = 1


@dataclass(frozen=True)
class SourceReference:
    url: str
    revision_id: int
    revision_timestamp: str
    fetched_at: str
    license_name: str
    license_url: str
    translation_curated_at: str = ""
    translation_basis_fetched_at: str = ""
    translation_basis_revision_id: int | None = None
    translation_basis_revision_timestamp: str = ""


@dataclass(frozen=True)
class IncarValueMeaning:
    value: str
    summary_zh: str
    details_zh: str = ""


@dataclass(frozen=True)
class IncarKnowledgeRecord:
    tag: str
    source: SourceReference
    title_zh: str = ""
    summary_zh: str = ""
    details_zh: str = ""
    category_zh: str = ""
    keywords_zh: tuple[str, ...] = ()
    value_meanings: tuple[IncarValueMeaning, ...] = ()
    content_origin: str = "official_index_only"
    is_official_recommendation: bool = False

    @property
    def has_knowledge_body(self) -> bool:
        return bool(self.summary_zh or self.details_zh)

    def meaning_for_value(self, value: str) -> IncarValueMeaning | None:
        key = _normalize_incar_value(value)
        return next(
            (
                item
                for item in self.value_meanings
                if _normalize_incar_value(item.value) == key
            ),
            None,
        )


@dataclass(frozen=True)
class IncarExplanation:
    assignment: IncarAssignment
    record: IncarKnowledgeRecord | None

    @property
    def value_meaning(self) -> IncarValueMeaning | None:
        if self.record is None:
            return None
        return self.record.meaning_for_value(self.assignment.value)


@dataclass(frozen=True)
class IncarFinding:
    severity: str
    code: str
    title_zh: str
    message_zh: str
    tags: tuple[str, ...]
    origin: str = "lyra_auxiliary_rule"
    source_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncarAnalysis:
    document: IncarDocument
    entries: tuple[IncarExplanation, ...]
    unknown_tags: tuple[str, ...]
    findings: tuple[IncarFinding, ...]
    disclaimer_zh: str = field(
        default=(
            "中文解释和风险提示由 Lyra 根据 VASP Wiki 整理，仅用于辅助检查，"
            "不是 VASP 官方建议；最终设置请结合具体体系并核对所附官方页面。"
        )
    )


def _normalize_incar_value(value: str) -> tuple[str, object]:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return "text", text.casefold()
    if number.is_finite():
        return "number", number
    return "text", text.casefold()

from __future__ import annotations

from .knowledge import IncarKnowledgeBase
from .models import IncarAnalysis, IncarExplanation
from .parser import parse_incar
from .rules import evaluate_rules


class IncarAssistant:
    def __init__(self, knowledge: IncarKnowledgeBase) -> None:
        self.knowledge = knowledge

    def analyse(self, text: str) -> IncarAnalysis:
        document = parse_incar(text)
        entries = tuple(
            IncarExplanation(item, self.knowledge.get(item.tag))
            for item in document.assignments
        )
        unknown = tuple(
            dict.fromkeys(
                item.assignment.tag for item in entries if item.record is None
            )
        )

        def source_url(tag: str) -> str | None:
            record = self.knowledge.get(tag)
            return record.source.url if record is not None else None

        return IncarAnalysis(
            document=document,
            entries=entries,
            unknown_tags=unknown,
            findings=evaluate_rules(document, source_url),
        )

from __future__ import annotations

import re
from collections.abc import Callable

from .models import IncarDocument, IncarFinding


_TRUE_VALUES = {"T", ".T.", "TRUE", ".TRUE."}
_FALSE_VALUES = {"F", ".F.", "FALSE", ".FALSE."}


def _boolean(value: str | None) -> bool | None:
    normalized = (value or "").strip().upper()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _number(value: str | None) -> float | None:
    token = (value or "").strip().replace("D", "E").replace("d", "e")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def evaluate_rules(
    document: IncarDocument,
    source_url: Callable[[str], str | None] | None = None,
) -> tuple[IncarFinding, ...]:
    values = document.effective_values
    findings: list[IncarFinding] = []

    def add(
        severity: str,
        code: str,
        title: str,
        message: str,
        tags: tuple[str, ...],
    ) -> None:
        urls = ()
        if source_url is not None:
            urls = tuple(
                url for tag in tags if (url := source_url(tag)) is not None
            )
        findings.append(
            IncarFinding(severity, code, title, message, tags, source_urls=urls)
        )

    for tag in document.duplicate_tags:
        add(
            "warning",
            "duplicate-tag",
            "标签重复",
            f"{tag} 出现多次；Lyra 按 VASP 常见读取行为展示最后一次赋值为有效值，请确认这不是误写。",
            (tag,),
        )

    if _boolean(values.get("LNONCOLLINEAR")) is True and _number(values.get("ISPIN")) == 2:
        add(
            "warning",
            "noncollinear-ispin",
            "非共线计算中的 ISPIN 版本风险",
            "LNONCOLLINEAR = TRUE 时，ISPIN 的处理随 VASP 版本及是否同时给出 MAGMOM 而异；"
            "VASP 6.5.0 起部分组合会直接退出。请结合实际版本核对，不把此组合一概视为物理冲突。",
            ("LNONCOLLINEAR", "ISPIN", "MAGMOM"),
        )

    if _boolean(values.get("LDAU")) is True:
        missing = tuple(tag for tag in ("LDAUL", "LDAUU", "LDAUJ") if tag not in values)
        if missing:
            add(
                "warning",
                "ldau-parameters-missing",
                "DFT+U 参数不完整",
                "LDAU 已启用，但 LDAUL、LDAUU、LDAUJ 中至少一项未显式给出；请确认默认值和元素顺序符合预期。",
                ("LDAU", *missing),
            )

    nsw = _number(values.get("NSW"))
    ibrion = _number(values.get("IBRION"))
    if nsw is not None and nsw > 0 and ibrion == -1:
        add(
            "warning",
            "ionic-steps-disabled",
            "离子步不会更新结构",
            "NSW 大于 0，但 IBRION = -1 不执行离子更新；若目标是结构优化，请重新核对。",
            ("NSW", "IBRION"),
        )

    icharg = _number(values.get("ICHARG"))
    if nsw is not None and nsw > 0 and icharg is not None and int(icharg) >= 10:
        add(
            "warning",
            "fixed-charge-ionic-run",
            "固定电荷密度用于离子步",
            "ICHARG >= 10 表示在固定电荷密度上求解；与 NSW > 0 组合时，力和结构演化通常需要特别审查。",
            ("ICHARG", "NSW"),
        )

    if _number(values.get("ISPIN")) == 2 and "MAGMOM" not in values:
        add(
            "info",
            "spin-with-default-magmom",
            "未显式设置初始磁矩",
            "ISPIN = 2 但未写 MAGMOM，计算将使用默认初始磁矩；磁性体系通常应确认初态是否合适。",
            ("ISPIN", "MAGMOM"),
        )

    return tuple(findings)

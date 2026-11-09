from __future__ import annotations

import re
from dataclasses import replace

from .models import IncarAssignment, IncarDocument, ParseDiagnostic


_TAG_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*(?:/[A-Za-z][A-Za-z0-9_]*)*$"
)


def _strip_comment(text: str) -> str:
    stripped, _quote = _strip_comment_state(text, None)
    return stripped


def _strip_comment_state(text: str, quote: str | None) -> tuple[str, str | None]:
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            continue
        if quote is None and character in {"#", "!"}:
            return text[:index], quote
    return text, quote


def _split_unquoted(text: str, delimiter: str) -> list[str]:
    result: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote is not None:
            escaped = True
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            continue
        if quote is None and character == delimiter:
            result.append(text[start:index])
            start = index + 1
    result.append(text[start:])
    return result


def _continuation_body(text: str) -> tuple[str, bool]:
    trimmed = text.rstrip()
    if not trimmed:
        return "", False
    if trimmed.endswith("\\"):
        return trimmed[:-1].rstrip(), True
    return trimmed, False


def _logical_lines(text: str) -> list[tuple[str, int, int]]:
    logical: list[tuple[str, int, int]] = []
    parts: list[str] = []
    start_line = 1
    quote: str | None = None
    physical_lines = text.splitlines()
    if not physical_lines and text:
        physical_lines = [text]
    for line_number, physical in enumerate(physical_lines, start=1):
        uncommented, quote = _strip_comment_state(physical, quote)
        body, continues = _continuation_body(uncommented)
        if not parts:
            start_line = line_number
        if body.strip():
            parts.append(body.strip())
        if not continues and quote is None:
            logical.append((" ".join(parts), start_line, line_number))
            parts = []
    if parts:
        logical.append((" ".join(parts), start_line, len(physical_lines)))
    return logical


def _syntax_tokens(text: str) -> list[tuple[str, str]]:
    """Split block syntax while preserving delimiters inside quoted values."""
    tokens: list[tuple[str, str]] = []
    buffer: list[str] = []
    quote: str | None = None
    escaped = False

    def flush(kind: str = "fragment") -> None:
        value = "".join(buffer).strip()
        buffer.clear()
        if value:
            tokens.append((kind, value))

    for character in text:
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == "\\" and quote is not None:
            buffer.append(character)
            escaped = True
            continue
        if character in {'"', "'"}:
            quote = None if quote == character else character if quote is None else quote
            buffer.append(character)
            continue
        if quote is None and character == ";":
            flush()
        elif quote is None and character == "{":
            flush("open")
        elif quote is None and character == "}":
            flush()
            tokens.append(("close", ""))
        else:
            buffer.append(character)
    flush()
    return tokens


def parse_incar(text: str) -> IncarDocument:
    assignments: list[IncarAssignment] = []
    diagnostics: list[ParseDiagnostic] = []
    prefixes: list[str] = []
    for logical, line_start, line_end in _logical_lines(text):
        for token_kind, raw_fragment in _syntax_tokens(logical):
            if token_kind == "open":
                component = raw_fragment.strip().upper()
                candidate = "/".join((*prefixes, component))
                if "=" in component or not _TAG_RE.fullmatch(candidate):
                    diagnostics.append(
                        ParseDiagnostic(
                            line_start,
                            line_end,
                            raw_fragment,
                            "花括号前应为有效的 INCAR 标签命名空间。",
                        )
                    )
                else:
                    prefixes.append(component)
                continue
            if token_kind == "close":
                if prefixes:
                    prefixes.pop()
                else:
                    diagnostics.append(
                        ParseDiagnostic(
                            line_start,
                            line_end,
                            "}",
                            "出现了没有对应起始花括号的结束符号。",
                        )
                    )
                continue
            fragment = raw_fragment.strip()
            if not fragment:
                continue
            if "=" not in fragment:
                diagnostics.append(
                    ParseDiagnostic(
                        line_start,
                        line_end,
                        fragment,
                        "无法识别：INCAR 标签应使用 TAG = VALUE 形式。",
                    )
                )
                continue
            raw_tag, raw_value = fragment.split("=", 1)
            local_tag = raw_tag.strip().upper()
            tag = "/".join((*prefixes, local_tag)) if prefixes else local_tag
            value = raw_value.strip()
            if not _TAG_RE.fullmatch(tag) or not value:
                diagnostics.append(
                    ParseDiagnostic(
                        line_start,
                        line_end,
                        fragment,
                        "标签名或参数值为空/格式异常。",
                    )
                )
                continue
            if value.rstrip().endswith("&"):
                diagnostics.append(
                    ParseDiagnostic(
                        line_start,
                        line_end,
                        fragment,
                        "检测到行尾 &；VASP INCAR 不把它定义为续行符，请改用反斜杠或核对原始输入。",
                    )
                )
            assignments.append(
                IncarAssignment(
                    tag=tag,
                    value=value,
                    line_start=line_start,
                    line_end=line_end,
                    ordinal=len(assignments),
                    raw=fragment,
                )
            )

    if prefixes:
        diagnostics.append(
            ParseDiagnostic(
                len(text.splitlines()) or 1,
                len(text.splitlines()) or 1,
                "/".join(prefixes),
                "花括号命名空间没有闭合。",
            )
        )

    last_occurrence = {item.tag: item.ordinal for item in assignments}
    marked = tuple(
        replace(item, is_effective=last_occurrence[item.tag] == item.ordinal)
        for item in assignments
    )
    return IncarDocument(marked, tuple(diagnostics))

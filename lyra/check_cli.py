from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import sys

from lyra.quality import assess_band, assess_dos


VASP_SENTINELS = {
    "INCAR",
    "OUTCAR",
    "OSZICAR",
    "POSCAR",
    "CONTCAR",
    "EIGENVAL",
    "DOSCAR",
    "PROCAR",
    "vasprun.xml",
    "vasprun.xml.gz",
}


def _is_vasp_directory(folder: Path) -> bool:
    try:
        names = {entry.name for entry in folder.iterdir() if entry.is_file()}
    except OSError:
        return False
    evidence_count = len(names & VASP_SENTINELS)
    return (
        "vasprun.xml" in names
        or "vasprun.xml.gz" in names
        or "OUTCAR" in names
        or evidence_count >= 2
    )


def find_vasp_directories(
    root: str | Path,
    *,
    recursive: bool,
) -> tuple[Path, ...]:
    source = Path(root).resolve()
    if not source.is_dir():
        raise OSError(f"目录不存在或不可读取：{source}")
    found = []
    if _is_vasp_directory(source):
        found.append(source)
    if recursive:
        for current, directories, _files in os.walk(source):
            directories[:] = sorted(
                name
                for name in directories
                if name not in {".lyra-cache", ".git", "__pycache__"}
            )
            folder = Path(current).resolve()
            if folder != source and _is_vasp_directory(folder):
                found.append(folder)
    return tuple(sorted(set(found), key=lambda path: str(path).casefold()))


def _capabilities(folder: Path) -> tuple[str, ...]:
    names = {entry.name for entry in folder.iterdir() if entry.is_file()}
    capabilities = []
    if {"EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR"} <= names:
        capabilities.append("Lite Band")
        if "PROCAR" in names:
            capabilities.append("Lite Projected Band")
    if {"DOSCAR", "POSCAR", "OUTCAR"} <= names:
        capabilities.append("Lite DOS/PDOS")
    if "vasprun.xml" in names or "vasprun.xml.gz" in names:
        capabilities.append("Full Mode source")
    return tuple(capabilities)


def inspect_directory(folder: Path) -> dict:
    names = {entry.name for entry in folder.iterdir() if entry.is_file()}
    workflows = []
    reports = []
    if "EIGENVAL" in names or "PROCAR" in names:
        workflows.append("band")
        reports.append(assess_band(None, folder))
    if "DOSCAR" in names:
        workflows.append("dos")
        reports.append(assess_dos(None, folder))
    if not reports:
        reports.append(assess_band(None, folder))
    unique_issues = {}
    for report in reports:
        for issue in report.issues:
            unique_issues[(issue.code, issue.evidence)] = issue
    issues = tuple(
        unique_issues[key]
        for key in sorted(
            unique_issues,
            key=lambda value: (
                {"error": 0, "warning": 1, "info": 2}.get(
                    unique_issues[value].severity,
                    3,
                ),
                value[0],
                value[1],
            ),
        )
    )
    verdict = (
        "存在阻断问题"
        if any(issue.severity == "error" for issue in issues)
        else "可分析，但需核查警告"
        if any(issue.severity == "warning" for issue in issues)
        else "未发现已知阻断问题"
    )
    return {
        "path": str(folder),
        "verdict": verdict,
        "workflows": workflows,
        "capabilities": list(_capabilities(folder)),
        "detected_files": sorted(names & VASP_SENTINELS),
        "issues": [asdict(issue) for issue in issues],
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_html(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for entry in payload["directories"]:
        issue_items = "".join(
            "<li>"
            f"<strong>{html.escape(issue['severity'].upper())}: "
            f"{html.escape(issue['summary'])}</strong><br>"
            f"证据：{html.escape(issue['evidence'])}<br>"
            f"建议：{html.escape(issue['recommendation'])}"
            "</li>"
            for issue in entry["issues"]
        ) or "<li>未发现当前规则覆盖范围内的已知问题。</li>"
        sections.append(
            f"<section><h2>{html.escape(entry['path'])}</h2>"
            f"<p>{html.escape(entry['verdict'])}</p>"
            f"<p>可用功能：{html.escape(', '.join(entry['capabilities']) or '未完整识别')}</p>"
            f"<ul>{issue_items}</ul></section>"
        )
    path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Lyra directory check</title>"
        "<style>body{font:15px/1.55 Arial,sans-serif;max-width:1100px;margin:32px auto;"
        "color:#20242c}section{border-top:1px solid #ccd0d7;padding:12px 0}li{margin:8px 0}</style>"
        "<h1>Lyra VASP 目录检查</h1>"
        + "".join(sections),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyra-check",
        description="只读检查 VASP 计算目录的文件能力与已知质量问题。",
    )
    parser.add_argument("root", help="计算目录或包含多个计算的上级目录")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归查找所有包含 VASP 文件证据的目录",
    )
    parser.add_argument("--json", dest="json_path", help="写入 JSON 报告")
    parser.add_argument("--html", dest="html_path", help="写入 HTML 报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        folders = find_vasp_directories(args.root, recursive=args.recursive)
        entries = [inspect_directory(folder) for folder in folders]
        payload = {
            "root": str(Path(args.root).resolve()),
            "directories": entries,
        }
        if args.json_path:
            _write_json(payload, Path(args.json_path))
        if args.html_path:
            _write_html(payload, Path(args.html_path))
    except (OSError, ValueError) as exc:
        print(f"lyra-check: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print("未找到包含 VASP 文件证据的目录。")
        return 0
    for entry in entries:
        issue_counts = {
            severity: sum(
                issue["severity"] == severity for issue in entry["issues"]
            )
            for severity in ("error", "warning")
        }
        print(
            f"{entry['verdict']} | {entry['path']} | "
            f"阻断 {issue_counts['error']}，警告 {issue_counts['warning']}"
        )
    return int(
        any(
            issue["severity"] == "error"
            for entry in entries
            for issue in entry["issues"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

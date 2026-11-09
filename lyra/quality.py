from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import re

import numpy as np

from lyra.adapters.common import read_poscar_structure
from lyra.models import (
    AnalysisQualityReport,
    ParsedBandData,
    ParsedDosData,
    ParsedFermiData,
    ParsedStructure,
    QualityIssue,
    QualityMetric,
)


_SETTING_PATTERN = re.compile(
    r"\b(NELM|NSW|IBRION|ISMEAR|SIGMA|EDIFF|ENCUT|KSPACING|"
    r"LORBIT|ICHARG|ISTART|ISPIN|ISYM)\s*=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)
_ELECTRONIC_STEP_PATTERN = re.compile(
    r"^\s*(?:DAV|RMM|CG|DMP):\s*(\d+)\b",
    re.IGNORECASE,
)
_FATAL_MARKERS = (
    "VERY BAD NEWS",
    "ZBRENT: fatal error",
    "internal error in subroutine",
    "ERROR: the triple product of the basis vectors",
)
_FINAL_TIMING_MARKER = "General timing and accounting informations for this job"
# Fractional coordinates are compared after minimum-image wrapping. This is
# tight enough for normal POSCAR text rounding while still detecting atom moves.
STRUCTURE_COORDINATE_ATOL = 1e-6


@dataclass(frozen=True)
class VaspRunEvidence:
    settings: dict[str, float]
    fatal_markers: tuple[str, ...]
    has_final_timing: bool
    max_electronic_iterations: int | None
    sources: tuple[str, ...]


@dataclass(frozen=True)
class StructureComparison:
    species_order_mismatch: bool
    species_order_evidence: str
    lattice_mismatch: bool
    lattice_evidence: str
    position_mismatch: bool
    position_evidence: str


def _iter_lines(path: Path):
    with path.open("r", errors="ignore") as handle:
        yield from handle


def scan_vasp_evidence(folder: str | Path) -> VaspRunEvidence:
    root = Path(folder)
    settings: dict[str, float] = {}
    fatal_markers: set[str] = set()
    has_final_timing = False
    max_electronic_iterations = None
    sources = []
    for name in ("INCAR", "OUTCAR", "OSZICAR"):
        path = root / name
        if not path.is_file():
            continue
        sources.append(name)
        for line in _iter_lines(path):
            settings_line = (
                re.split(r"[#!]", line, maxsplit=1)[0]
                if name == "INCAR"
                else line
            )
            for key, value in _SETTING_PATTERN.findall(settings_line):
                settings[key.upper()] = float(value)
            if name == "OUTCAR":
                if _FINAL_TIMING_MARKER in line:
                    has_final_timing = True
                upper_line = line.upper()
                fatal_markers.update(
                    marker
                    for marker in _FATAL_MARKERS
                    if marker.upper() in upper_line
                )
            if name == "OSZICAR":
                match = _ELECTRONIC_STEP_PATTERN.match(line)
                if match:
                    count = int(match.group(1))
                    max_electronic_iterations = max(
                        count,
                        max_electronic_iterations or 0,
                    )
    return VaspRunEvidence(
        settings=settings,
        fatal_markers=tuple(sorted(fatal_markers)),
        has_final_timing=has_final_timing,
        max_electronic_iterations=max_electronic_iterations,
        sources=tuple(sources),
    )


def _issue(
    severity: str,
    code: str,
    summary: str,
    evidence: str,
    recommendation: str,
) -> QualityIssue:
    return QualityIssue(severity, code, summary, evidence, recommendation)


def _folder_issues(
    folder: str | Path,
    workflow: str,
) -> tuple[VaspRunEvidence, list[QualityIssue]]:
    evidence = scan_vasp_evidence(folder)
    issues = []
    if evidence.fatal_markers:
        issues.append(
            _issue(
                "error",
                "vasp-fatal-marker",
                "VASP 输出包含明确的致命错误标记。",
                "；".join(evidence.fatal_markers),
                "先修复计算并重新运行，再使用该目录生成结论性图像。",
            )
        )
    if (Path(folder) / "OUTCAR").is_file() and not evidence.has_final_timing:
        issues.append(
            _issue(
                "warning",
                "missing-final-timing",
                "OUTCAR 未发现正常结束的最终计时段。",
                f"未找到“{_FINAL_TIMING_MARKER}”。",
                "确认作业是否仍在运行、异常终止或输出被截断。",
            )
        )
    nelm = evidence.settings.get("NELM")
    if (
        nelm is not None
        and evidence.max_electronic_iterations is not None
        and evidence.max_electronic_iterations >= int(nelm)
    ):
        issues.append(
            _issue(
                "warning",
                "electronic-step-limit",
                "至少一个电子自洽循环达到 NELM 上限。",
                f"检测到最大电子步 {evidence.max_electronic_iterations}，NELM={int(nelm)}。",
                "检查 OSZICAR/OUTCAR 的收敛历史，并在必要时改善初始电荷、混合参数或增加 NELM。",
            )
        )
    nsw = int(evidence.settings.get("NSW", 0))
    ibrion = int(evidence.settings.get("IBRION", -1))
    if workflow == "dos" and nsw > 0 and ibrion >= 0:
        issues.append(
            _issue(
                "warning",
                "relaxation-dos",
                "DOS 来源看起来是离子弛豫或分子动力学计算。",
                f"NSW={nsw}，IBRION={ibrion}。",
                "论文 DOS/PDOS 通常应在最终结构上另做一次静态自洽计算。",
            )
        )
    if workflow == "band" and nsw > 0 and ibrion == 0:
        issues.append(
            _issue(
                "warning",
                "md-eigenval",
                "EIGENVAL 来源看起来是分子动力学计算。",
                f"NSW={nsw}，IBRION=0。",
                "不要把 MD 的 EIGENVAL 直接当作最终静态能带；请对目标构型另做静态/能带计算。",
            )
        )
    return evidence, issues


def _verdict(issues: list[QualityIssue]) -> str:
    severities = {issue.severity for issue in issues}
    if "error" in severities:
        return "存在阻断问题"
    if "warning" in severities:
        return "可分析，但需核查警告"
    return "未发现已知阻断问题"


def _projection_metrics(data: ParsedBandData) -> list[QualityMetric]:
    values = [
        np.asarray(array, dtype=float).ravel()
        for array in data.projection_total.values()
        if np.asarray(array).size
    ]
    if not values:
        return []
    all_values = np.concatenate(values)
    usable = np.isfinite(all_values) & (all_values > 1e-8)
    finite = all_values[np.isfinite(all_values)]
    coverage = QualityMetric(
        "可归一化投影状态",
        f"{100.0 * np.mean(usable):.2f}",
        "%",
        "原始投影总量大于 1e-8 的 k点/能带状态比例；这是数据覆盖率，不是 DFT 准确率。",
    )
    if not finite.size:
        return [coverage]
    return [
        coverage,
        QualityMetric(
            "PROCAR/投影总量范围",
            f"{np.min(finite):.4g} – {np.max(finite):.4g}",
            "",
            "这是已解析投影通道的原始总量，不是误差百分比。",
        ),
        QualityMetric(
            "PROCAR/投影总量中位数",
            f"{np.median(finite):.4g}",
            "",
            "偏离 1 可能来自投影球、PAW 投影定义、数值舍入或输入设置；Lyra 不会强行截断。",
        ),
    ]


def _settings_metrics(evidence: VaspRunEvidence) -> list[QualityMetric]:
    labels = (
        ("ENCUT", "平面波截断能", "eV"),
        ("EDIFF", "电子收敛阈值 EDIFF", "eV"),
        ("NELM", "电子步上限 NELM", ""),
        ("ISMEAR", "展宽方法 ISMEAR", ""),
        ("SIGMA", "展宽宽度 SIGMA", "eV"),
        ("KSPACING", "自动 k 点间距 KSPACING", "Å⁻¹"),
        ("NSW", "离子步上限 NSW", ""),
        ("IBRION", "离子更新方法 IBRION", ""),
        ("LORBIT", "投影设置 LORBIT", ""),
        ("ICHARG", "电荷密度读取方式 ICHARG", ""),
        ("ISTART", "波函数启动方式 ISTART", ""),
        ("ISPIN", "自旋设置 ISPIN", ""),
        ("ISYM", "对称性设置 ISYM", ""),
    )
    return [
        QualityMetric(
            label,
            f"{evidence.settings[key]:.8g}",
            unit,
            "输入/输出中检测到的计算参数；是否收敛仍需独立加密或阈值测试。"
            if key in {"ENCUT", "EDIFF", "KSPACING", "SIGMA"}
            else "",
        )
        for key, label, unit in labels
        if key in evidence.settings
    ]


def assess_band(
    data: ParsedBandData | None,
    folder: str | Path | None = None,
) -> AnalysisQualityReport:
    root = Path(folder or (data.folder if data is not None else "."))
    evidence, issues = _folder_issues(root, "band")
    metrics = _settings_metrics(evidence)
    if data is not None:
        shapes = [np.asarray(values).shape for values in data.energies.values()]
        if shapes:
            metrics.extend(
                (
                    QualityMetric("k 点数", str(shapes[0][0])),
                    QualityMetric("能带数", str(shapes[0][1])),
                    QualityMetric("自旋类型", data.spin_kind.value),
                    QualityMetric("能量参考", f"{data.efermi:.6f}", "eV", "默认使用该 Band 数据源的 E-fermi。"),
                )
            )
        metrics.extend(_projection_metrics(data))
    return AnalysisQualityReport(
        verdict=_verdict(issues),
        method="按 VASP 离散 k 点绘制；投影百分比仅在已解析投影权重内归一化，不代表 DFT 系统误差。",
        metrics=tuple(metrics),
        issues=tuple(issues),
        sources=tuple(str(root / name) for name in evidence.sources),
    )


def assess_dos(
    data: ParsedDosData | None,
    folder: str | Path | None = None,
) -> AnalysisQualityReport:
    root = Path(folder or (data.folder if data is not None else "."))
    evidence, issues = _folder_issues(root, "dos")
    metrics = _settings_metrics(evidence)
    if data is not None:
        metrics.extend(
            (
                QualityMetric("能量网格点数", str(len(data.energies))),
                QualityMetric("自旋类型", data.spin_kind.value),
                QualityMetric("能量范围", f"{data.energies[0]:.4g} – {data.energies[-1]:.4g}", "eV"),
                QualityMetric("能量参考", f"{data.efermi:.6f}", "eV"),
            )
        )
    return AnalysisQualityReport(
        verdict=_verdict(issues),
        method="读取 VASP DOS 网格；Lyra 不从单次计算虚构展宽、k 点或交换关联泛函导致的总误差。",
        metrics=tuple(metrics),
        issues=tuple(issues),
        sources=tuple(str(root / name) for name in evidence.sources),
    )


def _poscar_comparison(
    left_folder: Path,
    right_folder: Path,
    left_structure: ParsedStructure | None = None,
    right_structure: ParsedStructure | None = None,
) -> StructureComparison | None:
    try:
        left = left_structure or read_poscar_structure(left_folder / "POSCAR")
        right = right_structure or read_poscar_structure(right_folder / "POSCAR")
    except (OSError, RuntimeError, ValueError):
        return None
    species_order_mismatch = left.atom_elements != right.atom_elements
    species_order_evidence = (
        f"Band atoms={left.atom_elements}; DOS atoms={right.atom_elements}."
    )
    lattice_delta = float(np.max(np.abs(left.lattice - right.lattice)))
    lattice_scale = max(
        float(np.max(np.abs(left.lattice))),
        float(np.max(np.abs(right.lattice))),
        1.0,
    )
    lattice_tolerance = STRUCTURE_COORDINATE_ATOL * lattice_scale
    lattice_mismatch = lattice_delta > lattice_tolerance
    lattice_evidence = (
        f"实空间晶格最大绝对差 {lattice_delta:.3g} Å；"
        f"容差 {lattice_tolerance:.3g} Å。"
    )
    position_mismatch = False
    position_evidence = "逐原子周期分数坐标一致。"
    if not species_order_mismatch:
        periodic_delta = left.fractional_positions - right.fractional_positions
        periodic_delta -= np.round(periodic_delta)
        atom_deltas = np.max(np.abs(periodic_delta), axis=1)
        mismatches = np.flatnonzero(atom_deltas > STRUCTURE_COORDINATE_ATOL)
        if len(mismatches):
            atom_index = int(mismatches[0])
            position_mismatch = True
            position_evidence = (
                f"第 {atom_index + 1} 个原子 "
                f"({left.atom_elements[atom_index]}) 的最小镜像分数坐标差为 "
                f"{atom_deltas[atom_index]:.3g}；容差 "
                f"{STRUCTURE_COORDINATE_ATOL:.0e}。"
            )
    return StructureComparison(
        species_order_mismatch=species_order_mismatch,
        species_order_evidence=species_order_evidence,
        lattice_mismatch=lattice_mismatch,
        lattice_evidence=lattice_evidence,
        position_mismatch=position_mismatch,
        position_evidence=position_evidence,
    )


def assess_combined(
    band: ParsedBandData,
    dos: ParsedDosData,
) -> AnalysisQualityReport:
    band_report = assess_band(band)
    dos_report = assess_dos(dos)
    issues = list(band_report.issues) + list(dos_report.issues)
    poscar = _poscar_comparison(
        Path(band.folder),
        Path(dos.folder),
        band.structure,
        dos.structure,
    )
    if (
        band.elements != dos.elements
        or (poscar is not None and poscar.species_order_mismatch)
    ):
        issues.append(
            _issue(
                "error",
                "composition-mismatch",
                "Band 与 DOS 的元素顺序/组成不一致。",
                (
                    poscar.species_order_evidence
                    if poscar is not None and poscar.species_order_mismatch
                    else f"Band={band.elements}；DOS={dos.elements}。"
                ),
                "确认两个文件夹属于同一材料和相同原子顺序。",
            )
        )
    if band.spin_kind != dos.spin_kind:
        issues.append(
            _issue(
                "warning",
                "spin-kind-mismatch",
                "Band 与 DOS 的自旋/SOC 类型不一致。",
                f"Band={band.spin_kind.value}；DOS={dos.spin_kind.value}。",
                "论文拼图应使用一致的磁性与 SOC 设置。",
            )
        )
    fermi_difference = abs(float(band.efermi - dos.efermi))
    if fermi_difference > 0.05:
        issues.append(
            _issue(
                "warning",
                "fermi-reference-difference",
                "Band 与 DOS 的 E-fermi 差异较大。",
                f"|ΔE-fermi|={fermi_difference:.6f} eV。",
                "Lyra 仍按 DOS/SCF 文件夹的 E-fermi 对齐；请确认两次计算的物理设置一致。",
            )
        )
    if poscar is not None and poscar.lattice_mismatch:
        issues.append(
            _issue(
                "error",
                "lattice-mismatch",
                "Band 与 DOS 的晶格不一致。",
                poscar.lattice_evidence,
                "选择来自同一最终结构的 Band 与 DOS/SCF 文件夹。",
            )
        )
    if poscar is not None and poscar.position_mismatch:
        issues.append(
            _issue(
                "error",
                "structure-position-mismatch",
                "Band 与 DOS 的原子内部坐标不一致。",
                poscar.position_evidence,
                "选择来自同一最终结构和相同原子顺序的 Band 与 DOS/SCF 文件夹。",
            )
        )
    metrics = tuple(
        QualityMetric(
            f"Band / {metric.name}",
            metric.value,
            metric.unit,
            metric.interpretation,
        )
        for metric in band_report.metrics
    ) + tuple(
        QualityMetric(
            f"DOS / {metric.name}",
            metric.value,
            metric.unit,
            metric.interpretation,
        )
        for metric in dos_report.metrics
    ) + (
        QualityMetric(
            "Band 与 DOS 的 E-fermi 差",
            f"{fermi_difference:.6f}",
            "eV",
            "Combined 默认采用 DOS/SCF 文件夹的 E-fermi 作为共同参考。",
        ),
    )
    return AnalysisQualityReport(
        verdict=_verdict(issues),
        method=(
            "Band 与 DOS 分别读取，统一采用 DOS/SCF 的 E-fermi；"
            "并检查原子种类/顺序、晶格和周期分数坐标的结构一致性。"
            "该检查不证明 POTCAR、交换关联泛函或其他计算方法完全一致。"
        ),
        metrics=metrics,
        issues=tuple(issues),
        sources=tuple(dict.fromkeys(band_report.sources + dos_report.sources)),
    )


def assess_fermi(data: ParsedFermiData) -> AnalysisQualityReport:
    evidence, issues = _folder_issues(data.folder, "fermi")
    dimensions = data.grid_shape
    spacing = []
    for axis, values in enumerate(data.axes):
        if len(values) < 2:
            spacing.append(float("nan"))
            continue
        fractional_step = float(np.median(np.diff(values)))
        spacing.append(
            abs(fractional_step)
            * float(np.linalg.norm(data.reciprocal_lattice[axis]))
        )
    metrics = tuple(_settings_metrics(evidence)) + (
        QualityMetric("三维 k 网格", " x ".join(str(value) for value in dimensions)),
        QualityMetric("自旋类型", data.spin_kind.value),
        QualityMetric(
            "倒易空间典型步长",
            ", ".join(f"{value:.4g}" for value in spacing),
            "Å⁻¹",
            "依次对应三个倒易基矢方向；网格收敛需通过独立加密计算验证。",
        ),
        QualityMetric("等能面参考", f"{data.efermi:.6f}", "eV"),
    )
    return AnalysisQualityReport(
        verdict=_verdict(issues),
        method="在完整规则三维 k 网格上进行 linear interpolation，并以 marching cubes 提取等能面。",
        metrics=metrics,
        issues=tuple(issues),
        sources=tuple(str(Path(data.folder) / name) for name in evidence.sources),
    )


def report_to_dict(report: AnalysisQualityReport) -> dict:
    return asdict(report)


def write_quality_reports(
    report: AnalysisQualityReport,
    directory: str | Path,
    basename: str,
) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{basename}.lyra-report.json"
    html_path = target / f"{basename}.lyra-report.html"
    json_path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    issue_rows = "".join(
        "<tr>"
        f"<td>{html.escape(issue.severity)}</td>"
        f"<td>{html.escape(issue.summary)}</td>"
        f"<td>{html.escape(issue.evidence)}</td>"
        f"<td>{html.escape(issue.recommendation)}</td>"
        "</tr>"
        for issue in report.issues
    ) or '<tr><td colspan="4">未发现已知问题。</td></tr>'
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(metric.name)}</td>"
        f"<td>{html.escape(metric.value)} {html.escape(metric.unit)}</td>"
        f"<td>{html.escape(metric.interpretation)}</td>"
        "</tr>"
        for metric in report.metrics
    )
    html_path.write_text(
        "<!doctype html><meta charset=\"utf-8\">"
        "<title>Lyra analysis quality report</title>"
        "<style>body{font:15px/1.55 Arial,sans-serif;max-width:1100px;margin:32px auto;"
        "color:#20242c}table{border-collapse:collapse;width:100%;margin:12px 0 24px}"
        "th,td{border:1px solid #ccd0d7;padding:8px;text-align:left;vertical-align:top}"
        "th{background:#f3f5f7}</style>"
        f"<h1>{html.escape(report.verdict)}</h1>"
        f"<p>{html.escape(report.method)}</p>"
        "<h2>可测量指标</h2><table><tr><th>指标</th><th>值</th><th>解释</th></tr>"
        f"{metric_rows}</table>"
        "<h2>问题与限制</h2><table><tr><th>级别</th><th>问题</th><th>证据</th><th>建议</th></tr>"
        f"{issue_rows}</table>",
        encoding="utf-8",
    )
    return json_path, html_path

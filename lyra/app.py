from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
from threading import Thread
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageTk

from lyra.adapters.common import read_outcar_metadata, read_poscar
from lyra.adapters.inspection import (
    inspect_band_folder,
    inspect_dos_folder,
    inspect_fermi_folder,
)
from lyra.data_service import ProjectionDataService
from lyra.band_inspector_window import BandProjectionInspector
from lyra.paired_band_window import PairedBandInspector
from lyra.i18n import LANGUAGE_CHOICES, LANGUAGE_NAMES, translate
from lyra.local_zoom_window import BandLocalZoomWindow, CombinedLocalZoomWindow
from lyra.models import (
    BandPlotSettings,
    BandStateSelection,
    CapabilityReport,
    CombinedPlotSettings,
    DosPlotSettings,
    ExportSettings,
    FermiPlotSettings,
    PlotRange,
    ProjectionChannel,
    ReadMode,
)
from lyra.preview_viewer import PreviewImageViewer
from lyra.settings_store import SettingsStore
from lyra.theme import COLORS, FONTS
from lyra.tool_capabilities import ToolCapabilities
from lyra.window_effects import apply_system_backdrop
from lyra.workers import SpawnWorker


ORBITALS = ("全部", "s", "p", "d", "f", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2")
COLOURS = (
    "#3974d8",
    "#d94841",
    "#1f9d72",
    "#e18727",
    "#8b5cf6",
    "#0891b2",
    "#db2777",
    "#7c8b2a",
)
XML_BACKGROUND_SCAN_BYTES = 8 * 1024 * 1024

PHASE_LABELS = {
    "validate": "核验数据源",
    "cache-hit": "读取文件夹缓存",
    "memory-hit": "读取内存缓存",
    "cache-miss": "缓存未命中",
    "cache-write": "写入缓存",
    "cache-write-failed": "缓存写入失败（仅本次内存可用）",
    "cache-written": "缓存已写入",
    "cache-not-used": "直接读取数据源",
    "source-parse": "解析数据源",
    "metadata": "读取 PROCAR 元数据",
    "parse": "解析 PROCAR 投影表",
    "complete": "PROCAR 解析完成",
    "parse-projections": "解析投影记录",
    "preview-render": "渲染预览图",
    "export": "保存图片",
}


def format_progress_status(event: dict) -> str:
    """Render worker evidence without inventing completion percentages."""
    phase = PHASE_LABELS.get(event.get("phase"), event.get("phase") or "任务进行中")
    current = event.get("current")
    total = event.get("total")
    parts = [phase]
    detail = event.get("detail")
    if isinstance(detail, str) and detail:
        parts.append(detail)
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        parts.append(f"{current:,} / {total:,}")
    elif isinstance(current, int) and current > 0:
        parts.append(f"{current:,}")
    elapsed = event.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed >= 0:
        parts.append(f"耗时 {elapsed:.1f} 秒")
    return " | ".join(parts)


def cache_status_text(event: dict) -> str | None:
    phase = event.get("phase")
    if phase in {"cache-hit", "memory-hit"}:
        return "缓存：已命中"
    if phase == "cache-written":
        return "缓存：已写入"
    if phase == "cache-write":
        return "缓存：写入中"
    if phase == "cache-write-failed":
        return "缓存：写入失败，仅本次内存可用"
    if phase == "validate":
        return "缓存：验证中"
    if phase == "cache-miss":
        state = event.get("cache_state")
        if state == "missing":
            return "缓存：未检测到"
        if state == "stale":
            return "缓存：已失效"
        return "缓存：未命中（待验证）"
    if phase == "cache-not-used":
        return "缓存：本次不使用"
    return None


def resolve_band_projection_mode(
    selected_style: str,
    channel_count: int,
) -> tuple[str, bool]:
    mode = {"RGB": "rgb", "Fat band": "fat"}[selected_style]
    if mode == "rgb" and channel_count > 3:
        raise ValueError(
            "RGB 混色最多 3 条投影通道。请减少启用通道，"
            "或由你手动选择 Fat band 多通道彩色投影。"
        )
    return mode, False


def quality_summary(report) -> str:
    if report is None:
        return "计算质量：尚无分析结果"
    counts = {
        severity: sum(issue.severity == severity for issue in report.issues)
        for severity in ("error", "warning")
    }
    issue_text = f"阻断 {counts['error']} | 警告 {counts['warning']}"
    return f"计算质量：{report.verdict}\n{issue_text} | 可测指标 {len(report.metrics)} 项"


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / relative


def initial_window_geometry(screen_width: int, screen_height: int) -> str:
    width = min(1480, max(1080, screen_width - 100))
    height = min(820, max(640, screen_height - 120))
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    return f"{width}x{height}+{x}+{y}"


def workbench_sections() -> tuple[dict[str, object], ...]:
    return (
        {"key": "band", "label": "能带", "enabled": True},
        {"key": "dos", "label": "态密度", "enabled": True},
        {"key": "combined", "label": "能带 + 态密度", "enabled": True},
        {"key": "paired_band", "label": "能带对照", "enabled": True},
        {"key": "fermi", "label": "费米面", "enabled": True},
        {
            "key": "bonding",
            "label": "成键分析 · 规划中",
            "enabled": False,
            "tooltip": "规划中的成键分析：COHPCAR.lobster、ICOHPLIST.lobster、lobsterout、POSCAR。",
        },
        {
            "key": "valence_charge",
            "label": "价态与电荷 · 规划中",
            "enabled": False,
            "tooltip": "规划中的价态与电荷：CHARGE.lobster/GROSSPOP.lobster，或 Bader ACF.dat 加 POSCAR 和 OUTCAR/POTCAR。",
        },
        {
            "key": "remote",
            "label": "远程 Linux",
            "enabled": True,
        },
        {"key": "incar", "label": "INCAR 助手", "enabled": True},
        {"key": "mobile", "label": "手机遥控", "enabled": True},
        {"key": "settings", "label": "设置", "enabled": True},
    )


def capability_summary(
    report: CapabilityReport,
    language: str = "zh_CN",
) -> str:
    mode = "Lite Mode" if report.mode is ReadMode.LITE else "Full Mode"
    none = "None" if language == "en_US" else "无"
    detected = ", ".join(report.detected_files) or none
    used = ", ".join(report.used_files) or none
    missing = ", ".join(report.missing_files) or none
    ignored = ", ".join(report.ignored_files) or none
    available = ", ".join(key for key, value in report.features.items() if value) or none
    unavailable = "; ".join(report.disabled_reasons.values()) or none
    if language == "en_US":
        return (
            f"{mode} | Detected: {detected}\n"
            f"Read: {used} | Missing: {missing} | Ignored: {ignored}\n"
            f"Available: {available} | Limitations: {unavailable}"
        )
    return (
        f"{mode} | 已检测：{detected}\n"
        f"读取：{used} | 缺少：{missing} | 忽略：{ignored}\n"
        f"可用功能：{available} | 限制：{unavailable}"
    )


def required_files_text(
    workflow: str,
    mode: ReadMode,
    language: str = "zh_CN",
) -> str:
    if language == "en_US":
        if mode is ReadMode.LITE:
            band = (
                "Complete Band: EIGENVAL + KPOINTS + POSCAR + OUTCAR\n"
                "Element/orbital projection: PROCAR is additionally required"
            )
            dos = "DOS/PDOS: DOSCAR + POSCAR + OUTCAR"
            fermi = (
                "Fermi Surface: EIGENVAL + KPOINTS + POSCAR + OUTCAR\n"
                "Requires a full 3D uniform k mesh; a dense mesh with ISYM=-1 is recommended"
            )
        else:
            band = (
                "Complete Band: vasprun.xml(.gz) + KPOINTS\n"
                "Element/orbital projection: XML must contain projection arrays"
            )
            dos = "DOS/PDOS: vasprun.xml(.gz)"
            fermi = (
                "Fermi Surface: vasprun.xml(.gz)\n"
                "XML must contain a full 3D uniform k mesh, not a Line-mode path"
            )
        texts = {
            "band": band,
            "dos": dos,
            "combined": f"Band Folder: {band}\nDOS/SCF Folder: {dos}",
            "fermi": fermi,
        }
        try:
            return texts[workflow]
        except KeyError as exc:
            raise ValueError(
                f"Unknown workflow for required files: {workflow}"
            ) from exc
    if mode is ReadMode.LITE:
        band = (
            "完整能带：EIGENVAL + KPOINTS + POSCAR + OUTCAR\n"
            "元素/轨道投影：额外需要 PROCAR"
        )
        dos = "DOS/PDOS：DOSCAR + POSCAR + OUTCAR"
        fermi = (
            "费米面：EIGENVAL + KPOINTS + POSCAR + OUTCAR\n"
            "必须使用完整三维均匀 k 网格；建议致密网格并设置 ISYM=-1"
        )
    else:
        band = (
            "完整能带：vasprun.xml(.gz) + KPOINTS\n"
            "元素/轨道投影：XML 需包含投影数组"
        )
        dos = "DOS/PDOS：vasprun.xml(.gz)"
        fermi = (
            "费米面：vasprun.xml(.gz)\n"
            "XML 必须包含完整三维均匀 k 网格，不接受 Line-mode 路径"
        )
    texts = {
        "band": band,
        "dos": dos,
        "combined": f"Band 文件夹：{band}\nDOS/SCF 文件夹：{dos}",
        "fermi": fermi,
    }
    try:
        return texts[workflow]
    except KeyError as exc:
        raise ValueError(f"Unknown workflow for required files: {workflow}") from exc


class _Tooltip:
    """Show source requirements for disabled, planned navigation entries."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self.window is not None or not self.widget.winfo_exists():
            return
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(
            f"+{self.widget.winfo_rootx() + 12}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}"
        )
        tk.Label(
            window,
            text=self.text,
            justify="left",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            font=FONTS["small"],
        ).pack()
        self.window = window

    def _hide(self, _event=None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


class LyraApp(tk.Tk):
    def __init__(self, settings_store: SettingsStore | None = None) -> None:
        super().__init__()
        self._settings_store = settings_store
        self._plot_input_watches_ready = False
        self.title("Lyra")
        self.geometry(initial_window_geometry(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.minsize(1080, 640)
        self.configure(bg=COLORS["background"])
        try:
            self._icon_photo = tk.PhotoImage(file=resource_path("assets/lyra_icon.png"))
            self.iconphoto(True, self._icon_photo)
        except tk.TclError:
            self._icon_photo = None
        self.worker = SpawnWorker()
        self._poll_after_id = None
        self._backdrop_after_id = None
        self._preview_photo = None
        self._preview_source_image = None
        self._preview_resize_after_id = None
        self._preview_render_size = None
        self._preview_viewer = None
        self._local_zoom_window = None
        self._band_inspector = None
        self._paired_band_inspector = None
        self._band_inspector_identity = None
        self._band_preview_settings = None
        self._combined_preview_settings = None
        self._band_detail_service = ProjectionDataService()
        self._band_detail_queue = SimpleQueue()
        self._band_detail_running = False
        self._band_detail_generation = 0
        self._pending_band_state_request = None
        self._remote_window = None
        self._incar_window = None
        self._mobile_window = None
        self.tool_capabilities = ToolCapabilities.detect()
        self.last_result = None
        self._active_job: dict[str, object] | None = None
        self._current_worker_job_id = None
        self._active_preview: tuple[str, tuple] | None = None
        self._preview_sources: tuple[tuple[Path, object], ...] = ()
        self._job_controls = []
        self._saved_control_states = []
        self._current_progress_event: dict | None = None
        self._progress_received_at = 0.0
        self._inspection_queue = SimpleQueue()
        self._inspection_pending: set[tuple] = set()
        self._inspection_completed: set[tuple] = set()
        self._projection_sections = {}
        self._projection_footers = {}
        self._projection_add_buttons = {}
        self._projection_states = {}
        self._projection_preferences = {
            "band_data_type": "complete",
            "dos_projected": False,
            "combined_band_projected": False,
            "combined_dos_projected": False,
        }
        self._closing = False
        self._destroyed = False
        self._inspection_followups: dict[tuple, set[str]] = {}
        self._inspection_reports: dict[tuple, CapabilityReport] = {}
        self.read_mode_var = tk.StringVar(value=ReadMode.LITE.value)
        self.language_var = tk.StringVar(value="zh_CN")
        self.language_display_var = tk.StringVar(value="中文")
        self.task_status_var = tk.StringVar(value="就绪")
        self.quality_summary_var = tk.StringVar(value="计算质量：尚无分析结果")
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Desktop" / "Lyra Outputs"))
        self.dpi_var = tk.StringVar(value="600")
        self.width_var = tk.StringVar(value="180")
        self.format_vars = {
            suffix: tk.BooleanVar(value=True) for suffix in ("png", "pdf", "svg")
        }
        self._build_style()
        self._build_layout()
        self._restore_settings()
        self._apply_language()
        self._watch_plot_inputs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Tk delivers the initial theme event once the root is first mapped.
        self.update_idletasks()
        self._schedule_poll()
        self._schedule_backdrop()

    def _build_style(self) -> None:
        self.style = ttk.Style(self)
        if self.style.theme_use() != "clam":
            self.style.theme_use("clam")
            self.update()
        style = self.style
        style.configure("Shell.TFrame", background=COLORS["background"])
        style.configure("Header.TFrame", background=COLORS["shell"])
        style.configure("Panel.TFrame", background=COLORS["panel"], borderwidth=1, relief="solid")
        style.configure("Inline.TFrame", background=COLORS["panel"], borderwidth=0, relief="flat")
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["body"])
        style.configure("PanelMuted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=FONTS["small"])
        style.configure("Title.TLabel", background=COLORS["background"], foreground=COLORS["text"], font=FONTS["title"])
        style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["section"])
        style.configure("Status.TLabel", background=COLORS["accent_soft"], foreground=COLORS["accent"], font=FONTS["small"], padding=(8, 5))
        style.configure("Cache.TLabel", background=COLORS["panel_alt"], foreground=COLORS["muted"], font=FONTS["small"], padding=(7, 4))
        style.configure("Required.TLabel", background=COLORS["panel_alt"], foreground=COLORS["text"], font=FONTS["small"], padding=(8, 6))
        style.configure("TButton", font=FONTS["button"], padding=(10, 7), background=COLORS["panel_alt"], borderwidth=1)
        style.map("TButton", background=[("active", COLORS["accent_soft"]), ("disabled", COLORS["border_soft"])])
        style.configure("Primary.TButton", font=FONTS["button"], padding=(14, 9), background=COLORS["accent"], foreground="#ffffff", borderwidth=0)
        style.map("Primary.TButton", background=[("active", COLORS["accent_hover"]), ("disabled", COLORS["border"])])
        style.configure("Nav.TButton", font=FONTS["button"], padding=(10, 7), background=COLORS["panel"], foreground=COLORS["text"], borderwidth=0)
        style.map("Nav.TButton", background=[("active", COLORS["accent_soft"])])
        style.configure("ActiveNav.TButton", font=FONTS["button"], padding=(10, 7), background=COLORS["accent_soft"], foreground=COLORS["accent"], borderwidth=1)
        style.map("ActiveNav.TButton", background=[("active", COLORS["accent_soft"])])
        style.configure("PlannedNav.TButton", font=FONTS["button"], padding=(10, 7), background=COLORS["panel"], foreground=COLORS["muted"], borderwidth=0)
        style.map("PlannedNav.TButton", foreground=[("disabled", COLORS["muted"])], background=[("disabled", COLORS["panel"])])
        style.configure("Segmented.TRadiobutton", background=COLORS["panel_alt"], foreground=COLORS["text"], font=FONTS["button"], padding=(9, 5), borderwidth=1)
        style.map("Segmented.TRadiobutton", background=[("selected", COLORS["accent"]), ("active", COLORS["accent_soft"])], foreground=[("selected", "#ffffff"), ("disabled", COLORS["muted"])])
        style.configure("Panel.TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["body"])
        style.configure("Panel.TRadiobutton", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["body"])
        style.map("Panel.TRadiobutton", background=[("active", COLORS["panel"])])
        style.configure("TEntry", fieldbackground=COLORS["input"], bordercolor=COLORS["border"])
        style.configure("TCombobox", fieldbackground=COLORS["input"], bordercolor=COLORS["border"])
        style.configure("Preview.TLabel", background=COLORS["preview_background"], foreground=COLORS["muted"], font=FONTS["small"])
        style.configure("Workspace.TPanedwindow", background=COLORS["background"])

    def _build_layout(self) -> None:
        shell = ttk.Frame(self, style="Shell.TFrame", padding=18)
        shell.pack(fill="both", expand=True)
        self._build_header(shell)
        body = ttk.Frame(shell, style="Shell.TFrame")
        body.pack(fill="both", expand=True, pady=(12, 0))
        body.columnconfigure(0, minsize=155)
        body.columnconfigure(1, weight=1, minsize=875)
        body.rowconfigure(0, weight=1)
        self._build_sidebar(body)
        self.content_panes = ttk.Panedwindow(
            body,
            orient="horizontal",
            style="Workspace.TPanedwindow",
        )
        self.content_panes.grid(row=0, column=1, sticky="nsew")
        self._build_workspace(self.content_panes)
        self._build_preview(self.content_panes)
        self.content_panes.add(self.workspace, weight=7)
        self.content_panes.add(self.preview_panel, weight=5)
        self._workspace_min_width = 440
        self._preview_min_width = 360
        self.content_panes.bind("<Configure>", self._enforce_pane_minimums, add="+")
        self._select_page("band")

    def _build_header(self, parent) -> None:
        header = ttk.Frame(parent, style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Lyra", style="Title.TLabel").pack(side="left")
        self.header_subtitle_label = ttk.Label(header, text="电子结构绘图工作台", foreground=COLORS["muted"], background=COLORS["background"], font=FONTS["body"])
        self.header_subtitle_label.pack(side="left", padx=(12, 0), pady=(7, 0))
        mode_group = ttk.Frame(header, style="Header.TFrame")
        mode_group.pack(side="right")
        ttk.Label(mode_group, text="读取模式", foreground=COLORS["muted"], background=COLORS["background"], font=FONTS["small"]).pack(side="left", padx=(0, 6))
        self.read_mode_buttons = {}
        for text, value in (("Lite Mode", "lite"), ("Full Mode", "full")):
            button = ttk.Radiobutton(
                mode_group,
                text=text,
                value=value,
                variable=self.read_mode_var,
                command=self._mode_changed,
                style="Segmented.TRadiobutton",
            )
            button.pack(side="left", padx=2)
            self.read_mode_buttons[value] = button
            self._job_controls.append(button)

    def _build_sidebar(self, parent) -> None:
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=12, width=155)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        sidebar.grid_propagate(False)
        self.sidebar = sidebar
        self.sidebar_title_label = ttk.Label(sidebar, text="绘图模块", style="Section.TLabel")
        self.sidebar_title_label.pack(anchor="w", pady=(0, 8))
        self.nav_buttons = {}
        for section in workbench_sections():
            enabled = bool(section["enabled"])
            button = ttk.Button(
                sidebar,
                text=section["label"],
                command=(lambda key=section["key"]: self._select_page(key)) if enabled else "",
                style="Nav.TButton" if enabled else "PlannedNav.TButton",
            )
            button.pack(fill="x", pady=3)
            if not enabled:
                button.configure(state="disabled")
                _Tooltip(button, str(section["tooltip"]))
            self.nav_buttons[section["key"]] = button
        ttk.Label(sidebar, text="v0.2\nLite 优先，Full 按需", style="PanelMuted.TLabel", justify="left").pack(side="bottom", anchor="w")

    def _build_workspace(self, parent) -> None:
        self.workspace = ttk.Frame(parent, style="Panel.TFrame", padding=16, width=440)
        self.workspace.rowconfigure(0, weight=1)
        self.workspace.columnconfigure(0, weight=1)
        self.pages = {}
        self._build_band_page()
        self._build_dos_page()
        self._build_combined_page()
        self._build_paired_band_page()
        self._build_fermi_page()
        self._build_remote_page()
        self._build_incar_page()
        self._build_mobile_page()
        self._build_settings_page()

    def _new_page(self, key: str, title: str, subtitle: str):
        page = ttk.Frame(self.workspace, style="Inline.TFrame")
        page.grid(row=0, column=0, sticky="nsew")
        page.columnconfigure(0, weight=1)
        ttk.Label(page, text=title, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(page, text=subtitle, style="PanelMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 14))
        self.pages[key] = page
        return page

    def _scrollable_page_content(self, page, prefix: str):
        page.rowconfigure(2, weight=1)
        canvas = tk.Canvas(
            page,
            background=COLORS["panel"],
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=2, column=0, sticky="nsew")
        scrollbar.grid(row=2, column=1, sticky="ns")
        content = ttk.Frame(canvas, style="Inline.TFrame")
        content.columnconfigure(0, weight=1)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        setattr(self, f"{prefix}_scroll_canvas", canvas)
        return content

    def _folder_row(self, page, row, label, variable, command, mode_variable=None, mode_command=None):
        label_holder = ttk.Frame(page, style="Inline.TFrame")
        label_holder.grid(row=row, column=0, sticky="ew")
        ttk.Label(label_holder, text=label, style="Panel.TLabel").pack(side="left")
        if mode_variable is not None:
            ttk.Label(label_holder, text="读取模式", style="PanelMuted.TLabel").pack(side="right", padx=(8, 0))
            mode_box = ttk.Combobox(
                label_holder,
                textvariable=mode_variable,
                values=(ReadMode.LITE.value, ReadMode.FULL.value),
                state="readonly",
                width=6,
            )
            mode_box.pack(side="right")
            if mode_command is not None:
                mode_box.bind("<<ComboboxSelected>>", lambda _event: mode_command())
            self._job_controls.append(mode_box)
        holder = ttk.Frame(page, style="Inline.TFrame")
        holder.grid(row=row + 1, column=0, sticky="ew", pady=(4, 10))
        holder.columnconfigure(0, weight=1)
        entry = ttk.Entry(holder, textvariable=variable)
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<FocusOut>", lambda _event: command(False))
        choose_button = ttk.Button(holder, text="选择文件夹", command=lambda: command(True))
        choose_button.grid(row=0, column=1, padx=(6, 0))
        self._job_controls.extend((entry, choose_button))

    def _cache_row(self, page, row, status_var, clear_command, rebuild_command):
        holder = ttk.Frame(page, style="Inline.TFrame")
        holder.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(holder, textvariable=status_var, style="Cache.TLabel").pack(side="left")
        rebuild_button = ttk.Button(holder, text="重建缓存", command=rebuild_command)
        rebuild_button.pack(side="right")
        clear_button = ttk.Button(holder, text="清缓存", command=clear_command)
        clear_button.pack(side="right", padx=(0, 6))
        self._job_controls.extend((clear_button, rebuild_button))
        return clear_button, rebuild_button

    def _range_row(
        self,
        page,
        row,
        minimum_var,
        maximum_var,
        label="能量范围 (eV)",
    ):
        holder = ttk.Frame(page, style="Inline.TFrame")
        holder.grid(row=row, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(holder, text=label, style="Panel.TLabel").pack(side="left")
        minimum_entry = ttk.Entry(holder, textvariable=minimum_var, width=8)
        minimum_entry.pack(side="left", padx=(12, 5))
        ttk.Label(holder, text="至", style="Panel.TLabel").pack(side="left")
        maximum_entry = ttk.Entry(holder, textvariable=maximum_var, width=8)
        maximum_entry.pack(side="left", padx=5)
        self._job_controls.extend((minimum_entry, maximum_entry))

    def _projection_rows(self, page, start_row, prefix):
        rows = []
        section = ttk.Frame(page, style="Inline.TFrame")
        section.grid(row=start_row, column=0, sticky="ew")
        footer = ttk.Frame(section, style="Inline.TFrame")
        add_button = ttk.Button(
            footer,
            text="+ 添加投影通道",
            command=lambda: self._add_projection_channel(prefix),
            state="disabled",
        )
        add_button.pack(side="left", pady=(4, 0))
        setattr(self, f"{prefix}_projection_rows", rows)
        self._projection_sections[prefix] = section
        self._projection_footers[prefix] = footer
        self._projection_add_buttons[prefix] = add_button
        self._projection_states[prefix] = (False, ())
        for _index in range(3):
            self._append_projection_row(prefix)
        self._sync_projection_row_layout(prefix)
        self._job_controls.append(add_button)
        return rows

    def _append_projection_row(self, prefix):
        rows = getattr(self, f"{prefix}_projection_rows")
        index = len(rows)
        colour = COLOURS[index % len(COLOURS)]
        holder = ttk.Frame(self._projection_sections[prefix], style="Inline.TFrame")
        enabled = tk.BooleanVar(value=index == 0)
        element = tk.StringVar(value="")
        orbital = tk.StringVar(value="全部")
        colour_var = tk.StringVar(value=colour)
        check = ttk.Checkbutton(
            holder,
            text=f"通道 {index + 1}",
            variable=enabled,
            style="Panel.TCheckbutton",
        )
        check.pack(side="left")
        element_box = ttk.Combobox(
            holder,
            textvariable=element,
            width=10,
            state="disabled",
        )
        element_box.pack(side="left", padx=(8, 5))
        orbital_box = ttk.Combobox(
            holder,
            textvariable=orbital,
            values=ORBITALS,
            width=9,
            state="disabled",
        )
        orbital_box.pack(side="left", padx=5)
        swatch = tk.Button(
            holder,
            width=3,
            bg=colour,
            activebackground=colour,
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            takefocus=True,
        )
        swatch.configure(
            command=lambda variable=colour_var, button=swatch: self._choose_projection_colour(
                variable, button
            )
        )
        swatch.pack(side="left", padx=(7, 0), ipady=7)
        remove_button = ttk.Button(
            holder,
            text="−",
            width=3,
            command=lambda frame=holder: self._remove_projection_channel(
                prefix, frame
            ),
        )
        remove_button.pack(side="left", padx=(5, 0))
        row = (
            enabled,
            element,
            orbital,
            colour_var,
            element_box,
            orbital_box,
            check,
            swatch,
            holder,
            remove_button,
        )
        rows.append(row)
        self._job_controls.extend(
            (check, element_box, orbital_box, swatch, remove_button)
        )
        if self._plot_input_watches_ready:
            for variable in (enabled, element, orbital, colour_var):
                variable.trace_add("write", self._mark_preview_stale)
        return row

    def _add_projection_channel(self, prefix):
        self._append_projection_row(prefix)
        enabled, elements = self._projection_states[prefix]
        self._set_projection_rows(
            getattr(self, f"{prefix}_projection_rows"),
            enabled,
            elements,
        )

    def _remove_projection_channel(self, prefix, holder):
        rows = getattr(self, f"{prefix}_projection_rows")
        if len(rows) <= 1:
            return
        row = next((item for item in rows if item[8] is holder), None)
        if row is None:
            return
        rows.remove(row)
        removed_controls = set(row[4:8]) | {row[9]}
        self._job_controls = [
            control for control in self._job_controls
            if control not in removed_controls
        ]
        holder.destroy()
        self._sync_projection_row_layout(prefix)

    def _sync_projection_row_layout(self, prefix):
        rows = getattr(self, f"{prefix}_projection_rows")
        enabled, _elements = self._projection_states[prefix]
        for index, row in enumerate(rows):
            row[6].configure(text=f"通道 {index + 1}")
            row[8].grid(row=index, column=0, sticky="ew", pady=2)
            row[9].configure(
                state="normal" if enabled and len(rows) > 1 else "disabled"
            )
        self._projection_footers[prefix].grid(
            row=len(rows),
            column=0,
            sticky="w",
        )

    def _ensure_projection_row_count(self, prefix, count):
        rows = getattr(self, f"{prefix}_projection_rows")
        while len(rows) < max(3, count):
            self._append_projection_row(prefix)
        self._sync_projection_row_layout(prefix)

    def _resize_projection_rows(self, prefix, count):
        rows = getattr(self, f"{prefix}_projection_rows")
        target = max(1, count)
        while len(rows) < target:
            self._append_projection_row(prefix)
        while len(rows) > target:
            self._remove_projection_channel(prefix, rows[-1][8])
        self._sync_projection_row_layout(prefix)

    def _choose_projection_colour(self, variable, swatch) -> None:
        _rgb, selected = colorchooser.askcolor(
            color=variable.get(),
            parent=self,
            title="选择投影颜色",
        )
        if not selected:
            return
        variable.set(selected)
        swatch.configure(bg=selected, activebackground=selected)

    def _build_band_page(self) -> None:
        page_shell = self._new_page(
            "band",
            "能带",
            "选择一次能带计算文件夹，Lite Mode 默认不读取 vasprun.xml。",
        )
        page = self._scrollable_page_content(page_shell, "band")
        self.band_folder_var = tk.StringVar()
        self.band_status_var = tk.StringVar(value="尚未选择文件夹")
        self.band_required_var = tk.StringVar(
            value=required_files_text("band", self._mode(), self.language_var.get())
        )
        self.band_min_var = tk.StringVar(value="-5")
        self.band_max_var = tk.StringVar(value="5")
        self.band_data_type_var = tk.StringVar(value="complete")
        self.band_projection_style_var = tk.StringVar(value="RGB")
        self.band_complete_colour_var = tk.StringVar(value="#000000")
        self.band_projection_reason_var = tk.StringVar(
            value="选择文件夹后检测 PROCAR 或 XML 投影数据。"
        )
        self.band_projection_available = False
        self.band_projection_elements: tuple[str, ...] = ()
        ttk.Label(page, textvariable=self.band_required_var, style="Required.TLabel", justify="left", wraplength=470).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._folder_row(page, 3, "Band 文件夹", self.band_folder_var, self._choose_band_folder)
        ttk.Label(page, textvariable=self.band_status_var, style="PanelMuted.TLabel", justify="left", wraplength=470).grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.band_cache_status_var = tk.StringVar(value="缓存：待验证")
        self.band_clear_cache_button, self.band_rebuild_cache_button = self._cache_row(
            page,
            6,
            self.band_cache_status_var,
            lambda: self._clear_cache("band"),
            lambda: self._run_band(rebuild=True),
        )
        self._range_row(page, 7, self.band_min_var, self.band_max_var)
        mode_row = ttk.Frame(page, style="Inline.TFrame")
        mode_row.grid(row=8, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(mode_row, text="数据类型", style="Panel.TLabel").pack(side="left")
        self.band_complete_radio = ttk.Radiobutton(
            mode_row,
            text="完整能带",
            value="complete",
            variable=self.band_data_type_var,
            command=self._band_projection_changed,
            style="Panel.TRadiobutton",
        )
        self.band_complete_radio.pack(side="left", padx=(10, 4))
        self.band_projected_radio = ttk.Radiobutton(
            mode_row,
            text="元素 / 轨道投影",
            value="projected",
            variable=self.band_data_type_var,
            command=self._band_projection_changed,
            style="Panel.TRadiobutton",
            state="disabled",
        )
        self.band_projected_radio.pack(side="left", padx=4)
        ttk.Label(mode_row, text="投影样式", style="Panel.TLabel").pack(side="left", padx=(14, 0))
        self.band_projection_style = ttk.Combobox(
            mode_row,
            textvariable=self.band_projection_style_var,
            values=("RGB", "Fat band"),
            width=8,
            state="disabled",
        )
        self.band_projection_style.pack(side="left", padx=(6, 0))
        self._job_controls.extend((self.band_complete_radio, self.band_projected_radio, self.band_projection_style))
        self.band_projection_mode = self.band_projection_style
        colour_row = ttk.Frame(page, style="Inline.TFrame")
        colour_row.grid(row=9, column=0, sticky="w", pady=(0, 7))
        ttk.Label(
            colour_row,
            text="完整能带颜色",
            style="Panel.TLabel",
        ).pack(side="left")
        self.band_complete_colour_swatch = tk.Button(
            colour_row,
            width=3,
            bg=self.band_complete_colour_var.get(),
            activebackground=self.band_complete_colour_var.get(),
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            takefocus=True,
            command=lambda: self._choose_projection_colour(
                self.band_complete_colour_var,
                self.band_complete_colour_swatch,
            ),
        )
        self.band_complete_colour_swatch.pack(side="left", padx=(10, 0), ipady=7)
        self._job_controls.append(self.band_complete_colour_swatch)
        ttk.Label(
            page,
            textvariable=self.band_projection_reason_var,
            style="PanelMuted.TLabel",
            wraplength=470,
        ).grid(row=10, column=0, sticky="w", pady=(0, 5))
        self._projection_rows(page, 11, "band")
        self.band_run_button = ttk.Button(page, text="预览", style="Primary.TButton", command=self._run_band, state="disabled")
        self.band_run_button.grid(row=14, column=0, sticky="e", pady=(12, 0))
        self._job_controls.append(self.band_run_button)

    def _build_dos_page(self) -> None:
        page_shell = self._new_page(
            "dos",
            "态密度",
            "DOS/PDOS 从独立的 DOS 或 SCF 文件夹读取。",
        )
        page = self._scrollable_page_content(page_shell, "dos")
        self.dos_folder_var = tk.StringVar()
        self.dos_status_var = tk.StringVar(value="尚未选择文件夹")
        self.dos_required_var = tk.StringVar(
            value=required_files_text("dos", self._mode(), self.language_var.get())
        )
        self.dos_min_var = tk.StringVar(value="-5")
        self.dos_max_var = tk.StringVar(value="5")
        self.dos_density_min_var = tk.StringVar(value="")
        self.dos_density_max_var = tk.StringVar(value="")
        self.dos_projection_enabled = tk.BooleanVar(value=False)
        ttk.Label(page, textvariable=self.dos_required_var, style="Required.TLabel", justify="left", wraplength=470).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._folder_row(page, 3, "DOS / SCF 文件夹", self.dos_folder_var, self._choose_dos_folder)
        ttk.Label(page, textvariable=self.dos_status_var, style="PanelMuted.TLabel", justify="left", wraplength=470).grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.dos_cache_status_var = tk.StringVar(value="缓存：待验证")
        self.dos_clear_cache_button, self.dos_rebuild_cache_button = self._cache_row(
            page,
            6,
            self.dos_cache_status_var,
            lambda: self._clear_cache("dos"),
            lambda: self._run_dos(rebuild=True),
        )
        self._range_row(page, 7, self.dos_min_var, self.dos_max_var)
        self._range_row(
            page,
            8,
            self.dos_density_min_var,
            self.dos_density_max_var,
            "DOS 纵轴范围 (states/eV)",
        )
        self.dos_projection_toggle = ttk.Checkbutton(
            page,
            text="绘制元素 / 轨道投影",
            variable=self.dos_projection_enabled,
            command=self._dos_projection_changed,
            style="Panel.TCheckbutton",
            state="disabled",
        )
        self.dos_projection_toggle.grid(row=9, column=0, sticky="w", pady=(0, 6))
        self._job_controls.append(self.dos_projection_toggle)
        self._projection_rows(page, 10, "dos")
        self.dos_run_button = ttk.Button(page, text="预览", style="Primary.TButton", command=self._run_dos, state="disabled")
        self.dos_run_button.grid(row=13, column=0, sticky="e", pady=(14, 0))
        self._job_controls.append(self.dos_run_button)

    def _build_combined_page(self) -> None:
        page_shell = self._new_page("combined", "能带 + 态密度", "两侧文件夹分别选择；默认统一采用 DOS/SCF 的 E-fermi。")
        page = self._scrollable_page_content(page_shell, "combined")
        self.combined_band_folder_var = tk.StringVar()
        self.combined_dos_folder_var = tk.StringVar()
        self.combined_band_read_mode_var = tk.StringVar(value=ReadMode.LITE.value)
        self.combined_dos_read_mode_var = tk.StringVar(value=ReadMode.LITE.value)
        self.combined_band_status_var = tk.StringVar(value="尚未选择 Band 文件夹")
        self.combined_dos_status_var = tk.StringVar(value="尚未选择 DOS/SCF 文件夹")
        self.combined_required_var = tk.StringVar(value=self._combined_required_text())
        self.combined_reference_var = tk.StringVar(value="等待 DOS/SCF 文件夹")
        self.combined_min_var = tk.StringVar(value="-5")
        self.combined_max_var = tk.StringVar(value="5")
        self.combined_dos_density_min_var = tk.StringVar(value="")
        self.combined_dos_density_max_var = tk.StringVar(value="")
        self.combined_complete_colour_var = tk.StringVar(value="#000000")
        ttk.Label(page, textvariable=self.combined_required_var, style="Required.TLabel", justify="left", wraplength=470).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._folder_row(
            page, 3, "Band 文件夹", self.combined_band_folder_var,
            self._choose_combined_band, self.combined_band_read_mode_var,
            lambda: self._combined_mode_changed("band"),
        )
        ttk.Label(page, textvariable=self.combined_band_status_var, style="PanelMuted.TLabel", wraplength=470).grid(row=5, column=0, sticky="ew")
        self.combined_band_cache_status_var = tk.StringVar(value="Band 缓存：待验证")
        self.combined_band_clear_cache_button, self.combined_band_rebuild_cache_button = self._cache_row(
            page, 6, self.combined_band_cache_status_var,
            lambda: self._clear_cache("combined-band"),
            lambda: self._run_combined(rebuild=True, rebuild_data_kind="band"),
        )
        self._folder_row(
            page, 7, "DOS / SCF 文件夹", self.combined_dos_folder_var,
            self._choose_combined_dos, self.combined_dos_read_mode_var,
            lambda: self._combined_mode_changed("dos"),
        )
        ttk.Label(page, textvariable=self.combined_dos_status_var, style="PanelMuted.TLabel", wraplength=470).grid(row=9, column=0, sticky="ew")
        self.combined_dos_cache_status_var = tk.StringVar(value="DOS 缓存：待验证")
        self.combined_dos_clear_cache_button, self.combined_dos_rebuild_cache_button = self._cache_row(
            page, 10, self.combined_dos_cache_status_var,
            lambda: self._clear_cache("combined-dos"),
            lambda: self._run_combined(rebuild=True, rebuild_data_kind="dos"),
        )
        ttk.Label(page, textvariable=self.combined_reference_var, style="Status.TLabel").grid(row=11, column=0, sticky="w", pady=(10, 5))
        self._range_row(page, 12, self.combined_min_var, self.combined_max_var)
        self._range_row(
            page,
            13,
            self.combined_dos_density_min_var,
            self.combined_dos_density_max_var,
            "DOS 横轴范围 (states/eV)",
        )
        self.combined_band_projected_var = tk.BooleanVar(value=False)
        self.combined_dos_projected_var = tk.BooleanVar(value=False)
        self.combined_projection_style_var = tk.StringVar(value="RGB")
        self.combined_band_projection_available = False
        self.combined_dos_projection_available = False
        self.combined_band_available = False
        self.combined_dos_available = False
        self.combined_band_elements: tuple[str, ...] = ()
        self.combined_dos_elements: tuple[str, ...] = ()
        projection_holder = ttk.Frame(page, style="Inline.TFrame")
        projection_holder.grid(row=14, column=0, sticky="ew", pady=(0, 3))
        self.combined_band_projection_toggle = ttk.Checkbutton(
            projection_holder, text="能带投影", variable=self.combined_band_projected_var,
            command=self._combined_projection_changed, style="Panel.TCheckbutton",
            state="disabled",
        )
        self.combined_band_projection_toggle.pack(side="left")
        self.combined_dos_projection_toggle = ttk.Checkbutton(
            projection_holder, text="DOS 投影", variable=self.combined_dos_projected_var,
            command=self._combined_projection_changed, style="Panel.TCheckbutton",
            state="disabled",
        )
        self.combined_dos_projection_toggle.pack(side="left", padx=(10, 0))
        self.combined_projection_style = ttk.Combobox(
            projection_holder, textvariable=self.combined_projection_style_var,
            values=("RGB", "Fat band"), width=8, state="disabled",
        )
        self.combined_projection_style.pack(side="right")
        ttk.Label(projection_holder, text="能带样式", style="PanelMuted.TLabel").pack(side="right", padx=(0, 6))
        colour_holder = ttk.Frame(page, style="Inline.TFrame")
        colour_holder.grid(row=15, column=0, sticky="w", pady=(2, 7))
        ttk.Label(
            colour_holder,
            text="完整能带颜色",
            style="Panel.TLabel",
        ).pack(side="left")
        self.combined_complete_colour_swatch = tk.Button(
            colour_holder,
            width=3,
            bg=self.combined_complete_colour_var.get(),
            activebackground=self.combined_complete_colour_var.get(),
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            takefocus=True,
            command=lambda: self._choose_projection_colour(
                self.combined_complete_colour_var,
                self.combined_complete_colour_swatch,
            ),
        )
        self.combined_complete_colour_swatch.pack(
            side="left", padx=(10, 0), ipady=7
        )
        self._projection_rows(page, 16, "combined")
        self.combined_run_button = ttk.Button(page, text="预览", style="Primary.TButton", command=self._run_combined, state="disabled")
        self.combined_run_button.grid(row=19, column=0, sticky="e", pady=(8, 0))
        self._job_controls.extend((
            self.combined_band_projection_toggle,
            self.combined_dos_projection_toggle,
            self.combined_projection_style,
            self.combined_complete_colour_swatch,
            self.combined_run_button,
        ))

    def _build_paired_band_page(self) -> None:
        page_shell = self._new_page(
            "paired_band",
            "能带对照",
            "并排比较两个计算；点击真实 k 点后给出投影相似候选与两侧组成。",
        )
        page = self._scrollable_page_content(page_shell, "paired_band")
        self.paired_left_folder_var = tk.StringVar()
        self.paired_right_folder_var = tk.StringVar()
        self.paired_left_mode_var = tk.StringVar(value=ReadMode.LITE.value)
        self.paired_right_mode_var = tk.StringVar(value=ReadMode.LITE.value)
        self.paired_left_status_var = tk.StringVar(value="尚未选择左侧 Band 文件夹")
        self.paired_right_status_var = tk.StringVar(value="尚未选择右侧 Band 文件夹")
        self.paired_min_var = tk.StringVar(value="-5")
        self.paired_max_var = tk.StringVar(value="5")
        self.paired_left_available = False
        self.paired_right_available = False
        ttk.Label(
            page,
            text="Lite：EIGENVAL + KPOINTS + POSCAR + OUTCAR + PROCAR\nFull：KPOINTS + vasprun.xml(.gz)，且 XML 需含 projected 数据",
            style="Required.TLabel",
            justify="left",
            wraplength=470,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._folder_row(
            page, 3, "左侧 Band 文件夹", self.paired_left_folder_var,
            lambda choose=True: self._choose_paired_folder("left", choose),
            self.paired_left_mode_var,
            lambda: self._paired_mode_changed("left"),
        )
        ttk.Label(
            page, textvariable=self.paired_left_status_var,
            style="PanelMuted.TLabel", justify="left", wraplength=470,
        ).grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self._folder_row(
            page, 6, "右侧 Band 文件夹", self.paired_right_folder_var,
            lambda choose=True: self._choose_paired_folder("right", choose),
            self.paired_right_mode_var,
            lambda: self._paired_mode_changed("right"),
        )
        ttk.Label(
            page, textvariable=self.paired_right_status_var,
            style="PanelMuted.TLabel", justify="left", wraplength=470,
        ).grid(row=8, column=0, sticky="ew", pady=(0, 10))
        self._range_row(page, 9, self.paired_min_var, self.paired_max_var)
        ttk.Label(
            page,
            text="科学口径：同 k 点、能量接近和投影成分相似只构成候选证据；WAVECAR 重叠分析上线前不标记为同一物理能带。",
            style="PanelMuted.TLabel",
            justify="left",
            wraplength=470,
        ).grid(row=10, column=0, sticky="ew", pady=(4, 8))
        self.paired_run_button = ttk.Button(
            page,
            text="打开能带对照",
            style="Primary.TButton",
            command=self._run_paired_band,
            state="disabled",
        )
        self.paired_run_button.grid(row=11, column=0, sticky="e", pady=(8, 0))
        self._job_controls.append(self.paired_run_button)

    def _build_fermi_page(self) -> None:
        page_shell = self._new_page(
            "fermi",
            "费米面",
            "选择完整三维均匀 k 网格计算文件夹；高对称 Line-mode 能带不能用于费米面。",
        )
        page = self._scrollable_page_content(page_shell, "fermi")
        self.fermi_folder_var = tk.StringVar()
        self.fermi_status_var = tk.StringVar(value="尚未选择文件夹")
        self.fermi_required_var = tk.StringVar(
            value=required_files_text("fermi", self._mode(), self.language_var.get())
        )
        self.fermi_view_mode_var = tk.StringVar(value="surface")
        self.fermi_energy_offset_var = tk.StringVar(value="0.0")
        self.fermi_spin_var = tk.StringVar(value="all")
        self.fermi_slice_axis_var = tk.StringVar(value="kz")
        self.fermi_slice_position_var = tk.StringVar(value="0.0")
        self.fermi_elevation_var = tk.StringVar(value="24")
        self.fermi_azimuth_var = tk.StringVar(value="-55")
        self.fermi_colour_var = tk.StringVar(value=COLOURS[0])
        self.fermi_available = False

        ttk.Label(
            page,
            textvariable=self.fermi_required_var,
            style="Required.TLabel",
            justify="left",
            wraplength=470,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._folder_row(
            page,
            3,
            "费米面计算文件夹",
            self.fermi_folder_var,
            self._choose_fermi_folder,
        )
        ttk.Label(
            page,
            textvariable=self.fermi_status_var,
            style="PanelMuted.TLabel",
            justify="left",
            wraplength=470,
        ).grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.fermi_cache_status_var = tk.StringVar(value="缓存：待验证")
        self.fermi_clear_cache_button, self.fermi_rebuild_cache_button = self._cache_row(
            page,
            6,
            self.fermi_cache_status_var,
            lambda: self._clear_cache("fermi"),
            lambda: self._run_fermi(rebuild=True),
        )

        mode_row = ttk.Frame(page, style="Inline.TFrame")
        mode_row.grid(row=7, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(mode_row, text="查看方式", style="Panel.TLabel").pack(side="left")
        surface_radio = ttk.Radiobutton(
            mode_row,
            text="三维费米面",
            value="surface",
            variable=self.fermi_view_mode_var,
            command=self._refresh_fermi_controls,
            style="Panel.TRadiobutton",
        )
        surface_radio.pack(side="left", padx=(10, 4))
        slice_radio = ttk.Radiobutton(
            mode_row,
            text="二维切片",
            value="slice",
            variable=self.fermi_view_mode_var,
            command=self._refresh_fermi_controls,
            style="Panel.TRadiobutton",
        )
        slice_radio.pack(side="left", padx=4)

        energy_row = ttk.Frame(page, style="Inline.TFrame")
        energy_row.grid(row=8, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(energy_row, text="能量偏移 E-Ef (eV)", style="Panel.TLabel").pack(side="left")
        energy_entry = ttk.Entry(
            energy_row,
            textvariable=self.fermi_energy_offset_var,
            width=9,
        )
        energy_entry.pack(side="left", padx=(8, 18))
        ttk.Label(energy_row, text="自旋", style="Panel.TLabel").pack(side="left")
        spin_box = ttk.Combobox(
            energy_row,
            textvariable=self.fermi_spin_var,
            values=("all", "total", "up", "down"),
            width=8,
            state="readonly",
        )
        spin_box.pack(side="left", padx=(6, 0))

        surface_row = ttk.Frame(page, style="Inline.TFrame")
        surface_row.grid(row=9, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(surface_row, text="三维视角：仰角", style="Panel.TLabel").pack(side="left")
        elevation_entry = ttk.Entry(
            surface_row,
            textvariable=self.fermi_elevation_var,
            width=7,
        )
        elevation_entry.pack(side="left", padx=(6, 12))
        ttk.Label(surface_row, text="方位角", style="Panel.TLabel").pack(side="left")
        azimuth_entry = ttk.Entry(
            surface_row,
            textvariable=self.fermi_azimuth_var,
            width=7,
        )
        azimuth_entry.pack(side="left", padx=(6, 12))
        self.fermi_colour_swatch = tk.Button(
            surface_row,
            width=3,
            bg=self.fermi_colour_var.get(),
            activebackground=self.fermi_colour_var.get(),
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            command=lambda: self._choose_projection_colour(
                self.fermi_colour_var,
                self.fermi_colour_swatch,
            ),
        )
        self.fermi_colour_swatch.pack(side="left", ipady=7)

        slice_row = ttk.Frame(page, style="Inline.TFrame")
        slice_row.grid(row=10, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(slice_row, text="切片法向", style="Panel.TLabel").pack(side="left")
        self.fermi_slice_axis_box = ttk.Combobox(
            slice_row,
            textvariable=self.fermi_slice_axis_var,
            values=("kx", "ky", "kz"),
            width=7,
            state="disabled",
        )
        self.fermi_slice_axis_box.pack(side="left", padx=(8, 18))
        ttk.Label(slice_row, text="位置 (-0.5 至 0.5)", style="Panel.TLabel").pack(side="left")
        self.fermi_slice_position_entry = ttk.Entry(
            slice_row,
            textvariable=self.fermi_slice_position_var,
            width=8,
            state="disabled",
        )
        self.fermi_slice_position_entry.pack(side="left", padx=(8, 0))

        self.fermi_run_button = ttk.Button(
            page,
            text="预览",
            style="Primary.TButton",
            command=self._run_fermi,
            state="disabled",
        )
        self.fermi_run_button.grid(row=11, column=0, sticky="e", pady=(12, 0))
        self._job_controls.extend(
            (
                surface_radio,
                slice_radio,
                energy_entry,
                spin_box,
                elevation_entry,
                azimuth_entry,
                self.fermi_colour_swatch,
                self.fermi_slice_axis_box,
                self.fermi_slice_position_entry,
                self.fermi_run_button,
            )
        )

    def _build_settings_page(self) -> None:
        page = self._new_page("settings", "设置", "论文图默认输出 PNG 600 dpi，并同时保留 PDF 与 SVG 矢量文件。")
        language = ttk.Frame(page, style="Inline.TFrame")
        language.grid(row=2, column=0, sticky="w", pady=(0, 14))
        ttk.Label(language, text="语言", style="Panel.TLabel").pack(side="left")
        self.language_box = ttk.Combobox(
            language,
            textvariable=self.language_display_var,
            values=tuple(LANGUAGE_CHOICES),
            state="readonly",
            width=12,
        )
        self.language_box.pack(side="left", padx=(10, 0))
        self.language_box.bind("<<ComboboxSelected>>", self._language_changed)
        ttk.Label(page, text="输出文件夹", style="Panel.TLabel").grid(row=3, column=0, sticky="w")
        holder = ttk.Frame(page, style="Inline.TFrame")
        holder.grid(row=4, column=0, sticky="ew", pady=(5, 14))
        holder.columnconfigure(0, weight=1)
        output_entry = ttk.Entry(holder, textvariable=self.output_dir_var)
        output_entry.grid(row=0, column=0, sticky="ew")
        output_button = ttk.Button(holder, text="选择文件夹", command=self._choose_output)
        output_button.grid(row=0, column=1, padx=(6, 0))
        self._job_controls.extend((output_entry, output_button))
        export = ttk.Frame(page, style="Inline.TFrame")
        export.grid(row=5, column=0, sticky="w")
        ttk.Label(export, text="格式", style="Panel.TLabel").pack(side="left")
        for suffix, variable in self.format_vars.items():
            button = ttk.Checkbutton(export, text=suffix.upper(), variable=variable, style="Panel.TCheckbutton")
            button.pack(side="left", padx=(10, 0))
            self._job_controls.append(button)
        quality = ttk.Frame(page, style="Inline.TFrame")
        quality.grid(row=6, column=0, sticky="w", pady=(14, 0))
        ttk.Label(quality, text="PNG DPI", style="Panel.TLabel").pack(side="left")
        dpi_entry = ttk.Entry(quality, textvariable=self.dpi_var, width=8)
        dpi_entry.pack(side="left", padx=(8, 20))
        ttk.Label(quality, text="图宽 (mm)", style="Panel.TLabel").pack(side="left")
        width_entry = ttk.Entry(quality, textvariable=self.width_var, width=8)
        width_entry.pack(side="left", padx=(8, 0))
        self._job_controls.extend((dpi_entry, width_entry))

    def _build_tool_landing_page(
        self,
        key: str,
        title: str,
        subtitle: str,
        status: str,
        button_text: str,
        command,
    ) -> None:
        page = self._new_page(key, title, subtitle)
        ttk.Label(
            page,
            text=status,
            style="Required.TLabel",
            justify="left",
            wraplength=620,
        ).grid(row=2, column=0, sticky="ew")
        ttk.Button(
            page,
            text=button_text,
            style="Primary.TButton",
            command=command,
        ).grid(row=3, column=0, sticky="w", pady=(16, 0))

    def _build_remote_page(self) -> None:
        self._build_tool_landing_page(
            "remote",
            "远程 Linux",
            "浏览服务器计算目录、检查 VASP 任务，并选择性同步绘图数据和缓存。",
            self.tool_capabilities.remote_summary,
            "打开远程工作台",
            self._open_remote_workspace,
        )

    def _build_incar_page(self) -> None:
        self._build_tool_landing_page(
            "incar",
            "INCAR 助手",
            "离线查询 VASP 标签，逐条解释当前 INCAR，并显示来源和风险证据。",
            "知识库离线可用；联网更新只在你主动执行时发生。",
            "打开 INCAR 助手",
            self._open_incar_assistant,
        )

    def _build_mobile_page(self) -> None:
        self._build_tool_landing_page(
            "mobile",
            "手机遥控",
            "在手机浏览器中管理 Lyra 任务、查看进度、放大预览和下载结果。",
            self.tool_capabilities.mobile_summary,
            "打开手机控制中心",
            self._open_mobile_control,
        )

    def _build_preview(self, parent) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.configure(width=360)
        self.preview_panel = panel
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="预览与输出", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(panel, text="生成图片后在此预览", style="Preview.TLabel", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 8))
        self.preview_label.bind("<Configure>", self._on_preview_resize, add="+")
        self.output_list = tk.Listbox(panel, height=4, borderwidth=1, relief="solid", font=FONTS["small"], fg=COLORS["text"], bg=COLORS["panel_alt"], selectbackground=COLORS["accent_soft"])
        self.output_list.grid(row=2, column=0, sticky="ew")
        self.output_list.bind("<Double-Button-1>", self._open_selected_output)
        self.output_list.grid_remove()
        actions = ttk.Frame(panel, style="Inline.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.open_plot_button = ttk.Button(actions, text="打开图片", command=self._open_plot, state="disabled")
        self.open_plot_button.pack(side="left")
        self.zoom_preview_button = ttk.Button(
            actions,
            text="局部放大图",
            command=self._open_preview_viewer,
            state="normal",
        )
        self.zoom_preview_button.pack(side="left", padx=(6, 0))
        self._job_controls.append(self.zoom_preview_button)
        self.open_folder_button = ttk.Button(actions, text="打开文件夹", command=self._open_output_folder)
        self.open_folder_button.pack(side="left", padx=(6, 0))
        self.save_image_button = ttk.Button(actions, text="保存图片", command=self._save_image, state="disabled")
        self.save_image_button.pack(side="right")
        self.cancel_job_button = ttk.Button(
            actions,
            text="取消",
            command=self._cancel_active_job,
        )
        self._job_controls.append(self.save_image_button)
        ttk.Label(panel, textvariable=self.task_status_var, style="Status.TLabel").grid(row=4, column=0, sticky="ew", pady=(10, 0))
        quality_row = ttk.Frame(panel, style="Inline.TFrame")
        quality_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        quality_row.columnconfigure(0, weight=1)
        ttk.Label(
            quality_row,
            textvariable=self.quality_summary_var,
            style="PanelMuted.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="ew")
        self.quality_report_button = ttk.Button(
            quality_row,
            text="查看质量报告",
            command=self._open_quality_report,
            state="disabled",
        )
        self.quality_report_button.grid(row=0, column=1, padx=(8, 0))

    def _select_page(self, key: str) -> None:
        if key not in self.pages:
            return
        self.pages[key].tkraise()
        self._active_page = key
        for nav_key, button in self.nav_buttons.items():
            if str(button.cget("state")) != tk.DISABLED:
                button.configure(
                    style="ActiveNav.TButton" if nav_key == key else "Nav.TButton"
                )

    def _open_managed_window(self, attribute: str, factory) -> None:
        existing = getattr(self, attribute, None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass
        try:
            window = factory()
        except Exception as exc:
            messagebox.showerror("功能启动失败", str(exc), parent=self)
            return
        setattr(self, attribute, window)

    def _open_remote_workspace(self) -> None:
        from lyra.remote_window import RemoteWorkspaceWindow

        self._open_managed_window(
            "_remote_window",
            lambda: RemoteWorkspaceWindow(
                self,
                settings_store=self._settings_store or SettingsStore.default(),
                capabilities=self.tool_capabilities,
            ),
        )

    def _open_incar_assistant(self) -> None:
        from lyra.incar_window import IncarAssistantWindow

        self._open_managed_window(
            "_incar_window",
            lambda: IncarAssistantWindow(
                self,
                settings_store=self._settings_store or SettingsStore.default(),
            ),
        )

    def _open_mobile_control(self) -> None:
        from lyra.mobile_window import MobileControlWindow

        self._open_managed_window(
            "_mobile_window",
            lambda: MobileControlWindow(
                self,
                settings_store=self._settings_store or SettingsStore.default(),
                capabilities=self.tool_capabilities,
            ),
        )

    def _enforce_pane_minimums(self, _event=None) -> None:
        if not self.content_panes.panes():
            return
        total_width = self.content_panes.winfo_width()
        minimum = self._workspace_min_width
        maximum = total_width - self._preview_min_width
        if maximum < minimum:
            return
        position = self.content_panes.sashpos(0)
        bounded = min(max(position, minimum), maximum)
        if bounded != position:
            self.content_panes.sashpos(0, bounded)

    def _mode(self) -> ReadMode:
        return ReadMode(self.read_mode_var.get())

    def _language_changed(self, _event=None) -> None:
        self.language_var.set(
            LANGUAGE_CHOICES.get(self.language_display_var.get(), "zh_CN")
        )
        self._apply_language()
        self._save_settings()

    def _apply_language(self) -> None:
        language = self.language_var.get()
        self.language_display_var.set(
            LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["zh_CN"])
        )
        pending = [self]
        while pending:
            parent = pending.pop()
            try:
                pending.extend(parent.winfo_children())
            except tk.TclError:
                continue
            try:
                current = str(parent.cget("text"))
            except (AttributeError, tk.TclError):
                continue
            source = getattr(parent, "_lyra_translation_source", None)
            if source is None:
                if translate(current, "en_US") == current:
                    continue
                source = current
                parent._lyra_translation_source = source
            parent.configure(text=translate(source, language))
        self.band_required_var.set(
            required_files_text("band", self._mode(), language)
        )
        self.dos_required_var.set(
            required_files_text("dos", self._mode(), language)
        )
        self.fermi_required_var.set(
            required_files_text("fermi", self._mode(), language)
        )
        self.combined_required_var.set(self._combined_required_text())
        for variable in (
            self.task_status_var,
            self.quality_summary_var,
            self.band_status_var,
            self.dos_status_var,
            self.fermi_status_var,
            self.band_cache_status_var,
            self.dos_cache_status_var,
            self.fermi_cache_status_var,
            self.band_projection_reason_var,
        ):
            value = variable.get()
            variable.set(translate(value, language))

    def _watch_plot_inputs(self) -> None:
        variables = [
            self.read_mode_var,
            self.band_folder_var,
            self.band_min_var,
            self.band_max_var,
            self.band_data_type_var,
            self.band_projection_style_var,
            self.band_complete_colour_var,
            self.dos_folder_var,
            self.dos_min_var,
            self.dos_max_var,
            self.dos_density_min_var,
            self.dos_density_max_var,
            self.dos_projection_enabled,
            self.combined_band_folder_var,
            self.combined_dos_folder_var,
            self.combined_band_read_mode_var,
            self.combined_dos_read_mode_var,
            self.combined_min_var,
            self.combined_max_var,
            self.combined_dos_density_min_var,
            self.combined_dos_density_max_var,
            self.combined_band_projected_var,
            self.combined_dos_projected_var,
            self.combined_projection_style_var,
            self.combined_complete_colour_var,
            self.fermi_folder_var,
            self.fermi_view_mode_var,
            self.fermi_energy_offset_var,
            self.fermi_spin_var,
            self.fermi_slice_axis_var,
            self.fermi_slice_position_var,
            self.fermi_elevation_var,
            self.fermi_azimuth_var,
            self.fermi_colour_var,
            self.width_var,
        ]
        for rows in (self.band_projection_rows, self.dos_projection_rows, self.combined_projection_rows):
            for enabled, element, orbital, colour, *_widgets in rows:
                variables.extend((enabled, element, orbital, colour))
        for variable in variables:
            variable.trace_add("write", self._mark_preview_stale)
        self._plot_input_watches_ready = True

    def _plot_signature(self, page: str) -> tuple:
        shared = (self.width_var.get(),)
        band = (
            self.band_folder_var.get(),
            self.band_min_var.get(),
            self.band_max_var.get(),
            self.band_data_type_var.get(),
            self.band_projection_style_var.get(),
            self.band_complete_colour_var.get(),
            tuple(
                (enabled.get(), element.get(), orbital.get(), colour.get())
                for enabled, element, orbital, colour, *_ in self.band_projection_rows
            ),
        )
        dos = (
            self.dos_folder_var.get(),
            self.dos_min_var.get(),
            self.dos_max_var.get(),
            self.dos_density_min_var.get(),
            self.dos_density_max_var.get(),
            self.dos_projection_enabled.get(),
            tuple(
                (enabled.get(), element.get(), orbital.get(), colour.get())
                for enabled, element, orbital, colour, *_ in self.dos_projection_rows
            ),
        )
        if page == "band":
            return (self.read_mode_var.get(),) + shared + band
        if page == "dos":
            return (self.read_mode_var.get(),) + shared + dos
        if page == "combined":
            return shared + (
                self.combined_band_read_mode_var.get(),
                self.combined_dos_read_mode_var.get(),
                self.combined_band_folder_var.get(),
                self.combined_dos_folder_var.get(),
                self.combined_min_var.get(),
                self.combined_max_var.get(),
                self.combined_dos_density_min_var.get(),
                self.combined_dos_density_max_var.get(),
                self.combined_band_projected_var.get(),
                self.combined_dos_projected_var.get(),
                self.combined_projection_style_var.get(),
                self.combined_complete_colour_var.get(),
                tuple(
                    (enabled.get(), element.get(), orbital.get(), colour.get())
                    for enabled, element, orbital, colour, *_ in self.combined_projection_rows
                ),
            )
        if page == "fermi":
            return (self.read_mode_var.get(),) + shared + (
                self.fermi_folder_var.get(),
                self.fermi_view_mode_var.get(),
                self.fermi_energy_offset_var.get(),
                self.fermi_spin_var.get(),
                self.fermi_slice_axis_var.get(),
                self.fermi_slice_position_var.get(),
                self.fermi_elevation_var.get(),
                self.fermi_azimuth_var.get(),
                self.fermi_colour_var.get(),
            )
        raise ValueError(f"Unknown preview page: {page}")

    def _mark_preview_stale(self, *_args) -> None:
        if self._active_preview is None:
            return
        page, signature = self._active_preview
        if self._plot_signature(page) != signature or not self._preview_sources_fresh():
            self._active_preview = None
            self._preview_sources = ()
            self._set_preview_clickable(False)
            self._close_band_inspector()
            self.save_image_button.configure(state="disabled")
            if not self.worker.running:
                self.task_status_var.set("预览已过期")

    def _refresh_save_image_state(self) -> None:
        self._mark_preview_stale()
        fresh = bool(
            self._active_preview
            and self._plot_signature(self._active_preview[0]) == self._active_preview[1]
            and self._preview_sources_fresh()
            and not self.worker.running
        )
        self.save_image_button.configure(state="normal" if fresh else "disabled")

    def _preview_sources_fresh(self) -> bool:
        return bool(self._preview_sources) and all(
            ProjectionDataService.identity_matches(folder, identity)
            for folder, identity in self._preview_sources
        )

    def _set_cache_status(self, data_kind: str, text: str, folder: str | None = None) -> None:
        resolved = str(Path(folder).resolve()) if folder else None
        if data_kind == "band":
            if resolved is None or resolved == str(Path(self.band_folder_var.get()).resolve()):
                self.band_cache_status_var.set(text)
            if resolved is None or resolved == str(Path(self.combined_band_folder_var.get()).resolve()):
                self.combined_band_cache_status_var.set(f"Band {text}")
        elif data_kind == "dos":
            if resolved is None or resolved == str(Path(self.dos_folder_var.get()).resolve()):
                self.dos_cache_status_var.set(text)
            if resolved is None or resolved == str(Path(self.combined_dos_folder_var.get()).resolve()):
                self.combined_dos_cache_status_var.set(f"DOS {text}")
        elif data_kind == "fermi":
            if resolved is None or resolved == str(Path(self.fermi_folder_var.get()).resolve()):
                self.fermi_cache_status_var.set(text)

    def _apply_progress(self, event: dict) -> None:
        self._current_progress_event = dict(event)
        self._progress_received_at = time.monotonic()
        self.task_status_var.set(format_progress_status(event))
        status = cache_status_text(event)
        data_kind = event.get("data_kind")
        if status and data_kind in {"band", "dos", "fermi"}:
            self._set_cache_status(data_kind, status, event.get("folder"))

    def _refresh_active_progress(self) -> None:
        if self._current_progress_event is None:
            return
        event = dict(self._current_progress_event)
        elapsed = event.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and elapsed >= 0:
            event["elapsed_seconds"] = elapsed + time.monotonic() - self._progress_received_at
        self.task_status_var.set(format_progress_status(event))

    def _mode_changed(self) -> None:
        mode = self._mode()
        language = self.language_var.get()
        self.band_required_var.set(required_files_text("band", mode, language))
        self.dos_required_var.set(required_files_text("dos", mode, language))
        self.fermi_required_var.set(required_files_text("fermi", mode, language))
        for callback, variable in (
            (self._choose_band_folder, self.band_folder_var),
            (self._choose_dos_folder, self.dos_folder_var),
            (self._choose_fermi_folder, self.fermi_folder_var),
        ):
            if variable.get():
                callback(False)

    def _combined_required_text(self) -> str:
        language = self.language_var.get()
        band = required_files_text(
            "band", ReadMode(self.combined_band_read_mode_var.get()), language
        )
        dos = required_files_text(
            "dos", ReadMode(self.combined_dos_read_mode_var.get()), language
        )
        return f"Band 文件夹：{band}\nDOS/SCF 文件夹：{dos}"

    def _combined_mode_changed(self, data_kind: str) -> None:
        self.combined_required_var.set(self._combined_required_text())
        if data_kind == "band" and self.combined_band_folder_var.get():
            self._choose_combined_band(False)
        elif data_kind == "dos" and self.combined_dos_folder_var.get():
            self._choose_combined_dos(False)

    @staticmethod
    def _xml_scan_key(folder: Path, mode: ReadMode, data_kind: str, target: str):
        if mode is not ReadMode.FULL:
            return None
        for name in ("vasprun.xml", "vasprun.xml.gz"):
            path = folder / name
            try:
                stat = path.stat()
            except OSError:
                continue
            return (
                target,
                str(folder.resolve()),
                mode.value,
                data_kind,
                name,
                stat.st_size,
                stat.st_mtime_ns,
            )
        return None

    def _inspect_for_gui(
        self,
        folder: Path,
        mode: ReadMode,
        data_kind: str,
        target: str,
    ) -> CapabilityReport:
        inspector = {
            "band": inspect_band_folder,
            "dos": inspect_dos_folder,
            "fermi": inspect_fermi_folder,
        }[data_kind]
        key = self._xml_scan_key(folder, mode, data_kind, target)
        if key is None or key[5] <= XML_BACKGROUND_SCAN_BYTES:
            return inspector(folder, mode)
        completed_match = next(
            (
                completed
                for completed in self._inspection_completed
                if completed[1:] == key[1:]
            ),
            None,
        )
        if completed_match is not None:
            self._inspection_completed.add(key)
            report = self._inspection_reports.get(key[1:])
            return report if report is not None else inspector(folder, mode)
        if key in self._inspection_completed:
            return inspector(folder, mode)
        pending_match = next(
            (
                pending
                for pending in self._inspection_pending
                if pending[1:] == key[1:]
            ),
            None,
        )
        if pending_match is not None:
            self._inspection_followups.setdefault(pending_match, set()).add(
                target
            )
            report = inspector(
                folder,
                mode,
                scan_projection_sections=False,
            )
            self._inspection_reports.setdefault(key[1:], report)
            return report
        if key not in self._inspection_pending:
            self._inspection_pending.add(key)
            self._inspection_followups[key] = {target}

            def scan() -> None:
                error = None
                report = None
                try:
                    report = inspector(folder, mode)
                except Exception as exc:
                    error = str(exc)
                self._inspection_queue.put((key, error, report))

            Thread(target=scan, name=f"lyra-{target}-inspection", daemon=True).start()
        report = inspector(folder, mode, scan_projection_sections=False)
        self._inspection_reports.setdefault(key[1:], report)
        return report

    def _poll_inspection_results(self) -> None:
        if self.worker.running:
            return
        callbacks = {
            "band": self._choose_band_folder,
            "dos": self._choose_dos_folder,
            "combined-band": self._choose_combined_band,
            "combined-dos": self._choose_combined_dos,
        }
        while True:
            try:
                result = self._inspection_queue.get_nowait()
            except Empty:
                return
            key, error = result[:2]
            report = result[2] if len(result) > 2 else None
            self._inspection_pending.discard(key)
            if report is not None:
                self._inspection_reports[key[1:]] = report
            targets = self._inspection_followups.pop(key, {key[0]})
            targets.add(key[0])
            for target in targets:
                self._inspection_completed.add((target, *key[1:]))
            if error:
                self.task_status_var.set(f"XML 投影检测失败：{error}")
            for target in targets:
                callbacks[target](False)

    def _select_folder(self, variable, choose):
        if choose:
            selected = filedialog.askdirectory(parent=self, initialdir=variable.get() or str(Path.home()))
            if not selected:
                return None
            variable.set(selected)
        value = variable.get().strip()
        return Path(value) if value else None

    def _elements_from_folder(self, folder: Path) -> tuple[str, ...]:
        try:
            _, elements, _ = read_poscar(folder / "POSCAR")
            return elements
        except Exception:
            return ()

    def _set_projection_rows(self, rows, enabled: bool, elements: tuple[str, ...]) -> None:
        prefix = next(
            (
                name
                for name in ("band", "dos", "combined")
                if getattr(self, f"{name}_projection_rows") is rows
            ),
            None,
        )
        if prefix is not None:
            self._projection_states[prefix] = (enabled, elements)
            self._projection_add_buttons[prefix].configure(
                state="normal" if enabled else "disabled"
            )
        state = "readonly" if enabled and elements else ("normal" if enabled else "disabled")
        for index, row in enumerate(rows):
            (
                _selected,
                element,
                _orbital,
                _colour,
                element_box,
                orbital_box,
                check,
                swatch,
                *_extra,
            ) = row
            choices = (("全部元素",) + elements) if elements else ()
            element_box.configure(values=choices, state=state)
            orbital_box.configure(state="readonly" if enabled else "disabled")
            check.configure(state="normal" if enabled else "disabled")
            swatch.configure(state="normal" if enabled else "disabled")
            if elements and element.get() not in choices:
                element.set(elements[index % len(elements)])
        if prefix is not None:
            self._sync_projection_row_layout(prefix)

    def _refresh_band_projection_state(self) -> None:
        projected = (
            self.band_data_type_var.get() == "projected"
            and self.band_projection_available
        )
        self.band_projection_style.configure(
            state="readonly" if projected else "disabled"
        )
        self._set_projection_rows(
            self.band_projection_rows,
            projected,
            self.band_projection_elements,
        )

    def _band_projection_changed(self) -> None:
        self._projection_preferences["band_data_type"] = (
            self.band_data_type_var.get()
        )
        self._refresh_band_projection_state()

    def _dos_projection_changed(self) -> None:
        self._projection_preferences["dos_projected"] = bool(
            self.dos_projection_enabled.get()
        )

    def _combined_projection_changed(self) -> None:
        self._projection_preferences["combined_band_projected"] = bool(
            self.combined_band_projected_var.get()
        )
        self._projection_preferences["combined_dos_projected"] = bool(
            self.combined_dos_projected_var.get()
        )
        self._refresh_combined_projection_state()

    def _band_render_mode(self) -> str:
        if self.band_data_type_var.get() == "complete":
            return "standard"
        return {"RGB": "rgb", "Fat band": "fat"}[
            self.band_projection_style_var.get()
        ]

    def _choose_band_folder(self, choose=True) -> None:
        folder = self._select_folder(self.band_folder_var, choose)
        if folder is None:
            return
        report = self._inspect_for_gui(folder, self._mode(), "band", "band")
        self.band_status_var.set(capability_summary(report, self.language_var.get()))
        self.band_cache_status_var.set("缓存：待验证")
        self.band_clear_cache_button.configure(state="normal")
        self.band_rebuild_cache_button.configure(state="normal")
        standard = report.is_available("standard_band")
        projected = report.is_available("projected_band")
        self.band_projection_available = projected
        self.band_projection_elements = self._elements_from_folder(folder)
        self._ensure_projection_row_count(
            "band",
            len(self.band_projection_elements),
        )
        self.band_run_button.configure(state="normal" if standard else "disabled")
        self.band_complete_radio.configure(state="normal" if standard else "disabled")
        self.band_projected_radio.configure(state="normal" if projected else "disabled")
        if not projected:
            self.band_data_type_var.set("complete")
        elif self._projection_preferences["band_data_type"] == "projected":
            self.band_data_type_var.set("projected")
        if projected:
            projection_reason = "可选择元素，再细分总轨道、s/p/d/f 或具体轨道。"
        elif standard and self._mode() is ReadMode.LITE:
            projection_reason = "未检测到 PROCAR，元素 / 轨道投影不可用。"
        else:
            projection_reason = report.reason("projected_band")
        self.band_projection_reason_var.set(projection_reason)
        self._refresh_band_projection_state()

    def _choose_dos_folder(self, choose=True) -> None:
        folder = self._select_folder(self.dos_folder_var, choose)
        if folder is None:
            return
        report = self._inspect_for_gui(folder, self._mode(), "dos", "dos")
        self.dos_status_var.set(capability_summary(report, self.language_var.get()))
        self.dos_cache_status_var.set("缓存：待验证")
        self.dos_clear_cache_button.configure(state="normal")
        self.dos_rebuild_cache_button.configure(state="normal")
        total = report.is_available("total_dos")
        projected = report.is_available("projected_dos")
        self.dos_run_button.configure(state="normal" if total else "disabled")
        self.dos_projection_toggle.configure(state="normal" if projected else "disabled")
        self.dos_projection_enabled.set(
            projected and self._projection_preferences["dos_projected"]
        )
        elements = self._elements_from_folder(folder)
        self._ensure_projection_row_count("dos", len(elements))
        self._set_projection_rows(self.dos_projection_rows, projected, elements)

    def _choose_paired_folder(self, side: str, choose=True) -> None:
        folder_var = (
            self.paired_left_folder_var if side == "left" else self.paired_right_folder_var
        )
        mode_var = self.paired_left_mode_var if side == "left" else self.paired_right_mode_var
        status_var = (
            self.paired_left_status_var if side == "left" else self.paired_right_status_var
        )
        folder = self._select_folder(folder_var, choose)
        if folder is None:
            return
        report = inspect_band_folder(folder, ReadMode(mode_var.get()))
        available = report.is_available("standard_band")
        status_var.set(capability_summary(report, self.language_var.get()))
        if side == "left":
            self.paired_left_available = available
        else:
            self.paired_right_available = available
        self._refresh_paired_band_state()

    def _paired_mode_changed(self, side: str) -> None:
        self._choose_paired_folder(side, choose=False)

    def _refresh_paired_band_state(self) -> None:
        enabled = self.paired_left_available and self.paired_right_available
        self.paired_run_button.configure(state="normal" if enabled else "disabled")

    def _run_paired_band(self) -> None:
        try:
            energy_range = PlotRange(
                float(self.paired_min_var.get()),
                float(self.paired_max_var.get()),
            )
            energy_range.validate()
            self._start_job(
                "prepare-paired-band",
                {
                    "left_folder": self.paired_left_folder_var.get(),
                    "left_mode": self.paired_left_mode_var.get(),
                    "right_folder": self.paired_right_folder_var.get(),
                    "right_mode": self.paired_right_mode_var.get(),
                    "energy_range": (energy_range.minimum, energy_range.maximum),
                },
                page="paired_band",
                intent="prepare-paired-band",
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)

    def _choose_fermi_folder(self, choose=True) -> None:
        folder = self._select_folder(self.fermi_folder_var, choose)
        if folder is None:
            return
        report = inspect_fermi_folder(folder, self._mode())
        self.fermi_status_var.set(capability_summary(report, self.language_var.get()))
        self.fermi_cache_status_var.set("缓存：待验证")
        self.fermi_clear_cache_button.configure(state="normal")
        self.fermi_rebuild_cache_button.configure(state="normal")
        self.fermi_available = report.is_available("fermi_surface")
        self.fermi_run_button.configure(
            state="normal" if self.fermi_available else "disabled"
        )
        self._refresh_fermi_controls()

    def _refresh_fermi_controls(self) -> None:
        sliced = self.fermi_view_mode_var.get() == "slice"
        self.fermi_slice_axis_box.configure(
            state="readonly" if sliced else "disabled"
        )
        self.fermi_slice_position_entry.configure(
            state="normal" if sliced else "disabled"
        )

    def _choose_combined_band(self, choose=True) -> None:
        folder = self._select_folder(self.combined_band_folder_var, choose)
        if folder is None:
            return
        report = self._inspect_for_gui(
            folder,
            ReadMode(self.combined_band_read_mode_var.get()),
            "band",
            "combined-band",
        )
        self.combined_band_status_var.set(
            capability_summary(report, self.language_var.get())
        )
        self.combined_band_cache_status_var.set("Band 缓存：待验证")
        self.combined_band_clear_cache_button.configure(state="normal")
        self.combined_band_rebuild_cache_button.configure(state="normal")
        self.combined_band_projection_available = report.is_available("projected_band")
        self.combined_band_available = report.is_available("standard_band")
        self.combined_band_elements = self._elements_from_folder(folder)
        self.combined_band_projected_var.set(
            self.combined_band_projection_available
            and self._projection_preferences["combined_band_projected"]
        )
        self._refresh_combined_projection_state()
        self._refresh_combined_button()

    def _choose_combined_dos(self, choose=True) -> None:
        folder = self._select_folder(self.combined_dos_folder_var, choose)
        if folder is None:
            return
        dos_mode = ReadMode(self.combined_dos_read_mode_var.get())
        report = self._inspect_for_gui(folder, dos_mode, "dos", "combined-dos")
        self.combined_dos_status_var.set(
            capability_summary(report, self.language_var.get())
        )
        self.combined_dos_cache_status_var.set("DOS 缓存：待验证")
        self.combined_dos_clear_cache_button.configure(state="normal")
        self.combined_dos_rebuild_cache_button.configure(state="normal")
        self.combined_dos_projection_available = report.is_available("projected_dos")
        self.combined_dos_available = report.is_available("total_dos")
        self.combined_dos_elements = self._elements_from_folder(folder)
        self.combined_dos_projected_var.set(
            self.combined_dos_projection_available
            and self._projection_preferences["combined_dos_projected"]
        )
        if report.is_available("total_dos"):
            if dos_mode is ReadMode.LITE:
                try:
                    efermi = read_outcar_metadata(folder / "OUTCAR").efermi
                    self.combined_reference_var.set(f"共同能量参考：DOS/SCF E-fermi = {efermi:.6f} eV")
                except Exception:
                    self.combined_reference_var.set("共同能量参考：DOS/SCF E-fermi")
            else:
                self.combined_reference_var.set("共同能量参考：DOS/SCF XML E-fermi（读取时确定）")
        else:
            self.combined_reference_var.set("DOS/SCF 数据不可用")
        self._refresh_combined_projection_state()
        self._refresh_combined_button()

    def _refresh_combined_button(self) -> None:
        enabled = self.combined_band_available and self.combined_dos_available
        self.combined_run_button.configure(state="normal" if enabled else "disabled")

    def _refresh_combined_projection_state(self) -> None:
        if not self.combined_band_projection_available:
            self.combined_band_projected_var.set(False)
        if not self.combined_dos_projection_available:
            self.combined_dos_projected_var.set(False)
        self.combined_band_projection_toggle.configure(
            state="normal" if self.combined_band_projection_available else "disabled"
        )
        self.combined_dos_projection_toggle.configure(
            state="normal" if self.combined_dos_projection_available else "disabled"
        )
        band_projected = self.combined_band_projected_var.get()
        dos_projected = self.combined_dos_projected_var.get()
        self.combined_projection_style.configure(
            state="readonly" if band_projected else "disabled"
        )
        if band_projected and dos_projected:
            elements = tuple(
                element for element in self.combined_band_elements
                if element in self.combined_dos_elements
            )
        elif band_projected:
            elements = self.combined_band_elements
        else:
            elements = self.combined_dos_elements
        self._ensure_projection_row_count("combined", len(elements))
        self._set_projection_rows(
            self.combined_projection_rows,
            band_projected or dos_projected,
            elements,
        )

    def _channels(self, rows) -> tuple[ProjectionChannel, ...]:
        channels = []
        for enabled, element, orbital, colour, element_box, *_widgets in rows:
            if enabled.get() and element.get():
                value = orbital.get()
                selected_elements = (
                    tuple(element_box.cget("values"))[1:]
                    if element.get() == "全部元素"
                    else (element.get(),)
                )
                channels.extend(
                    ProjectionChannel(
                        name,
                        None if value == "全部" else value,
                        colour.get(),
                    )
                    for name in selected_elements
                )
        return tuple(channels)

    def _export(self, basename: str) -> ExportSettings:
        formats = tuple(suffix for suffix, variable in self.format_vars.items() if variable.get())
        if not formats:
            raise ValueError("请至少选择一种输出格式。")
        dpi = int(self.dpi_var.get())
        width = float(self.width_var.get())
        if dpi <= 0:
            raise ValueError("PNG DPI 必须大于 0。")
        if width <= 0:
            raise ValueError("图宽必须大于 0。")
        return ExportSettings(
            basename=basename,
            formats=formats,
            dpi=dpi,
            width_mm=width,
        )

    def _settings_export(self, basename: str, intent: str) -> ExportSettings:
        if intent == "preview":
            width = float(self.width_var.get())
            if width <= 0:
                raise ValueError("图宽必须大于 0。")
            return ExportSettings(basename=basename, formats=("png",), dpi=120, width_mm=width)
        return self._export(basename)

    @staticmethod
    def _optional_plot_range(minimum_var, maximum_var, label: str) -> PlotRange | None:
        minimum = minimum_var.get().strip()
        maximum = maximum_var.get().strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError(f"{label}需要同时填写最小值和最大值。")
        result = PlotRange(float(minimum), float(maximum))
        result.validate()
        return result

    def _run_band(self, intent: str = "preview", rebuild: bool = False) -> None:
        try:
            mode = self._band_render_mode()
            channels = self._channels(self.band_projection_rows) if mode != "standard" else ()
            if mode != "standard" and not channels:
                raise ValueError("投影能带至少需要一个元素或轨道通道。")
            switched_to_fat = False
            if mode != "standard":
                mode, switched_to_fat = resolve_band_projection_mode(
                    self.band_projection_style_var.get(),
                    len(channels),
                )
                if switched_to_fat:
                    self.band_projection_style_var.set("Fat band")
                    self.band_projection_reason_var.set(
                        "已启用 4 条或更多投影通道：RGB 最多表达 3 条，"
                        "已自动改用 Fat band 并保留全部通道。"
                    )
            settings = BandPlotSettings(
                energy_range=PlotRange(float(self.band_min_var.get()), float(self.band_max_var.get())),
                mode=mode,
                projection_channels=channels,
                complete_band_colour=self.band_complete_colour_var.get(),
                export=self._settings_export("lyra_band", "preview" if rebuild else intent),
            )
            operation = "rebuild-preview-band" if rebuild else f"{intent}-band"
            self._start_job(
                operation,
                {"folder": self.band_folder_var.get(), "settings": settings},
                page="band",
                intent="preview" if rebuild else intent,
            )
            if switched_to_fat:
                self.task_status_var.set(
                    "RGB 最多 3 条通道，已自动改用 Fat band"
                )
        except (ValueError, OSError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)

    def _run_dos(self, intent: str = "preview", rebuild: bool = False) -> None:
        try:
            channels = self._channels(self.dos_projection_rows) if self.dos_projection_enabled.get() else ()
            settings = DosPlotSettings(
                energy_range=PlotRange(float(self.dos_min_var.get()), float(self.dos_max_var.get())),
                density_range=self._optional_plot_range(
                    self.dos_density_min_var,
                    self.dos_density_max_var,
                    "DOS 纵轴范围",
                ),
                projection_channels=channels,
                export=self._settings_export("lyra_dos", "preview" if rebuild else intent),
            )
            operation = "rebuild-preview-dos" if rebuild else f"{intent}-dos"
            self._start_job(
                operation,
                {"folder": self.dos_folder_var.get(), "settings": settings},
                page="dos",
                intent="preview" if rebuild else intent,
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)

    def _run_combined(
        self,
        intent: str = "preview",
        rebuild: bool = False,
        rebuild_data_kind: str | None = None,
    ) -> None:
        try:
            band_mode = "standard"
            switched_to_fat = False
            band_channels = ()
            dos_channels = ()
            channels = self._channels(self.combined_projection_rows)
            if self.combined_band_projected_var.get():
                band_channels = channels
                if not band_channels:
                    raise ValueError("Combined 能带投影至少需要一个元素或轨道通道。")
                band_mode, switched_to_fat = resolve_band_projection_mode(
                    self.combined_projection_style_var.get(),
                    len(band_channels),
                )
                if switched_to_fat:
                    self.combined_projection_style_var.set("Fat band")
            if self.combined_dos_projected_var.get():
                dos_channels = channels
                if not dos_channels:
                    raise ValueError("Combined DOS 投影至少需要一个元素或轨道通道。")
            settings = CombinedPlotSettings(
                energy_range=PlotRange(float(self.combined_min_var.get()), float(self.combined_max_var.get())),
                dos_density_range=self._optional_plot_range(
                    self.combined_dos_density_min_var,
                    self.combined_dos_density_max_var,
                    "DOS 横轴范围",
                ),
                band_mode=band_mode,
                band_projection_channels=band_channels,
                band_complete_colour=self.combined_complete_colour_var.get(),
                dos_projection_channels=dos_channels,
                export=self._settings_export("lyra_band_dos", "preview" if rebuild else intent),
            )
            operation = "rebuild-preview-combined" if rebuild else f"{intent}-combined"
            self._start_job(
                operation,
                {
                    "band_folder": self.combined_band_folder_var.get(),
                    "dos_folder": self.combined_dos_folder_var.get(),
                    "band_mode": self.combined_band_read_mode_var.get(),
                    "dos_mode": self.combined_dos_read_mode_var.get(),
                    "settings": settings,
                    "rebuild_data_kind": rebuild_data_kind,
                },
                page="combined",
                intent="preview" if rebuild else intent,
            )
            if switched_to_fat:
                self.task_status_var.set(
                    "RGB 最多 3 条通道，Combined 已自动改用 Fat band"
                )
        except (ValueError, OSError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)

    def _run_fermi(self, intent: str = "preview", rebuild: bool = False) -> None:
        try:
            settings = FermiPlotSettings(
                energy_offset=float(self.fermi_energy_offset_var.get()),
                view_mode=self.fermi_view_mode_var.get(),
                spin_channel=self.fermi_spin_var.get(),
                slice_axis={"kx": 0, "ky": 1, "kz": 2}[
                    self.fermi_slice_axis_var.get()
                ],
                slice_position=float(self.fermi_slice_position_var.get()),
                elevation=float(self.fermi_elevation_var.get()),
                azimuth=float(self.fermi_azimuth_var.get()),
                colour=self.fermi_colour_var.get(),
                export=self._settings_export(
                    "lyra_fermi_surface",
                    "preview" if rebuild else intent,
                ),
            )
            settings.validate()
            operation = "rebuild-preview-fermi" if rebuild else f"{intent}-fermi"
            self._start_job(
                operation,
                {"folder": self.fermi_folder_var.get(), "settings": settings},
                page="fermi",
                intent="preview" if rebuild else intent,
            )
        except (KeyError, ValueError, OSError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)

    def _start_job(self, operation: str, payload: dict, *, page: str, intent: str) -> None:
        if self.worker.running:
            messagebox.showinfo("任务进行中", "请等待当前绘图任务完成。", parent=self)
            return
        payload.update({"mode": self.read_mode_var.get(), "output": self.output_dir_var.get()})
        if intent == "export":
            payload["preview_sources"] = self._preview_sources
        self._current_worker_job_id = self.worker.start(operation, payload)
        self._active_job = {
            "page": page,
            "intent": intent,
            "settings": payload.get("settings"),
            "energy_range": payload.get("energy_range"),
        }
        self._current_progress_event = None
        if intent == "preview":
            self._active_preview = None
        self.task_status_var.set("任务已开始")
        self._set_job_controls("disabled")
        self.cancel_job_button.configure(state="normal")
        self.cancel_job_button.pack(side="right", padx=(0, 6))

    def _cancel_active_job(self) -> None:
        if self.worker.cancel():
            self.task_status_var.set("正在取消")
            self.cancel_job_button.configure(state="disabled")

    def _set_job_controls(self, state: str) -> None:
        if state == "disabled":
            self._saved_control_states = [
                (widget, str(widget.cget("state"))) for widget in self._job_controls
            ]
            for widget in self._job_controls:
                widget.configure(state="disabled")
            return
        for widget, saved_state in self._saved_control_states:
            if widget is not self.save_image_button:
                widget.configure(state=saved_state)
        self._saved_control_states = []
        self._refresh_save_image_state()

    def _clear_cache(self, target: str) -> None:
        folders = {
            "band": (self.band_folder_var.get(), "band"),
            "dos": (self.dos_folder_var.get(), "dos"),
            "combined-band": (self.combined_band_folder_var.get(), "band"),
            "combined-dos": (self.combined_dos_folder_var.get(), "dos"),
            "fermi": (self.fermi_folder_var.get(), "fermi"),
        }
        folder, data_kind = folders[target]
        if not folder:
            return
        self._active_preview = None
        self._current_worker_job_id = self.worker.start(
            "clear-cache", {"folder": folder, "data_kind": data_kind}
        )
        self._active_job = {"page": target, "intent": "clear", "data_kind": data_kind, "folder": folder}
        self.task_status_var.set("正在清缓存")
        self._set_job_controls("disabled")
        self.cancel_job_button.pack_forget()

    def _save_image(self) -> None:
        if self._active_preview is None:
            return
        page, signature = self._active_preview
        if self._plot_signature(page) != signature or not self._preview_sources_fresh():
            self._mark_preview_stale()
            return
        if page == "band":
            self._run_band(intent="export")
        elif page == "dos":
            self._run_dos(intent="export")
        elif page == "fermi":
            self._run_fermi(intent="export")
        else:
            self._run_combined(intent="export")

    def _schedule_poll(self) -> None:
        self._poll_after_id = self.after(120, self._poll_worker)

    def _schedule_backdrop(self) -> None:
        self._backdrop_after_id = self.after_idle(self._apply_system_backdrop)

    def _apply_system_backdrop(self) -> None:
        self._backdrop_after_id = None
        try:
            if not self.winfo_ismapped():
                self._backdrop_after_id = self.after(20, self._apply_system_backdrop)
                return
            apply_system_backdrop(self)
        except Exception:
            # A material backdrop is cosmetic; the solid semantic theme is the fallback.
            pass

    def _poll_worker(self) -> None:
        pending_id = self._poll_after_id
        self._poll_after_id = None
        if pending_id is not None:
            try:
                self.after_cancel(pending_id)
            except tk.TclError:
                pass
        self._poll_inspection_results()
        self._poll_band_detail_results()
        completed_intent = (
            self._active_job.get("intent") if self._active_job is not None else None
        )
        message = self.worker.poll()
        if message is not None and message.get("job_id") == self._current_worker_job_id:
            kind = message["kind"]
            payload = message["payload"]
            if kind == "progress":
                self._apply_progress(payload)
            elif kind == "success":
                self._current_progress_event = None
                self._handle_job_success(payload)
                self._active_job = None
                self._current_worker_job_id = None
                self.cancel_job_button.pack_forget()
                self._set_job_controls("normal")
            elif kind == "cache-cleared":
                self._current_progress_event = None
                self.task_status_var.set(payload["message"])
                if self._active_job is not None:
                    self._set_cache_status(self._active_job["data_kind"], "缓存：未检测到", self._active_job["folder"])
                self._discard_preview_state()
                self._active_job = None
                self._current_worker_job_id = None
                self.cancel_job_button.pack_forget()
                self._set_job_controls("normal")
            elif kind == "cancelled":
                self._current_progress_event = None
                self.task_status_var.set("任务已取消")
                self._active_job = None
                self._current_worker_job_id = None
                self.cancel_job_button.pack_forget()
                self._set_job_controls("normal")
            else:
                self._current_progress_event = None
                self.task_status_var.set("任务失败")
                messagebox.showerror("绘图失败", payload, parent=self)
                self._active_job = None
                self._current_worker_job_id = None
                self.cancel_job_button.pack_forget()
                self._set_job_controls("normal")
            if (
                kind in {"success", "cache-cleared", "cancelled", "error"}
                and completed_intent in {"prepare-band-zoom", "prepare-combined-zoom"}
            ):
                self.zoom_preview_button.configure(
                    text=translate("局部放大图", self.language_var.get())
                )
        else:
            if self.worker.running:
                self._refresh_active_progress()
            if self._active_preview is not None:
                self._refresh_save_image_state()
        self._schedule_poll()

    def _handle_job_success(self, result) -> None:
        intent = self._active_job["intent"] if self._active_job else "export"
        if intent == "prepare-band-zoom":
            if (
                self._local_zoom_window is not None
                and self._local_zoom_window.winfo_exists()
            ):
                self._local_zoom_window.close()
            self._local_zoom_window = BandLocalZoomWindow(
                self,
                result["data"],
                result["settings"],
                self.output_dir_var.get(),
            )
            self.task_status_var.set("局部放大图已就绪")
            return
        if intent == "prepare-combined-zoom":
            if (
                self._local_zoom_window is not None
                and self._local_zoom_window.winfo_exists()
            ):
                self._local_zoom_window.close()
            self._local_zoom_window = CombinedLocalZoomWindow(
                self,
                result["band_data"],
                result["dos_data"],
                result["settings"],
                self.output_dir_var.get(),
            )
            self.task_status_var.set("Band + DOS 局部放大图已就绪")
            return
        if intent == "prepare-paired-band":
            if (
                self._paired_band_inspector is not None
                and self._paired_band_inspector.winfo_exists()
            ):
                self._paired_band_inspector.close()

            def query_detail(side: str, selection: BandStateSelection):
                folder = result.left_folder if side == "left" else result.right_folder
                identity = result.left_identity if side == "left" else result.right_identity
                return self._band_detail_service.query_cached_band_state(
                    folder, identity, selection
                )

            self._paired_band_inspector = PairedBandInspector(
                self,
                result,
                query_detail=query_detail,
                energy_range=self._active_job.get("energy_range"),
            )
            self.task_status_var.set("双能带缓存已就绪，可逐点对照")
            return
        if intent == "prepare-inspector":
            self._band_inspector_identity = result["identity"]
            if (
                self._band_inspector is not None
                and self._band_inspector.winfo_exists()
            ):
                self._band_inspector.close()
            self._band_inspector = BandProjectionInspector(
                self,
                result["index"],
                self._request_band_state,
                (
                    float(self.band_min_var.get()),
                    float(self.band_max_var.get()),
                ),
                display_data=result.get("display_data"),
                plot_settings=result.get("settings"),
            )
            self.task_status_var.set("能带投影分析已就绪")
            return
        if intent == "query-band-state":
            if (
                self._band_inspector is not None
                and self._band_inspector.winfo_exists()
            ):
                self._band_inspector.set_detail(result)
            self.task_status_var.set("能带状态明细已更新")
            return
        self._show_result(result)

    def _show_result(self, result) -> None:
        self.last_result = result
        self.quality_summary_var.set(quality_summary(result.quality_report))
        self.quality_report_button.configure(
            state="normal" if result.quality_report is not None else "disabled"
        )
        intent = self._active_job["intent"] if self._active_job else "export"
        if intent == "preview":
            page = self._active_job["page"] if self._active_job else "band"
            self._active_preview = (page, self._plot_signature(page))
            if page == "band":
                self._band_preview_settings = self._active_job.get("settings")
            elif page == "combined":
                self._combined_preview_settings = self._active_job.get("settings")
            self._preview_sources = result.source_identities
            self.task_status_var.set("预览完成")
            self.output_list.delete(0, tk.END)
            self.output_list.grid_remove()
        else:
            self.task_status_var.set("图片已保存")
            self.output_list.delete(0, tk.END)
            for path in result.output_files:
                self.output_list.insert(tk.END, str(path))
            for path in result.report_files:
                self.output_list.insert(tk.END, str(path))
            if self.output_list.size():
                self.output_list.grid()
            else:
                self.output_list.grid_remove()
        self.open_plot_button.configure(state="normal")
        self.zoom_preview_button.configure(
            text=translate("局部放大图", self.language_var.get()),
            state="normal",
        )
        try:
            with Image.open(result.preview_file) as source:
                self._preview_source_image = source.copy()
            self._preview_render_size = None
            self.update_idletasks()
            self._render_preview_image()
            self._set_preview_clickable(True)
        except OSError:
            self.preview_label.configure(text=str(result.preview_file), image="")
            self._set_preview_clickable(False)
        self._refresh_save_image_state()

    def _show_external_preview(self, preview_file: str | Path, page: str) -> None:
        from lyra.models import GeneratedResult

        self._active_job = {"intent": "preview", "page": page, "settings": None}
        try:
            self._show_result(
                GeneratedResult(
                    output_files=(Path(preview_file),),
                    preview_file=Path(preview_file),
                    message="Remote preview ready.",
                )
            )
        finally:
            self._active_job = None

    def _on_preview_resize(self, event=None) -> None:
        if self._preview_source_image is None:
            return
        if self._preview_resize_after_id is not None:
            self.after_cancel(self._preview_resize_after_id)
        self._preview_resize_after_id = self.after_idle(
            self._render_preview_image
        )

    def _render_preview_image(self) -> None:
        self._preview_resize_after_id = None
        if self._preview_source_image is None:
            return
        width = max(1, self.preview_label.winfo_width() - 4)
        height = max(1, self.preview_label.winfo_height() - 4)
        render_size = (width, height)
        if render_size == self._preview_render_size:
            return
        image = self._preview_source_image.copy()
        image.thumbnail(render_size, Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(image)
        self._preview_render_size = render_size
        self.preview_label.configure(image=self._preview_photo, text="")

    def _set_preview_clickable(self, enabled: bool) -> None:
        analyzable = bool(
            enabled
            and self._active_preview is not None
            and self._active_preview[0] == "band"
            and self.band_projection_available
        )
        if analyzable:
            self.preview_label.configure(cursor="hand2")
            self.preview_label.bind(
                "<Double-Button-1>",
                self._open_band_inspector,
            )
        else:
            self.preview_label.configure(cursor="")
            self.preview_label.unbind("<Double-Button-1>")

    def _discard_preview_state(self) -> None:
        self.last_result = None
        self._active_preview = None
        self._preview_sources = ()
        self._preview_photo = None
        self._preview_source_image = None
        self._preview_render_size = None
        self._combined_preview_settings = None
        self.output_list.delete(0, tk.END)
        self.output_list.grid_remove()
        self.open_plot_button.configure(state="disabled")
        self.zoom_preview_button.configure(state="normal")
        self.quality_summary_var.set("计算质量：尚无分析结果")
        self.quality_report_button.configure(state="disabled")
        self.save_image_button.configure(state="disabled")
        self.preview_label.configure(image="", text="生成图片后在此预览")
        self._set_preview_clickable(False)
        self._close_band_inspector()
        self._close_local_zoom_window()

    def _open_selected_output(self, _event=None) -> None:
        selection = self.output_list.curselection()
        if not selection:
            return
        path = self.output_list.get(selection[0])
        if path:
            os.startfile(path)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(parent=self, initialdir=self.output_dir_var.get() or str(Path.home()))
        if selected:
            self.output_dir_var.set(selected)

    def _open_plot(self) -> None:
        if self.last_result:
            os.startfile(self.last_result.preview_file)

    def _open_preview_viewer(self, _event=None) -> None:
        if self.last_result is None:
            messagebox.showinfo(
                "尚无预览",
                "请先在能带、态密度或组合页面生成一次预览，随后即可局部放大。",
                parent=self,
            )
            return
        if (
            self._active_preview is not None
            and self._active_preview[0] == "band"
        ):
            if self._band_preview_settings is None:
                messagebox.showerror(
                    "无法局部放大",
                    "当前能带预览缺少绘图设置，请重新生成预览。",
                    parent=self,
                )
                return
            band_folder = Path(self.band_folder_var.get()).resolve()
            preview_identity = next(
                (
                    identity
                    for folder, identity in self._preview_sources
                    if Path(folder).resolve() == band_folder
                    and getattr(identity, "data_kind", None) == "band"
                ),
                None,
            )
            if preview_identity is None:
                messagebox.showerror(
                    "无法局部放大",
                    "未找到当前预览的数据身份，请重新生成预览。",
                    parent=self,
                )
                return
            self._start_job(
                "prepare-band-zoom",
                {
                    "folder": self.band_folder_var.get(),
                    "identity": preview_identity,
                    "settings": self._band_preview_settings,
                },
                page="band",
                intent="prepare-band-zoom",
            )
            if self._active_job and self._active_job.get("intent") == "prepare-band-zoom":
                self.zoom_preview_button.configure(
                    text=translate("准备局部图…", self.language_var.get())
                )
            return
        if self._active_preview is not None and self._active_preview[0] == "combined":
            if self._combined_preview_settings is None:
                messagebox.showerror(
                    "无法局部放大",
                    "当前 Band + DOS 预览缺少绘图设置，请重新生成预览。",
                    parent=self,
                )
                return
            band_folder = Path(self.combined_band_folder_var.get()).resolve()
            dos_folder = Path(self.combined_dos_folder_var.get()).resolve()
            band_identity = next(
                (
                    identity
                    for folder, identity in self._preview_sources
                    if Path(folder).resolve() == band_folder
                    and getattr(identity, "data_kind", None) == "band"
                ),
                None,
            )
            dos_identity = next(
                (
                    identity
                    for folder, identity in self._preview_sources
                    if Path(folder).resolve() == dos_folder
                    and getattr(identity, "data_kind", None) == "dos"
                ),
                None,
            )
            if band_identity is None or dos_identity is None:
                messagebox.showerror(
                    "无法局部放大",
                    "未找到当前 Band + DOS 预览的数据身份，请重新生成预览。",
                    parent=self,
                )
                return
            self._start_job(
                "prepare-combined-zoom",
                {
                    "band_folder": self.combined_band_folder_var.get(),
                    "dos_folder": self.combined_dos_folder_var.get(),
                    "band_mode": self.combined_band_read_mode_var.get(),
                    "dos_mode": self.combined_dos_read_mode_var.get(),
                    "band_identity": band_identity,
                    "dos_identity": dos_identity,
                    "settings": self._combined_preview_settings,
                },
                page="combined",
                intent="prepare-combined-zoom",
            )
            if self._active_job and self._active_job.get("intent") == "prepare-combined-zoom":
                self.zoom_preview_button.configure(
                    text=translate("准备局部图…", self.language_var.get())
                )
            return
        preview_file = Path(self.last_result.preview_file)
        if not preview_file.is_file():
            self._set_preview_clickable(False)
            messagebox.showerror(
                "无法查看图片",
                "预览图片已不存在，请重新生成预览。",
                parent=self,
            )
            return
        try:
            if (
                self._preview_viewer is not None
                and self._preview_viewer.winfo_exists()
            ):
                self._preview_viewer.close()
            self._preview_viewer = PreviewImageViewer(self, preview_file)
        except (OSError, tk.TclError) as exc:
            messagebox.showerror(
                "无法查看图片",
                f"预览图片无法打开：{exc}",
                parent=self,
            )

    def _open_band_inspector(self, _event=None) -> str:
        if (
            self._active_preview is None
            or self._active_preview[0] != "band"
            or not self.band_projection_available
        ):
            return "break"
        if (
            self._band_inspector is not None
            and self._band_inspector.winfo_exists()
        ):
            self._band_inspector.deiconify()
            self._band_inspector.lift()
            self._band_inspector.focus_force()
            return "break"
        if self.worker.running:
            if self._active_job and self._active_job.get("intent") == "prepare-inspector":
                self.task_status_var.set("正在构建完整能带分析缓存，请稍候…")
            else:
                self.task_status_var.set("当前任务完成后即可打开能带分析")
            return "break"
        band_folder = Path(self.band_folder_var.get()).resolve()
        preview_identity = next(
            (
                identity
                for folder, identity in self._preview_sources
                if Path(folder).resolve() == band_folder
                and getattr(identity, "data_kind", None) == "band"
            ),
            None,
        )
        self._start_job(
            "prepare-band-inspector",
            {
                "folder": self.band_folder_var.get(),
                "identity": preview_identity,
                "settings": self._band_preview_settings,
                "display_channels": (
                    self._band_preview_settings.projection_channels
                    if self._band_preview_settings is not None
                    else ()
                ),
            },
            page="band",
            intent="prepare-inspector",
        )
        self._set_preview_clickable(False)
        self.preview_label.configure(cursor="wait")
        return "break"

    def _close_local_zoom_window(self) -> None:
        if self._local_zoom_window is None:
            return
        try:
            if self._local_zoom_window.winfo_exists():
                self._local_zoom_window.close()
        except tk.TclError:
            pass
        self._local_zoom_window = None

    def _request_band_state(
        self,
        selection: BandStateSelection,
        whole_band: bool,
        band_indices: tuple[int, ...],
    ) -> None:
        if self._band_inspector_identity is None:
            return
        request = (selection, whole_band, band_indices)
        if self._band_detail_running:
            self._pending_band_state_request = request
            self.task_status_var.set("正在分析，完成后将直接显示最后一次选择")
            return
        self._launch_band_state_request(request)

    def _launch_band_state_request(self, request: tuple) -> None:
        if self._band_inspector_identity is None:
            return
        selection, whole_band, band_indices = request
        folder = self.band_folder_var.get()
        identity = self._band_inspector_identity
        generation = self._band_detail_generation
        self._band_detail_running = True
        self.task_status_var.set(
            f"正在分析 k 点 {selection.kpoint_index + 1} / Band {selection.band_index + 1}"
        )

        def query() -> None:
            detail = None
            error = None
            try:
                detail = self._band_detail_service.query_cached_band_state(
                    folder,
                    identity,
                    selection,
                    whole_band=whole_band,
                    band_indices=band_indices,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._band_detail_queue.put((generation, request, detail, error))

        Thread(
            target=query,
            name="lyra-band-point-analysis",
            daemon=True,
        ).start()

    def _poll_band_detail_results(self) -> None:
        while True:
            try:
                generation, _request, detail, error = (
                    self._band_detail_queue.get_nowait()
                )
            except Empty:
                return
            if generation != self._band_detail_generation:
                continue
            self._band_detail_running = False
            pending = self._pending_band_state_request
            self._pending_band_state_request = None
            if pending is not None:
                self._launch_band_state_request(pending)
                continue
            if (
                self._band_inspector is None
                or not self._band_inspector.winfo_exists()
            ):
                continue
            if error is not None:
                self._band_inspector.detail_status_var.set(
                    f"所选点分析失败：{error}"
                )
                self.task_status_var.set("能带点分析失败")
                continue
            self._band_inspector.set_detail(detail)
            self.task_status_var.set("所选能带点组成已更新")

    def _close_band_inspector(self) -> None:
        if (
            self._band_inspector is not None
            and self._band_inspector.winfo_exists()
        ):
            self._band_inspector.close()
        self._band_inspector = None
        self._band_inspector_identity = None
        self._band_detail_generation += 1
        self._band_detail_running = False
        self._pending_band_state_request = None

    def _open_output_folder(self) -> None:
        folder = Path(self.output_dir_var.get())
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _open_quality_report(self) -> None:
        if self.last_result is None or self.last_result.quality_report is None:
            return
        report = self.last_result.quality_report
        window = tk.Toplevel(self)
        window.title("Lyra - 计算质量报告")
        window.geometry("860x640")
        window.minsize(620, 420)
        text = tk.Text(
            window,
            wrap="word",
            padx=18,
            pady=16,
            font=FONTS["body"],
            bg=COLORS["panel"],
            fg=COLORS["text"],
            relief="flat",
        )
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        lines = [
            report.verdict,
            "",
            "方法与口径",
            report.method,
            "",
            "可测量指标",
        ]
        lines.extend(
            f"{metric.name}: {metric.value} {metric.unit}\n  {metric.interpretation}".rstrip()
            for metric in report.metrics
        )
        lines.extend(("", "问题与限制"))
        if report.issues:
            lines.extend(
                f"[{issue.severity.upper()}] {issue.summary}\n"
                f"证据：{issue.evidence}\n建议：{issue.recommendation}"
                for issue in report.issues
            )
        else:
            lines.append("未发现当前规则覆盖范围内的已知问题。")
        if report.sources:
            lines.extend(("", "检查来源", *report.sources))
        text.insert("1.0", "\n\n".join(lines))
        text.configure(state="disabled")

    @staticmethod
    def _projection_settings(rows) -> list[dict[str, object]]:
        return [
            {
                "enabled": bool(enabled.get()),
                "element": element.get(),
                "orbital": orbital.get(),
                "colour": colour.get(),
            }
            for enabled, element, orbital, colour, *_widgets in rows
        ]

    def _settings_snapshot(self) -> dict[str, object]:
        band_data_type = (
            self.band_data_type_var.get()
            if self.band_projection_available
            else self._projection_preferences["band_data_type"]
        )
        dos_projected = (
            bool(self.dos_projection_enabled.get())
            if str(self.dos_projection_toggle.cget("state")) != tk.DISABLED
            else self._projection_preferences["dos_projected"]
        )
        combined_band_projected = (
            bool(self.combined_band_projected_var.get())
            if self.combined_band_projection_available
            else self._projection_preferences["combined_band_projected"]
        )
        combined_dos_projected = (
            bool(self.combined_dos_projected_var.get())
            if self.combined_dos_projection_available
            else self._projection_preferences["combined_dos_projected"]
        )
        return {
            "global": {
                "read_mode": self.read_mode_var.get(),
                "language": self.language_var.get(),
            },
            "band": {
                "folder": self.band_folder_var.get(),
                "energy_min": self.band_min_var.get(),
                "energy_max": self.band_max_var.get(),
                "data_type": band_data_type,
                "projection_style": self.band_projection_style_var.get(),
                "complete_band_colour": self.band_complete_colour_var.get(),
                "channels": self._projection_settings(self.band_projection_rows),
            },
            "dos": {
                "folder": self.dos_folder_var.get(),
                "energy_min": self.dos_min_var.get(),
                "energy_max": self.dos_max_var.get(),
                "density_min": self.dos_density_min_var.get(),
                "density_max": self.dos_density_max_var.get(),
                "projected": dos_projected,
                "channels": self._projection_settings(self.dos_projection_rows),
            },
            "combined": {
                "band_folder": self.combined_band_folder_var.get(),
                "dos_folder": self.combined_dos_folder_var.get(),
                "band_read_mode": self.combined_band_read_mode_var.get(),
                "dos_read_mode": self.combined_dos_read_mode_var.get(),
                "energy_min": self.combined_min_var.get(),
                "energy_max": self.combined_max_var.get(),
                "dos_density_min": self.combined_dos_density_min_var.get(),
                "dos_density_max": self.combined_dos_density_max_var.get(),
                "band_projected": combined_band_projected,
                "dos_projected": combined_dos_projected,
                "projection_style": self.combined_projection_style_var.get(),
                "complete_band_colour": self.combined_complete_colour_var.get(),
                "channels": self._projection_settings(
                    self.combined_projection_rows
                ),
            },
            "paired_band": {
                "left_folder": self.paired_left_folder_var.get(),
                "right_folder": self.paired_right_folder_var.get(),
                "left_mode": self.paired_left_mode_var.get(),
                "right_mode": self.paired_right_mode_var.get(),
                "energy_min": self.paired_min_var.get(),
                "energy_max": self.paired_max_var.get(),
            },
            "fermi": {
                "folder": self.fermi_folder_var.get(),
                "view_mode": self.fermi_view_mode_var.get(),
                "energy_offset": self.fermi_energy_offset_var.get(),
                "spin": self.fermi_spin_var.get(),
                "slice_axis": self.fermi_slice_axis_var.get(),
                "slice_position": self.fermi_slice_position_var.get(),
                "elevation": self.fermi_elevation_var.get(),
                "azimuth": self.fermi_azimuth_var.get(),
                "colour": self.fermi_colour_var.get(),
            },
            "export": {
                "folder": self.output_dir_var.get(),
                "dpi": self.dpi_var.get(),
                "width_mm": self.width_var.get(),
                "formats": {
                    suffix: bool(variable.get())
                    for suffix, variable in self.format_vars.items()
                },
            },
        }

    def _save_settings(self) -> str | None:
        if self._settings_store is None:
            return None
        try:
            self._settings_store.save_namespace(
                "classic", self._settings_snapshot()
            )
        except (OSError, TypeError, ValueError) as exc:
            return str(exc) or exc.__class__.__name__
        return None

    @staticmethod
    def _settings_section(settings: dict, name: str) -> dict:
        section = settings.get(name)
        return section if isinstance(section, dict) else {}

    @staticmethod
    def _restore_string_values(section: dict, variables: dict[str, tk.Variable]) -> None:
        for key, variable in variables.items():
            value = section.get(key)
            if isinstance(value, str):
                variable.set(value)

    def _restore_projection_settings(self, prefix: str, saved_rows) -> None:
        if not isinstance(saved_rows, list) or not saved_rows:
            return
        self._resize_projection_rows(prefix, len(saved_rows))
        rows = getattr(self, f"{prefix}_projection_rows")
        enabled, elements = self._projection_states[prefix]
        self._set_projection_rows(rows, enabled, elements)
        for row, saved in zip(rows, saved_rows):
            if not isinstance(saved, dict):
                continue
            selected, element, orbital, colour, element_box, _orbital_box, _check, swatch, *_ = row
            if isinstance(saved.get("enabled"), bool):
                selected.set(saved["enabled"])
            saved_element = saved.get("element")
            choices = tuple(element_box.cget("values"))
            if isinstance(saved_element, str) and (
                not choices or saved_element in choices
            ):
                element.set(saved_element)
            saved_orbital = saved.get("orbital")
            if isinstance(saved_orbital, str) and saved_orbital in ORBITALS:
                orbital.set(saved_orbital)
            saved_colour = saved.get("colour")
            if isinstance(saved_colour, str):
                try:
                    self.winfo_rgb(saved_colour)
                except tk.TclError:
                    continue
                colour.set(saved_colour)
                swatch.configure(
                    bg=saved_colour,
                    activebackground=saved_colour,
                )

    @staticmethod
    def _folder_is_available(variable: tk.Variable) -> bool:
        value = variable.get().strip()
        try:
            return bool(value) and Path(value).is_dir()
        except OSError:
            return False

    def _restore_settings(self) -> None:
        if self._settings_store is None:
            return
        try:
            settings = self._settings_store.load_namespace("classic")
        except (OSError, TypeError, ValueError):
            return
        if not isinstance(settings, dict):
            return

        global_settings = self._settings_section(settings, "global")
        read_mode = global_settings.get("read_mode")
        if read_mode in {mode.value for mode in ReadMode}:
            self.read_mode_var.set(read_mode)
        language = global_settings.get("language")
        if language in LANGUAGE_NAMES:
            self.language_var.set(language)
            self.language_display_var.set(LANGUAGE_NAMES[language])
        mode = self._mode()
        language = self.language_var.get()
        self.band_required_var.set(required_files_text("band", mode, language))
        self.dos_required_var.set(required_files_text("dos", mode, language))
        self.fermi_required_var.set(required_files_text("fermi", mode, language))

        band = self._settings_section(settings, "band")
        self._restore_string_values(
            band,
            {
                "folder": self.band_folder_var,
                "energy_min": self.band_min_var,
                "energy_max": self.band_max_var,
            },
        )
        dos = self._settings_section(settings, "dos")
        self._restore_string_values(
            dos,
            {
                "folder": self.dos_folder_var,
                "energy_min": self.dos_min_var,
                "energy_max": self.dos_max_var,
                "density_min": self.dos_density_min_var,
                "density_max": self.dos_density_max_var,
            },
        )
        combined = self._settings_section(settings, "combined")
        self._restore_string_values(
            combined,
            {
                "band_folder": self.combined_band_folder_var,
                "dos_folder": self.combined_dos_folder_var,
                "energy_min": self.combined_min_var,
                "energy_max": self.combined_max_var,
                "dos_density_min": self.combined_dos_density_min_var,
                "dos_density_max": self.combined_dos_density_max_var,
            },
        )
        for key, variable in (
            ("band_read_mode", self.combined_band_read_mode_var),
            ("dos_read_mode", self.combined_dos_read_mode_var),
        ):
            value = combined.get(key)
            if value in {mode.value for mode in ReadMode}:
                variable.set(value)
        self.combined_required_var.set(self._combined_required_text())

        paired = self._settings_section(settings, "paired_band")
        self._restore_string_values(
            paired,
            {
                "left_folder": self.paired_left_folder_var,
                "right_folder": self.paired_right_folder_var,
                "energy_min": self.paired_min_var,
                "energy_max": self.paired_max_var,
            },
        )
        for key, variable in (
            ("left_mode", self.paired_left_mode_var),
            ("right_mode", self.paired_right_mode_var),
        ):
            value = paired.get(key)
            if value in {mode.value for mode in ReadMode}:
                variable.set(value)

        fermi = self._settings_section(settings, "fermi")
        self._restore_string_values(
            fermi,
            {
                "folder": self.fermi_folder_var,
                "energy_offset": self.fermi_energy_offset_var,
                "slice_position": self.fermi_slice_position_var,
                "elevation": self.fermi_elevation_var,
                "azimuth": self.fermi_azimuth_var,
            },
        )
        for key, variable, choices in (
            ("view_mode", self.fermi_view_mode_var, {"surface", "slice"}),
            ("spin", self.fermi_spin_var, {"all", "total", "up", "down"}),
            ("slice_axis", self.fermi_slice_axis_var, {"kx", "ky", "kz"}),
        ):
            value = fermi.get(key)
            if value in choices:
                variable.set(value)
        fermi_colour = fermi.get("colour")
        if isinstance(fermi_colour, str):
            try:
                self.winfo_rgb(fermi_colour)
            except tk.TclError:
                pass
            else:
                self.fermi_colour_var.set(fermi_colour)
                self.fermi_colour_swatch.configure(
                    bg=fermi_colour,
                    activebackground=fermi_colour,
                )

        export = self._settings_section(settings, "export")
        self._restore_string_values(
            export,
            {
                "folder": self.output_dir_var,
                "dpi": self.dpi_var,
                "width_mm": self.width_var,
            },
        )
        formats = export.get("formats")
        if isinstance(formats, dict):
            for suffix, variable in self.format_vars.items():
                value = formats.get(suffix)
                if isinstance(value, bool):
                    variable.set(value)

        saved_band_data_type = band.get("data_type")
        if saved_band_data_type in {"complete", "projected"}:
            self._projection_preferences["band_data_type"] = (
                saved_band_data_type
            )
        saved_dos_projected = dos.get("projected")
        if isinstance(saved_dos_projected, bool):
            self._projection_preferences["dos_projected"] = (
                saved_dos_projected
            )
        saved_combined_band_projected = combined.get("band_projected")
        if isinstance(saved_combined_band_projected, bool):
            self._projection_preferences["combined_band_projected"] = (
                saved_combined_band_projected
            )
        saved_combined_dos_projected = combined.get("dos_projected")
        if isinstance(saved_combined_dos_projected, bool):
            self._projection_preferences["combined_dos_projected"] = (
                saved_combined_dos_projected
            )

        for variable, callback in (
            (self.band_folder_var, self._choose_band_folder),
            (self.dos_folder_var, self._choose_dos_folder),
            (self.combined_band_folder_var, self._choose_combined_band),
            (self.combined_dos_folder_var, self._choose_combined_dos),
            (
                self.paired_left_folder_var,
                lambda choose=False: self._choose_paired_folder("left", choose),
            ),
            (
                self.paired_right_folder_var,
                lambda choose=False: self._choose_paired_folder("right", choose),
            ),
            (self.fermi_folder_var, self._choose_fermi_folder),
        ):
            if self._folder_is_available(variable):
                callback(False)

        projection_style = band.get("projection_style")
        if projection_style in {"RGB", "Fat band"}:
            self.band_projection_style_var.set(projection_style)
        complete_band_colour = band.get("complete_band_colour")
        if isinstance(complete_band_colour, str):
            try:
                self.winfo_rgb(complete_band_colour)
            except tk.TclError:
                self.band_complete_colour_var.set("#000000")
                self.band_complete_colour_swatch.configure(
                    bg="#000000",
                    activebackground="#000000",
                )
            else:
                self.band_complete_colour_var.set(complete_band_colour)
                self.band_complete_colour_swatch.configure(
                    bg=complete_band_colour,
                    activebackground=complete_band_colour,
                )
        data_type = self._projection_preferences["band_data_type"]
        if data_type == "projected" and self.band_projection_available:
            self.band_data_type_var.set("projected")
        elif data_type == "complete":
            self.band_data_type_var.set("complete")
        self._refresh_band_projection_state()

        dos_projection_available = str(
            self.dos_projection_toggle.cget("state")
        ) != tk.DISABLED
        self.dos_projection_enabled.set(
            self._projection_preferences["dos_projected"]
            and dos_projection_available
        )
        self._set_projection_rows(
            self.dos_projection_rows,
            dos_projection_available,
            self._projection_states["dos"][1],
        )

        combined_style = combined.get("projection_style")
        if combined_style in {"RGB", "Fat band"}:
            self.combined_projection_style_var.set(combined_style)
        combined_complete_colour = combined.get("complete_band_colour")
        if isinstance(combined_complete_colour, str):
            try:
                self.winfo_rgb(combined_complete_colour)
            except tk.TclError:
                self.combined_complete_colour_var.set("#000000")
                self.combined_complete_colour_swatch.configure(
                    bg="#000000",
                    activebackground="#000000",
                )
            else:
                self.combined_complete_colour_var.set(combined_complete_colour)
                self.combined_complete_colour_swatch.configure(
                    bg=combined_complete_colour,
                    activebackground=combined_complete_colour,
                )
        self.combined_band_projected_var.set(
            self._projection_preferences["combined_band_projected"]
            and self.combined_band_projection_available
        )
        self.combined_dos_projected_var.set(
            self._projection_preferences["combined_dos_projected"]
            and self.combined_dos_projection_available
        )
        self._refresh_combined_projection_state()
        self._refresh_fermi_controls()

        self._restore_projection_settings("band", band.get("channels"))
        self._restore_projection_settings("dos", dos.get("channels"))
        self._restore_projection_settings(
            "combined",
            combined.get("channels"),
        )

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        save_error = self._save_settings()
        if save_error:
            messagebox.showwarning(
                "设置未保存",
                f"Lyra 无法保存本次设置：\n{save_error}",
                parent=self,
            )
        self._close_band_inspector()
        if self._paired_band_inspector is not None:
            self._paired_band_inspector.close()
            self._paired_band_inspector = None
        self.destroy()

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        for attribute in (
            "_local_zoom_window",
            "_preview_viewer",
            "_paired_band_inspector",
            "_remote_window",
            "_incar_window",
            "_mobile_window",
        ):
            window = getattr(self, attribute, None)
            if window is None:
                continue
            try:
                close = getattr(window, "close", None)
                close() if callable(close) else window.destroy()
            except (RuntimeError, tk.TclError):
                pass
            setattr(self, attribute, None)
        for attribute in (
            "_poll_after_id",
            "_backdrop_after_id",
            "_preview_resize_after_id",
        ):
            after_id = getattr(self, attribute, None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
                setattr(self, attribute, None)
        if hasattr(self, "worker"):
            self.worker.close()
        super().destroy()

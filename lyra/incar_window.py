from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from lyra.incar import IncarAssistant, IncarKnowledgeBase
from lyra.theme import COLORS, FONTS


MAX_INCAR_BYTES = 8 * 1024 * 1024


def resolve_incar_path(path: str | Path, *, max_bytes: int = MAX_INCAR_BYTES) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "INCAR"
    if not candidate.is_file():
        raise FileNotFoundError(f"未找到 INCAR：{candidate}")
    if candidate.stat().st_size > max_bytes:
        raise ValueError(f"INCAR is too large ({candidate.stat().st_size:,} bytes)")
    return candidate.resolve()


def format_incar_record(record, *, prefix: str = "", selected_value: str | None = None) -> str:
    body = [prefix, record.tag]
    if record.title_zh:
        body.append(record.title_zh)
    body.append("")
    body.append(record.summary_zh or "当前仅保存官方索引，尚无经过整理的中文正文。")
    if record.details_zh:
        body.extend(("", record.details_zh))
    if record.value_meanings:
        selected = record.meaning_for_value(selected_value) if selected_value is not None else None
        body.extend(("", "具体取值（依据上述官方修订）："))
        for meaning in record.value_meanings:
            marker = "▶" if meaning is selected else " "
            body.append(f"{marker} {record.tag} = {meaning.value}：{meaning.summary_zh}")
            if meaning.details_zh:
                body.append(f"    {meaning.details_zh}")
    body.extend(
        (
            "",
            f"内容来源：{record.content_origin}",
            f"当前离线官方索引修订：{record.source.revision_id} / {record.source.revision_timestamp}",
            f"当前索引抓取日期：{record.source.fetched_at}",
            f"许可：{record.source.license_name}",
            f"来源：{record.source.url}",
        )
    )
    if record.source.translation_curated_at:
        body.extend(
            (
                f"中文整理日期：{record.source.translation_curated_at}",
                "中文整理所依据的索引抓取日期："
                f"{record.source.translation_basis_fetched_at}",
            )
        )
    if record.source.translation_basis_revision_id is not None:
        body.append(
            "中文正文依据的官方修订："
            f"{record.source.translation_basis_revision_id} / "
            f"{record.source.translation_basis_revision_timestamp}"
        )
    return "\n".join(body)


class IncarAssistantWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, *, settings_store) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        self.knowledge = IncarKnowledgeBase.bundled()
        self.assistant = IncarAssistant(self.knowledge)
        shared = settings_store.load_namespace("shared")
        saved = shared.get("incar", {}) if isinstance(shared, dict) else {}
        self.path_var = tk.StringVar(
            value=str(saved.get("last_path", "")) if isinstance(saved, dict) else ""
        )
        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value=(
                f"离线知识库 {len(self.knowledge)} 项 | "
                f"抓取 {self.knowledge.dataset.fetched_at}"
            )
        )
        self._analysis = None
        self._selected_source_url = ""

        self.title("Lyra - INCAR 助手")
        self.geometry("1180x760")
        self.minsize(900, 620)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self._build()
        self.search_var.trace_add("write", lambda *_args: self._refresh_search())
        self._refresh_search()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="INCAR 助手", font=FONTS["section"]).pack(anchor="w")
        ttk.Label(
            shell,
            text=(
                "中文解释与风险规则由 Lyra 根据 VASP Wiki 整理，不是 VASP 官方建议；"
                "每项均保留官方页面与修订证据。"
            ),
            foreground=COLORS["muted"],
            wraplength=1080,
        ).pack(anchor="w", pady=(3, 10))

        source = ttk.Frame(shell)
        source.pack(fill="x")
        ttk.Entry(source, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(source, text="选择 INCAR", command=self._choose_file).pack(side="left", padx=(6, 0))
        ttk.Button(source, text="选择计算文件夹", command=self._choose_folder).pack(side="left", padx=(6, 0))
        ttk.Button(source, text="分析", style="Primary.TButton", command=self._analyse).pack(side="left", padx=(6, 0))

        search = ttk.Frame(shell)
        search.pack(fill="x", pady=(10, 8))
        ttk.Label(search, text="搜索标签或中文关键词").pack(side="left")
        ttk.Entry(search, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        panes = ttk.Panedwindow(shell, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.Notebook(panes)
        right = ttk.Frame(panes, padding=(12, 6))
        panes.add(left, weight=5)
        panes.add(right, weight=6)

        assignment_tab = ttk.Frame(left, padding=6)
        dictionary_tab = ttk.Frame(left, padding=6)
        finding_tab = ttk.Frame(left, padding=6)
        left.add(assignment_tab, text="当前 INCAR")
        left.add(dictionary_tab, text="标签词典")
        left.add(finding_tab, text="风险与诊断")

        self.assignment_tree = ttk.Treeview(
            assignment_tab,
            columns=("line", "tag", "value", "state"),
            show="headings",
        )
        for key, text, width in (
            ("line", "行", 55),
            ("tag", "标签", 120),
            ("value", "值", 220),
            ("state", "状态", 80),
        ):
            self.assignment_tree.heading(key, text=text)
            self.assignment_tree.column(key, width=width, stretch=key == "value")
        self.assignment_tree.pack(fill="both", expand=True)
        self.assignment_tree.bind("<<TreeviewSelect>>", self._assignment_selected)

        self.dictionary_tree = ttk.Treeview(
            dictionary_tab,
            columns=("tag", "title", "category"),
            show="headings",
        )
        for key, text, width in (
            ("tag", "标签", 130),
            ("title", "中文说明", 230),
            ("category", "分类", 110),
        ):
            self.dictionary_tree.heading(key, text=text)
            self.dictionary_tree.column(key, width=width, stretch=key == "title")
        self.dictionary_tree.pack(fill="both", expand=True)
        self.dictionary_tree.bind("<<TreeviewSelect>>", self._dictionary_selected)

        self.finding_tree = ttk.Treeview(
            finding_tab,
            columns=("severity", "title", "tags"),
            show="headings",
        )
        for key, text, width in (
            ("severity", "级别", 70),
            ("title", "结论", 260),
            ("tags", "相关标签", 120),
        ):
            self.finding_tree.heading(key, text=text)
            self.finding_tree.column(key, width=width, stretch=key == "title")
        self.finding_tree.pack(fill="both", expand=True)
        self.finding_tree.bind("<<TreeviewSelect>>", self._finding_selected)

        ttk.Label(right, text="说明与证据", font=FONTS["section"]).pack(anchor="w")
        self.detail = tk.Text(
            right,
            wrap="word",
            borderwidth=1,
            relief="solid",
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            font=FONTS["body"],
            padx=12,
            pady=10,
            state="disabled",
        )
        self.detail.pack(fill="both", expand=True, pady=(8, 8))
        self.source_button = ttk.Button(
            right,
            text="打开 VASP Wiki 原始页面",
            command=self._open_source,
            state="disabled",
        )
        self.source_button.pack(anchor="e")
        ttk.Label(shell, textvariable=self.status_var, foreground=COLORS["accent"]).pack(fill="x", pady=(8, 0))

    def _choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="选择 INCAR",
            filetypes=(("VASP INCAR", "INCAR*"), ("全部文件", "*.*")),
        )
        if selected:
            self.path_var.set(selected)
            self._analyse()

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="选择 VASP 计算文件夹")
        if selected:
            self.path_var.set(selected)
            self._analyse()

    def _analyse(self) -> None:
        try:
            path = resolve_incar_path(self.path_var.get())
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
                encoding_note = "UTF-8"
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
                encoding_note = "兼容单字节编码"
            analysis = self.assistant.analyse(text)
        except (OSError, ValueError) as exc:
            messagebox.showerror("INCAR 读取失败", str(exc), parent=self)
            return
        self._analysis = analysis
        self.path_var.set(str(path))
        for item in self.assignment_tree.get_children():
            self.assignment_tree.delete(item)
        for entry in analysis.entries:
            assignment = entry.assignment
            state = "生效" if assignment.is_effective else "被覆盖"
            if entry.record is None:
                state = f"{state} / 未知"
            self.assignment_tree.insert(
                "",
                "end",
                iid=str(assignment.ordinal),
                values=(assignment.line_start, assignment.tag, assignment.value, state),
            )
        self._refresh_findings()
        self.settings_store.save_namespace("shared", {"incar": {"last_path": str(path)}})
        self.status_var.set(
            f"已解析 {len(analysis.entries)} 条设置 | 未知 {len(analysis.unknown_tags)} | "
            f"规则提示 {len(analysis.findings)} | {encoding_note}"
        )
        children = self.assignment_tree.get_children()
        if children:
            self.assignment_tree.selection_set(children[0])
            self._assignment_selected()

    def _refresh_search(self) -> None:
        query = self.search_var.get().strip()
        records = self.knowledge.search(query, limit=200) if query else self.knowledge.records[:200]
        for item in self.dictionary_tree.get_children():
            self.dictionary_tree.delete(item)
        for record in records:
            self.dictionary_tree.insert(
                "",
                "end",
                iid=record.tag,
                values=(record.tag, record.title_zh or "仅官方索引", record.category_zh),
            )

    def _refresh_findings(self) -> None:
        for item in self.finding_tree.get_children():
            self.finding_tree.delete(item)
        if self._analysis is None:
            return
        for index, finding in enumerate(self._analysis.findings):
            self.finding_tree.insert(
                "",
                "end",
                iid=f"finding-{index}",
                values=(finding.severity, finding.title_zh, ", ".join(finding.tags)),
            )
        for index, diagnostic in enumerate(self._analysis.document.diagnostics):
            self.finding_tree.insert(
                "",
                "end",
                iid=f"diagnostic-{index}",
                values=("解析", diagnostic.message_zh, f"第 {diagnostic.line_start} 行"),
            )

    def _assignment_selected(self, _event=None) -> None:
        if self._analysis is None or not self.assignment_tree.selection():
            return
        ordinal = int(self.assignment_tree.selection()[0])
        entry = next(item for item in self._analysis.entries if item.assignment.ordinal == ordinal)
        prefix = (
            f"{entry.assignment.tag} = {entry.assignment.value}\n"
            f"位置：第 {entry.assignment.line_start}-{entry.assignment.line_end} 行\n"
            f"状态：{'最终生效' if entry.assignment.is_effective else '被后续同名设置覆盖'}\n\n"
        )
        self._show_record(
            entry.record,
            prefix=prefix,
            selected_value=entry.assignment.value,
        )

    def _dictionary_selected(self, _event=None) -> None:
        selected = self.dictionary_tree.selection()
        if selected:
            self._show_record(self.knowledge.get(selected[0]))

    def _finding_selected(self, _event=None) -> None:
        if self._analysis is None or not self.finding_tree.selection():
            return
        item_id = self.finding_tree.selection()[0]
        if item_id.startswith("finding-"):
            finding = self._analysis.findings[int(item_id.split("-")[1])]
            text = (
                f"{finding.title_zh}\n\n{finding.message_zh}\n\n"
                f"相关标签：{', '.join(finding.tags)}\n"
                "来源类型：Lyra 辅助规则，不是 VASP 官方结论"
            )
            self._selected_source_url = finding.source_urls[0] if finding.source_urls else ""
            self._set_detail(text)
            return
        diagnostic = self._analysis.document.diagnostics[int(item_id.split("-")[1])]
        self._selected_source_url = ""
        self._set_detail(
            f"{diagnostic.message_zh}\n\n第 {diagnostic.line_start}-{diagnostic.line_end} 行\n"
            f"原始片段：{diagnostic.fragment}"
        )

    def _show_record(
        self,
        record,
        *,
        prefix: str = "",
        selected_value: str | None = None,
    ) -> None:
        if record is None:
            self._selected_source_url = ""
            self._set_detail(prefix + "Lyra 离线索引中没有这个标签，请检查拼写和 VASP 版本。")
            return
        self._selected_source_url = record.source.url
        self._set_detail(
            format_incar_record(
                record,
                prefix=prefix,
                selected_value=selected_value,
            )
        )

    def _set_detail(self, text: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")
        self.source_button.configure(state="normal" if self._selected_source_url else "disabled")

    def _open_source(self) -> None:
        if self._selected_source_url:
            webbrowser.open(self._selected_source_url)

    def close(self) -> None:
        try:
            self.destroy()
        except tk.TclError:
            pass

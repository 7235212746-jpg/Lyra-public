from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path, PurePosixPath
from queue import Empty, SimpleQueue
import re
import tempfile
from threading import Thread
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from lyra.remote import (
    CredentialNotFoundError,
    CredentialStore,
    CredentialStoreError,
    ProfileStore,
    RemoteBrowser,
    RemoteCredentials,
    RemoteProfile,
    RemoteRunner,
    RuntimeAuthentication,
    build_calculation_manifest,
    connect_pinned,
    load_remote_workspace_preferences,
    RemoteNavigation,
    RemoteWorkspacePreferences,
    save_remote_workspace_preferences,
)
from lyra.remote.manifest import CalculationManifest
from lyra.remote.transport import RemoteStat, validate_remote_path
from lyra.remote.protocol import RemoteTaskState
from lyra.remote.winscp import WinScpInstallation, WinScpTransport
from lyra.theme import COLORS, FONTS
from lyra.tool_capabilities import ToolCapabilities


WORKFLOW_LABELS = {
    "band": "完整能带",
    "projected_band": "投影能带",
    "dos": "DOS / PDOS",
    "fermi_surface": "费米面",
    "magnetic_export": "VESTA 磁矩",
}
_UNSET = object()
_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def available_workflows(manifest: CalculationManifest) -> tuple[str, ...]:
    return tuple(
        name
        for name in WORKFLOW_LABELS
        if manifest.capability(name).available
    )


def format_modified_time(modified_ns: int) -> str:
    if modified_ns <= 0:
        return ""
    return datetime.fromtimestamp(modified_ns / 1_000_000_000).astimezone().strftime(
        "%Y-%m-%d %H:%M"
    )


def format_task_progress(item_name: str, index: int, total: int) -> str:
    return f"正在同步 {item_name}（{index}/{total}）"


def workflow_supports_preview(workflow: str) -> bool:
    return workflow in {"band", "projected_band", "dos", "fermi_surface"}


def workflow_supports_remote_analysis(workflow: str) -> bool:
    return workflow in {"band", "projected_band", "dos"}


def breadcrumb_paths(path: str, *, max_parts: int = 4) -> tuple[tuple[str, str], ...]:
    normalized = validate_remote_path(path)
    parts = PurePosixPath(normalized).parts[1:]
    result: list[tuple[str, str]] = [("/", "/")]
    hidden_count = max(0, len(parts) - max_parts)
    if hidden_count:
        hidden_target = str(PurePosixPath("/").joinpath(*parts[:hidden_count]))
        result.append(("…", hidden_target))
    start = hidden_count
    for index in range(start, len(parts)):
        target = str(PurePosixPath("/").joinpath(*parts[: index + 1]))
        result.append((parts[index], target))
    return tuple(result)


def _local_app_root() -> Path:
    value = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    return (Path(value) if value else Path.home() / ".lyra") / "Lyra"


def credential_store_path(root: str | Path, profile_id: str) -> Path:
    if not isinstance(profile_id, str) or not _PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise ValueError("远程 profile ID 不安全。")
    base = Path(root).expanduser().resolve()
    target = base / "remote_credentials" / f"{profile_id}.json"
    if not target.is_relative_to(base):
        raise ValueError("远程凭据路径超出 Lyra 用户目录。")
    return target


def resolve_runtime_authentication(
    password: str,
    private_key_passphrase: str,
    saved: RemoteCredentials | None,
) -> RuntimeAuthentication:
    entered_password = password or None
    entered_passphrase = private_key_passphrase or None
    if entered_password is not None or entered_passphrase is not None:
        return RuntimeAuthentication(entered_password, entered_passphrase)
    if saved is not None:
        return RuntimeAuthentication(
            saved.password,
            saved.private_key_passphrase,
        )
    return RuntimeAuthentication()


def selected_sync_files(
    manifest: CalculationManifest,
    workflow: str,
) -> tuple[RemoteStat, ...]:
    capability = manifest.capability(workflow)
    if not capability.available:
        raise ValueError("远程目录缺少：" + ", ".join(capability.missing_files))
    required = set(capability.required_files)
    selected = tuple(item for item in manifest.files if item.name in required)
    if workflow != "wavecar_evidence" and any(item.name == "WAVECAR" for item in selected):
        raise ValueError("普通绘图不会自动下载 WAVECAR。")
    return selected


def cache_directory(
    root: str | Path,
    profile_id: str,
    manifest: CalculationManifest,
) -> Path:
    if not profile_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in profile_id):
        raise ValueError("远程 profile ID 不安全。")
    target = Path(root).expanduser().resolve() / profile_id / manifest.fingerprint
    target.mkdir(parents=True, exist_ok=True)
    return target


class RemoteWorkspaceWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        settings_store,
        capabilities: ToolCapabilities,
    ) -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.settings_store = settings_store
        self.capabilities = capabilities
        self.profile_store = ProfileStore()
        self.credential_root = _local_app_root()
        self.profiles = list(self.profile_store.load())
        self.workspace_preferences = load_remote_workspace_preferences(settings_store)
        self.navigation = RemoteNavigation(self.workspace_preferences.last_path)
        self.bookmarks = list(self.workspace_preferences.bookmarks)
        self.transport: WinScpTransport | None = None
        self.session = None
        self.remote_runner: RemoteRunner | None = None
        self.remote_job_id = None
        self._remote_preview_directory = tempfile.TemporaryDirectory(
            prefix="lyra-remote-preview-"
        )
        self.entries: tuple[RemoteStat, ...] = ()
        self.manifest: CalculationManifest | None = None
        self._queue = SimpleQueue()
        self._poll_id = None
        self._closing = False
        self._request_generation = 0
        self._threads: set[Thread] = set()
        self._threads_lock = threading.Lock()
        self._busy = False
        self._retry_action = None
        self._task_details_visible = False

        self.profile_var = tk.StringVar(value=self.workspace_preferences.last_profile)
        self.name_var = tk.StringVar(value="Linux 集群")
        self.host_var = tk.StringVar()
        self.port_var = tk.StringVar(value="22")
        self.user_var = tk.StringVar()
        self.fingerprint_var = tk.StringVar()
        self.key_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.key_passphrase_var = tk.StringVar()
        self.remember_credentials_var = tk.BooleanVar(value=False)
        self.credential_status_var = tk.StringVar(value="凭据仅保存在本机内存")
        self.path_var = tk.StringVar(value=self.workspace_preferences.last_path)
        self.workflow_var = tk.StringVar(value=self.workspace_preferences.workflow)
        self.editor_expanded_var = tk.BooleanVar(
            value=self.workspace_preferences.editor_expanded
        )
        self.status_var = tk.StringVar(value=capabilities.remote_summary)
        self.capability_var = tk.StringVar(value="尚未读取计算目录")
        self.connection_var = tk.StringVar(value="未连接")
        self.profile_summary_var = tk.StringVar(value="选择服务器后连接")
        self.detected_files_var = tk.StringVar(value="尚未读取文件")
        self.task_detail_var = tk.StringVar(value="当前没有远程任务")
        self.error_var = tk.StringVar()

        self.title("Lyra - 远程 Linux 工作台")
        self.geometry("1280x800")
        self.minsize(1020, 650)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self._build()
        self._refresh_profile_choices()
        self._refresh_action_states()
        self._schedule_poll()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="远程 Linux", font=FONTS["section"]).pack(side="left")
        ttk.Label(
            header,
            textvariable=self.connection_var,
            foreground=COLORS["accent"],
        ).pack(side="right")
        ttk.Label(
            shell,
            text="主机指纹必须先核验；密码和私钥口令不会写入 Lyra。普通绘图不会自动下载 WAVECAR。",
            foreground=COLORS["muted"],
        ).grid(row=1, column=0, sticky="w", pady=(3, 10))

        body = ttk.Panedwindow(shell, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew")

        rail_host = ttk.Frame(body, width=275)
        rail_host.rowconfigure(0, weight=1)
        rail_host.columnconfigure(0, weight=1)
        body.add(rail_host, weight=0)
        self.server_canvas = tk.Canvas(
            rail_host,
            width=250,
            background=COLORS["background"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.server_scrollbar = ttk.Scrollbar(
            rail_host,
            orient="vertical",
            command=self.server_canvas.yview,
        )
        self.server_canvas.configure(yscrollcommand=self.server_scrollbar.set)
        self.server_canvas.grid(row=0, column=0, sticky="nsew")
        self.server_scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        rail = ttk.Frame(self.server_canvas, padding=(0, 0, 10, 0))
        self._server_rail_window = self.server_canvas.create_window(
            (0, 0),
            window=rail,
            anchor="nw",
        )
        rail.bind("<Configure>", self._server_rail_content_resized, add="+")
        self.server_canvas.bind(
            "<Configure>", self._server_rail_canvas_resized, add="+"
        )
        self.bind("<MouseWheel>", self._scroll_server_rail, add="+")
        ttk.Label(rail, text="服务器", font=FONTS["section"]).pack(anchor="w")
        profile_row = ttk.Frame(rail)
        profile_row.pack(fill="x", pady=(8, 0))
        self.profile_box = ttk.Combobox(
            profile_row,
            textvariable=self.profile_var,
            state="readonly",
            width=21,
        )
        self.profile_box.pack(side="left", fill="x", expand=True)
        self.profile_box.bind("<<ComboboxSelected>>", self._profile_selected)
        self.new_profile_button = ttk.Button(
            profile_row,
            text="新建",
            command=self._new_profile,
        )
        self.new_profile_button.pack(side="left", padx=(6, 0))
        ttk.Label(
            rail,
            textvariable=self.profile_summary_var,
            foreground=COLORS["muted"],
            wraplength=230,
            justify="left",
        ).pack(fill="x", pady=(6, 8))
        connection_actions = ttk.Frame(rail)
        connection_actions.pack(fill="x")
        self.connect_button = ttk.Button(
            connection_actions,
            text="连接",
            style="Primary.TButton",
            command=self._connect,
        )
        self.connect_button.pack(side="left", fill="x", expand=True)
        self.disconnect_button = ttk.Button(
            connection_actions,
            text="断开",
            command=self._disconnect,
            state="disabled",
        )
        self.disconnect_button.pack(side="left", padx=(6, 0))
        self.editor_toggle_button = ttk.Button(
            rail,
            text="编辑连接配置",
            command=self._toggle_editor,
        )
        self.editor_toggle_button.pack(fill="x", pady=(8, 0))

        self.profile_editor = ttk.Frame(rail)
        self.profile_editor.columnconfigure(0, weight=1)
        self._labeled_entry(self.profile_editor, 0, "名称", self.name_var)
        host_row = ttk.Frame(self.profile_editor)
        host_row.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        host_row.columnconfigure(0, weight=1)
        ttk.Label(host_row, text="主机").grid(row=0, column=0, sticky="w")
        ttk.Label(host_row, text="端口").grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Entry(host_row, textvariable=self.host_var).grid(row=1, column=0, sticky="ew")
        ttk.Entry(host_row, textvariable=self.port_var, width=7).grid(row=1, column=1, padx=(6, 0))
        self._labeled_entry(self.profile_editor, 3, "用户", self.user_var)
        key_row = ttk.Frame(self.profile_editor)
        key_row.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        key_row.columnconfigure(0, weight=1)
        ttk.Label(key_row, text="私钥").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(key_row, textvariable=self.key_var).grid(row=1, column=0, sticky="ew")
        ttk.Button(key_row, text="选择", command=self._choose_key).grid(row=1, column=1, padx=(6, 0))
        self._labeled_entry(self.profile_editor, 6, "SHA-256 指纹", self.fingerprint_var)
        self._labeled_entry(self.profile_editor, 8, "SSH 密码", self.password_var, show="●")
        self._labeled_entry(self.profile_editor, 10, "私钥口令", self.key_passphrase_var, show="●")
        ttk.Checkbutton(
            self.profile_editor,
            text="使用 Windows 安全保存",
            variable=self.remember_credentials_var,
        ).grid(row=12, column=0, sticky="w", pady=(7, 0))
        ttk.Label(
            self.profile_editor,
            textvariable=self.credential_status_var,
            foreground=COLORS["muted"],
            wraplength=225,
        ).grid(row=13, column=0, sticky="w", pady=(3, 0))
        editor_actions = ttk.Frame(self.profile_editor)
        editor_actions.grid(row=14, column=0, sticky="ew", pady=(8, 0))
        self.scan_button = ttk.Button(editor_actions, text="扫描指纹", command=self._scan_fingerprint)
        self.scan_button.pack(side="left")
        self.save_profile_button = ttk.Button(
            editor_actions,
            text="保存配置",
            command=self._save_profile,
        )
        self.save_profile_button.pack(side="left", padx=(6, 0))

        self.rail_separator = ttk.Separator(rail)
        self.rail_separator.pack(fill="x", pady=12)
        ttk.Label(rail, text="收藏目录").pack(anchor="w")
        self.bookmark_list = tk.Listbox(
            rail,
            height=7,
            exportselection=False,
            borderwidth=1,
            highlightthickness=0,
        )
        self.bookmark_list.pack(fill="both", expand=True, pady=(6, 0))
        self.bookmark_list.bind("<Double-Button-1>", self._open_selected_bookmark)
        self.bookmark_list.bind("<Return>", self._open_selected_bookmark)

        workspace = ttk.Frame(body)
        body.add(workspace, weight=1)
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(workspace)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.back_button = ttk.Button(toolbar, text="返回", width=6, command=self._go_back, state="disabled")
        self.back_button.pack(side="left")
        self.forward_button = ttk.Button(toolbar, text="前进", width=6, command=self._go_forward, state="disabled")
        self.forward_button.pack(side="left", padx=(4, 0))
        self.up_button = ttk.Button(toolbar, text="上一级", command=self._go_parent, state="disabled")
        self.up_button.pack(side="left", padx=(4, 0))
        self.home_button = ttk.Button(toolbar, text="主页", command=self._go_home, state="disabled")
        self.home_button.pack(side="left", padx=(4, 0))
        self.location_holder = ttk.Frame(toolbar)
        self.location_holder.pack(side="left", fill="x", expand=True, padx=8)
        self.breadcrumb_frame = ttk.Frame(self.location_holder)
        self.breadcrumb_frame.pack(fill="x")
        self.path_entry = ttk.Entry(self.location_holder, textvariable=self.path_var)
        self.path_entry.bind("<Return>", self._commit_location)
        self.path_entry.bind("<Escape>", self._cancel_location)
        self.list_button = ttk.Button(toolbar, text="刷新", command=lambda: self._list_directory(record=False), state="disabled")
        self.list_button.pack(side="left")
        self.bookmark_button = ttk.Button(toolbar, text="收藏", command=self._toggle_bookmark, state="disabled")
        self.bookmark_button.pack(side="left", padx=(4, 0))

        content = ttk.Panedwindow(workspace, orient="horizontal")
        content.grid(row=1, column=0, sticky="nsew")
        browser = ttk.Frame(content)
        browser.columnconfigure(0, weight=1)
        browser.rowconfigure(0, weight=1)
        content.add(browser, weight=3)

        self.file_tree = ttk.Treeview(
            browser,
            columns=("name", "type", "size", "modified"),
            show="headings",
            selectmode="browse",
        )
        self.file_tree.heading("name", text="名称")
        self.file_tree.heading("type", text="类型")
        self.file_tree.heading("size", text="大小")
        self.file_tree.heading("modified", text="修改时间")
        self.file_tree.column("name", width=410, minwidth=180)
        self.file_tree.column("type", width=90, anchor="center")
        self.file_tree.column("size", width=130, anchor="e")
        self.file_tree.column("modified", width=145, anchor="center")
        tree_scroll = ttk.Scrollbar(browser, orient="vertical", command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.file_tree.bind("<Double-Button-1>", self._open_selected_directory)
        self.file_tree.bind("<Return>", self._open_selected_directory)
        self.file_tree.bind("<BackSpace>", self._go_parent)

        inspector = ttk.Frame(content, padding=(12, 0, 0, 0), width=285)
        content.add(inspector, weight=1)
        ttk.Label(inspector, text="计算目录", font=FONTS["section"]).pack(anchor="w")
        ttk.Label(
            inspector,
            textvariable=self.capability_var,
            foreground=COLORS["muted"],
            wraplength=275,
            justify="left",
        ).pack(fill="x", pady=(6, 8))
        ttk.Label(inspector, text="检测到的文件").pack(anchor="w")
        ttk.Label(
            inspector,
            textvariable=self.detected_files_var,
            foreground=COLORS["muted"],
        ).pack(anchor="w", pady=(2, 0))
        self.detected_tree = ttk.Treeview(
            inspector,
            columns=("file", "size"),
            show="headings",
            height=9,
        )
        self.detected_tree.heading("file", text="文件")
        self.detected_tree.heading("size", text="大小")
        self.detected_tree.column("file", width=160)
        self.detected_tree.column("size", width=80, anchor="e")
        self.detected_tree.pack(fill="x", pady=(5, 12))
        ttk.Label(inspector, text="用于").pack(anchor="w")
        self.workflow_buttons = {}
        for name, label in WORKFLOW_LABELS.items():
            button = ttk.Radiobutton(
                inspector,
                text=label,
                value=name,
                variable=self.workflow_var,
                command=self._workflow_changed,
                state="disabled",
            )
            button.pack(anchor="w", pady=(5, 0))
            self.workflow_buttons[name] = button
        self.preview_button = ttk.Button(
            inspector,
            text="远端解析并预览",
            style="Primary.TButton",
            command=self._remote_preview,
            state="disabled",
        )
        self.preview_button.pack(fill="x", pady=(14, 0))
        self.cancel_remote_button = ttk.Button(
            inspector,
            text="取消远端任务",
            command=self._cancel_remote_job,
            state="disabled",
        )
        self.cancel_remote_button.pack(fill="x", pady=(6, 0))
        self.sync_button = ttk.Button(
            inspector,
            text="同步到本地（备用）",
            command=self._sync,
            state="disabled",
        )
        self.sync_button.pack(fill="x", pady=(6, 0))

        task_bar = ttk.Frame(shell)
        task_bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Separator(task_bar).pack(fill="x", pady=(0, 8))
        ttk.Label(task_bar, textvariable=self.status_var, foreground=COLORS["accent"]).pack(side="left", fill="x", expand=True)
        self.task_toggle_button = ttk.Button(task_bar, text="任务详情", command=self._toggle_task_details)
        self.task_toggle_button.pack(side="right")
        self.task_details = ttk.Frame(shell)
        self.task_details.columnconfigure(0, weight=1)
        self.error_label = ttk.Label(
            self.task_details,
            textvariable=self.error_var,
            foreground=COLORS["warning"],
            wraplength=1050,
            justify="left",
        )
        self.error_label.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            self.task_details,
            textvariable=self.task_detail_var,
            foreground=COLORS["muted"],
            wraplength=1100,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(3, 0))
        self.retry_button = ttk.Button(
            self.task_details,
            text="重试",
            command=self._retry_last_action,
            state="disabled",
        )
        self.retry_button.grid(row=0, column=1, rowspan=2, padx=(8, 0))

        self._apply_editor_visibility()
        self._refresh_bookmarks()
        self._render_breadcrumb(self.path_var.get())
        self._bind_shortcuts()

    def _server_rail_content_resized(self, _event=None) -> None:
        self.server_canvas.configure(scrollregion=self.server_canvas.bbox("all"))

    def _server_rail_canvas_resized(self, event) -> None:
        self.server_canvas.itemconfigure(
            self._server_rail_window,
            width=max(1, event.width),
        )

    def _scroll_server_rail(self, event) -> None:
        pointer_x = self.winfo_pointerx()
        pointer_y = self.winfo_pointery()
        left = self.server_canvas.winfo_rootx()
        top = self.server_canvas.winfo_rooty()
        if not (
            left <= pointer_x < left + self.server_canvas.winfo_width()
            and top <= pointer_y < top + self.server_canvas.winfo_height()
        ):
            return
        if event.delta:
            units = -1 if event.delta > 0 else 1
            self.server_canvas.yview_scroll(units * 3, "units")

    @staticmethod
    def _labeled_entry(parent, row, label, variable, *, show=None) -> ttk.Entry:
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=0, sticky="ew", pady=(7, 0))
        holder.columnconfigure(0, weight=1)
        ttk.Label(holder, text=label).grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(holder, textvariable=variable, show=show)
        entry.grid(row=1, column=0, sticky="ew")
        return entry

    def _toggle_editor(self) -> None:
        self.editor_expanded_var.set(not self.editor_expanded_var.get())
        self._apply_editor_visibility()
        self._save_workspace_preferences()

    def _apply_editor_visibility(self) -> None:
        if self.editor_expanded_var.get():
            self.profile_editor.pack(
                fill="x",
                pady=(4, 0),
                before=self.rail_separator,
            )
            self.editor_toggle_button.configure(text="收起连接配置")
        else:
            self.profile_editor.pack_forget()
            self.editor_toggle_button.configure(text="编辑连接配置")

    def _toggle_task_details(self, *, show: bool | None = None) -> None:
        visible = not self._task_details_visible if show is None else show
        self._task_details_visible = visible
        if visible:
            self.task_details.grid(row=4, column=0, sticky="ew", pady=(6, 0))
            self.task_toggle_button.configure(text="收起详情")
        else:
            self.task_details.grid_remove()
            self.task_toggle_button.configure(text="任务详情")

    def _bind_shortcuts(self) -> None:
        self.bind("<Alt-Left>", lambda _event: self._go_back())
        self.bind("<Alt-Right>", lambda _event: self._go_forward())
        self.bind("<Alt-Up>", lambda _event: self._go_parent())
        self.bind("<F5>", lambda _event: self._list_directory(record=False))
        self.bind("<Control-l>", self._focus_location)
        self.bind("<Control-d>", lambda _event: self._toggle_bookmark())

    def _focus_location(self, _event=None):
        if self.session is None:
            return "break"
        self.breadcrumb_frame.pack_forget()
        self.path_entry.pack(fill="x")
        self.path_entry.focus_set()
        self.path_entry.selection_range(0, "end")
        return "break"

    def _commit_location(self, _event=None):
        self._show_breadcrumb()
        self._navigate(self.path_var.get())
        return "break"

    def _cancel_location(self, _event=None):
        self.path_var.set(self.navigation.current)
        self._show_breadcrumb()
        return "break"

    def _show_breadcrumb(self) -> None:
        self.path_entry.pack_forget()
        self._render_breadcrumb(self.path_var.get())
        self.breadcrumb_frame.pack(fill="x")

    def _render_breadcrumb(self, path: str) -> None:
        for child in self.breadcrumb_frame.winfo_children():
            child.destroy()
        try:
            segments = breadcrumb_paths(path)
        except (TypeError, ValueError):
            segments = breadcrumb_paths("/home")
        for index, (label, target) in enumerate(segments):
            if index:
                ttk.Label(self.breadcrumb_frame, text="/").pack(side="left", padx=2)
            button = ttk.Button(
                self.breadcrumb_frame,
                text=label,
                command=lambda value=target: self._navigate(value),
            )
            if label == "/":
                button.configure(width=3)
            button.pack(side="left")

    def _new_profile(self) -> None:
        if self._busy:
            return
        self.profile_var.set("")
        self.name_var.set("Linux 集群")
        self.host_var.set("")
        self.port_var.set("22")
        self.user_var.set("")
        self.fingerprint_var.set("")
        self.key_var.set("")
        self.password_var.set("")
        self.key_passphrase_var.set("")
        self.remember_credentials_var.set(False)
        self.credential_status_var.set("凭据仅保存在本机内存")
        self.profile_summary_var.set("填写新服务器配置")
        self.editor_expanded_var.set(True)
        self._apply_editor_visibility()
        self.status_var.set("正在创建新的服务器配置")
        self._save_workspace_preferences()

    def _refresh_bookmarks(self) -> None:
        self.bookmark_list.delete(0, "end")
        for path in self.bookmarks:
            self.bookmark_list.insert("end", path)
        current = self.path_var.get().strip()
        self.bookmark_button.configure(
            text="取消收藏" if current in self.bookmarks else "收藏"
        )

    def _toggle_bookmark(self) -> str:
        if self.session is None:
            return "break"
        try:
            path = validate_remote_path(self.path_var.get().strip())
        except ValueError as exc:
            self._show_inline_error(str(exc), retry=self._toggle_bookmark)
            return "break"
        if path in self.bookmarks:
            self.bookmarks.remove(path)
            self.status_var.set(f"已取消收藏：{path}")
        elif len(self.bookmarks) < 32:
            self.bookmarks.append(path)
            self.status_var.set(f"已收藏：{path}")
        else:
            self._show_inline_error("收藏目录最多保留 32 项。")
            return "break"
        self._refresh_bookmarks()
        self._save_workspace_preferences()
        return "break"

    def _open_selected_bookmark(self, _event=None):
        selection = self.bookmark_list.curselection()
        if selection:
            self._navigate(self.bookmarks[selection[0]])
        return "break"

    def _save_workspace_preferences(self) -> None:
        preferences = RemoteWorkspacePreferences.from_mapping(
            {
                "last_profile": self.profile_var.get(),
                "last_path": self.path_var.get(),
                "workflow": self.workflow_var.get(),
                "bookmarks": self.bookmarks,
                "editor_expanded": self.editor_expanded_var.get(),
            }
        )
        try:
            save_remote_workspace_preferences(self.settings_store, preferences)
        except (OSError, ValueError):
            return
        self.workspace_preferences = preferences

    def _workflow_changed(self) -> None:
        self._refresh_action_states()
        self._save_workspace_preferences()

    def _show_inline_error(self, message: str, *, retry=None) -> None:
        self.error_var.set(message)
        self.task_detail_var.set(f"操作失败：{message}")
        self.status_var.set("远程操作失败")
        self._retry_action = retry
        self.retry_button.configure(state="normal" if retry is not None else "disabled")
        self._toggle_task_details(show=True)

    def _clear_inline_error(self) -> None:
        self.error_var.set("")
        self._retry_action = None
        self.retry_button.configure(state="disabled")

    def _retry_last_action(self) -> None:
        action = self._retry_action
        if action is None or self._busy:
            return
        self._clear_inline_error()
        action()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._toggle_task_details(show=True)
        self._refresh_action_states()

    def _refresh_action_states(self) -> None:
        connected = self.session is not None
        idle = not self._busy
        self.connect_button.configure(state="normal" if idle and not connected else "disabled")
        self.disconnect_button.configure(state="normal" if idle and connected else "disabled")
        self.new_profile_button.configure(state="normal" if idle and not connected else "disabled")
        self.profile_box.configure(state="readonly" if idle and not connected else "disabled")
        self.scan_button.configure(state="normal" if idle and not connected else "disabled")
        self.save_profile_button.configure(state="normal" if idle and not connected else "disabled")
        self.editor_toggle_button.configure(state="normal" if idle and not connected else "disabled")
        self.list_button.configure(state="normal" if idle and connected else "disabled")
        for button in (self.back_button, self.forward_button, self.up_button, self.home_button, self.bookmark_button):
            button.configure(state="normal" if idle and connected else "disabled")
        if connected and idle:
            self.back_button.configure(state="normal" if self.navigation.can_back else "disabled")
            self.forward_button.configure(state="normal" if self.navigation.can_forward else "disabled")
        available = set(available_workflows(self.manifest)) if self.manifest is not None else set()
        for name, button in self.workflow_buttons.items():
            button.configure(state="normal" if idle and connected and name in available else "disabled")
        selected_available = self.workflow_var.get() in available
        action_state = "normal" if idle and connected and selected_available else "disabled"
        self.sync_button.configure(state=action_state)
        preview_state = (
            "normal"
            if action_state == "normal" and workflow_supports_remote_analysis(self.workflow_var.get())
            else "disabled"
        )
        self.preview_button.configure(state=preview_state)
        self.cancel_remote_button.configure(
            state="normal" if self.remote_job_id is not None else "disabled"
        )

    def _disconnect(self) -> None:
        self._request_generation += 1
        if self.session is not None:
            self.session.close()
            self.session = None
        self.remote_runner = None
        self.remote_job_id = None
        self.entries = ()
        self.manifest = None
        self.connection_var.set("未连接")
        self.status_var.set("连接已断开")
        self.task_detail_var.set("当前没有远程任务")
        for tree in (self.file_tree, self.detected_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.capability_var.set("尚未读取计算目录")
        self.detected_files_var.set("尚未读取文件")
        self._refresh_action_states()

    def _profile_from_fields(self, *, fingerprint=_UNSET) -> RemoteProfile:
        current = self._selected_profile()
        profile_id = current.profile_id if current else f"profile-{uuid.uuid4().hex[:12]}"
        key = self.key_var.get().strip() or None
        return RemoteProfile(
            profile_id,
            self.name_var.get().strip(),
            self.host_var.get().strip(),
            int(self.port_var.get()),
            self.user_var.get().strip(),
            self.fingerprint_var.get().strip() if fingerprint is _UNSET else fingerprint,
            key,
        )

    def _selected_profile(self) -> RemoteProfile | None:
        name = self.profile_var.get()
        return next((item for item in self.profiles if item.name == name), None)

    def _refresh_profile_choices(self) -> None:
        names = tuple(item.name for item in self.profiles)
        self.profile_box.configure(values=names)
        if not self.profiles:
            self.profile_summary_var.set("尚未保存服务器配置")
            self.editor_expanded_var.set(True)
            self._apply_editor_visibility()
            return
        if self.profile_var.get() not in names:
            self.profile_var.set(names[0])
        self._profile_selected()

    def _profile_selected(self, _event=None) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        self.name_var.set(profile.name)
        self.host_var.set(profile.host)
        self.port_var.set(str(profile.port))
        self.user_var.set(profile.username)
        self.fingerprint_var.set(profile.host_fingerprint or "")
        self.key_var.set(profile.private_key_path or "")
        self.profile_summary_var.set(
            f"{profile.username}@{profile.host}:{profile.port}"
        )
        self.password_var.set("")
        self.key_passphrase_var.set("")
        try:
            self._credential_store(profile.profile_id).load(profile.profile_id)
        except CredentialNotFoundError:
            self.remember_credentials_var.set(False)
            if profile.private_key_path:
                self.credential_status_var.set("使用私钥；未保存私钥口令")
            else:
                self.credential_status_var.set(
                    "未保存 SSH 密码；请在此输入，或选择私钥"
                )
                self.editor_expanded_var.set(True)
                self._apply_editor_visibility()
        except CredentialStoreError:
            self.remember_credentials_var.set(False)
            self.credential_status_var.set("已保存凭据无法读取，请重新输入")
            self.editor_expanded_var.set(True)
            self._apply_editor_visibility()
        else:
            self.remember_credentials_var.set(True)
            self.credential_status_var.set("已由 Windows DPAPI 加密保存")
        self._save_workspace_preferences()

    def _credential_store(self, profile_id: str) -> CredentialStore:
        return CredentialStore(
            credential_store_path(self.credential_root, profile_id)
        )

    def _entered_credentials(self) -> RemoteCredentials | None:
        password = self.password_var.get() or None
        passphrase = self.key_passphrase_var.get() or None
        if password is None and passphrase is None:
            return None
        return RemoteCredentials(password, passphrase)

    def _saved_credentials(self, profile_id: str) -> RemoteCredentials | None:
        try:
            return self._credential_store(profile_id).load(profile_id)
        except CredentialNotFoundError:
            return None

    def _runtime_authentication(self, profile_id: str) -> RuntimeAuthentication:
        return resolve_runtime_authentication(
            self.password_var.get(),
            self.key_passphrase_var.get(),
            self._saved_credentials(profile_id),
        )

    def _choose_key(self) -> None:
        selected = filedialog.askopenfilename(parent=self, title="选择 SSH 私钥")
        if selected:
            self.key_var.set(selected)

    def _transport(self) -> WinScpTransport:
        if self.transport is None:
            installation = WinScpInstallation.find()
            if installation is None:
                raise FileNotFoundError("未找到 WinSCP 及 WinSCPnet.dll。")
            self.transport = WinScpTransport(installation)
        return self.transport

    def _scan_fingerprint(self) -> None:
        try:
            profile = self._profile_from_fields(fingerprint=None)
            profile = RemoteProfile(
                profile.profile_id,
                profile.name,
                profile.host,
                profile.port,
                profile.username,
                None,
                profile.private_key_path,
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("连接配置无效", str(exc), parent=self)
            return
        self.status_var.set("正在向服务器读取 SHA-256 主机指纹…")
        self._async(
            lambda: self._transport().observe_host_fingerprint(profile),
            self._fingerprint_observed,
            retry=self._scan_fingerprint,
        )

    def _fingerprint_observed(self, fingerprint: str) -> None:
        self.fingerprint_var.set(fingerprint)
        self.status_var.set("已读取指纹。请通过服务器管理员或可信渠道核对后再保存。")
        self.task_detail_var.set("主机指纹读取完成；保存前请通过可信渠道核对")

    def _save_profile(self) -> None:
        try:
            profile = self._profile_from_fields()
        except (ValueError, OSError) as exc:
            messagebox.showerror("连接配置无效", str(exc), parent=self)
            return
        by_id = {item.profile_id: item for item in self.profiles}
        by_id[profile.profile_id] = profile
        self.profiles = sorted(by_id.values(), key=lambda item: item.name.casefold())
        self.profile_store.save(self.profiles)
        credential_store = self._credential_store(profile.profile_id)
        try:
            entered = self._entered_credentials()
            if self.remember_credentials_var.get():
                if entered is not None:
                    credential_store.save(profile.profile_id, entered)
                elif self._saved_credentials(profile.profile_id) is None:
                    self.credential_status_var.set("配置已保存；尚未提供登录凭据")
                else:
                    self.credential_status_var.set("已保留 Windows 加密凭据")
            else:
                try:
                    credential_store.delete(profile.profile_id)
                except CredentialNotFoundError:
                    pass
                self.credential_status_var.set("配置已保存；凭据仅用于本次运行")
        except (CredentialStoreError, OSError, ValueError) as exc:
            messagebox.showwarning(
                "凭据未保存",
                f"服务器配置已保存，但 Windows 无法安全保存登录凭据：\n{exc}",
                parent=self,
            )
        self.profile_var.set(profile.name)
        self._refresh_profile_choices()
        self._save_workspace_preferences()
        self.status_var.set(
            f"连接配置已保存：{self.profile_store.path}"
        )

    def _connect(self) -> None:
        try:
            profile = self._profile_from_fields()
            authentication = self._runtime_authentication(profile.profile_id)
        except (CredentialStoreError, ValueError, OSError) as exc:
            messagebox.showerror("连接配置无效", str(exc), parent=self)
            return
        self._clear_inline_error()
        self.status_var.set("正在核验主机指纹并连接…")
        self.task_detail_var.set("正在核验主机指纹并建立安全连接")
        self.connection_var.set("正在连接")
        self._request_generation += 1
        generation = self._request_generation
        self._async(
            lambda: (
                generation,
                connect_pinned(self._transport(), profile, authentication),
            ),
            self._connected,
            retry=self._connect,
        )

    def _connected(self, result) -> None:
        generation, session = result
        if generation != self._request_generation or self._closing:
            session.close()
            return
        if self.session is not None:
            self.session.close()
        self.session = session
        self.remote_runner = RemoteRunner(session, Path(__file__).resolve().parent)
        self.connection_var.set("已连接")
        self.status_var.set("连接已建立")
        self.task_detail_var.set("安全连接已建立，正在读取上次目录")
        self.editor_expanded_var.set(False)
        self._apply_editor_visibility()
        self._refresh_action_states()
        self._list_directory(record=False)

    def _navigate(self, path: str) -> None:
        if self.session is None:
            return
        try:
            normalized = validate_remote_path(path.strip())
        except (AttributeError, ValueError) as exc:
            self._show_inline_error(str(exc), retry=lambda: self._navigate(path))
            return
        self.path_var.set(normalized)
        self._list_directory(record=True)

    def _list_directory(self, *, record: bool = True) -> None:
        if self.session is None:
            return
        try:
            path = validate_remote_path(self.path_var.get().strip())
        except ValueError as exc:
            self._show_inline_error(
                str(exc),
                retry=lambda: self._list_directory(record=record),
            )
            return
        self.path_var.set(path)
        session = self.session
        self._request_generation += 1
        generation = self._request_generation
        self._clear_inline_error()
        self.status_var.set(f"正在读取 {path}")
        self.task_detail_var.set(f"正在读取服务器目录：{path}")
        self._async(
            lambda: self._directory_snapshot(generation, session, path, record),
            self._listed,
            retry=lambda: self._list_directory(record=record),
        )

    def _directory_snapshot(
        self,
        generation: int,
        session,
        path: str,
        record: bool,
    ):
        entries = RemoteBrowser(session).list_directory(path)
        contents: dict[str, str] = {}
        kpoints = next(
            (
                entry
                for entry in entries
                if not entry.is_directory
                and entry.name == "KPOINTS"
                and 0 < entry.size <= 256 * 1024
            ),
            None,
        )
        if kpoints is not None:
            contents["KPOINTS"] = session.read_bytes(
                kpoints.path,
                max_bytes=256 * 1024,
            ).decode("utf-8", errors="replace")
        return generation, session, path, entries, contents, record

    def _listed(self, snapshot) -> None:
        generation, session, path, entries, contents, record = snapshot
        if (
            self._closing
            or generation != self._request_generation
            or session is not self.session
        ):
            return
        if record:
            self.navigation.visit(path)
        self.path_var.set(path)
        self._render_breadcrumb(path)
        self.entries = entries
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        for index, entry in enumerate(entries):
            size = "" if entry.is_directory else self._format_size(entry.size)
            self.file_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    entry.name,
                    "目录" if entry.is_directory else "文件",
                    size,
                    format_modified_time(entry.modified_ns),
                ),
            )
        self.manifest = build_calculation_manifest(
            path,
            entries,
            file_contents=contents,
        )
        for item in self.detected_tree.get_children():
            self.detected_tree.delete(item)
        for index, entry in enumerate(self.manifest.files):
            self.detected_tree.insert(
                "",
                "end",
                iid=f"file-{index}",
                values=(entry.name, self._format_size(entry.size)),
            )
        available_names = available_workflows(self.manifest)
        available_labels = [WORKFLOW_LABELS[name] for name in available_names]
        self.capability_var.set(
            "可用：" + ("、".join(available_labels) or "未识别为可绘图计算目录")
        )
        if self.workflow_var.get() not in available_names and available_names:
            self.workflow_var.set(available_names[0])
        self.detected_files_var.set(f"检测到 {len(self.manifest.files)} 个文件")
        self.status_var.set(f"已读取 {len(entries)} 项")
        self.task_detail_var.set(
            f"目录读取完成：{len(entries)} 项，识别到 {len(available_names)} 种工作流"
        )
        self._refresh_bookmarks()
        self._refresh_action_states()
        self._save_workspace_preferences()

    def _open_selected_directory(self, _event=None) -> None:
        selected = self.file_tree.selection()
        if not selected:
            return
        entry = self.entries[int(selected[0])]
        if entry.is_directory:
            self._navigate(entry.path)
        return "break"

    def _go_back(self, _event=None):
        if self.session is not None and self.navigation.can_back and not self._busy:
            self.path_var.set(self.navigation.back())
            self._list_directory(record=False)
        return "break"

    def _go_forward(self, _event=None):
        if self.session is not None and self.navigation.can_forward and not self._busy:
            self.path_var.set(self.navigation.forward())
            self._list_directory(record=False)
        return "break"

    def _go_parent(self, _event=None):
        try:
            parent = RemoteBrowser.parent(self.path_var.get().strip())
        except ValueError as exc:
            self._show_inline_error(str(exc), retry=self._go_parent)
            return "break"
        self._navigate(parent)
        return "break"

    def _go_home(self, _event=None):
        self._navigate("/home")
        return "break"

    def _sync(self, *, preview: bool = False) -> None:
        if self.session is None or self.manifest is None:
            return
        session = self.session
        manifest = self.manifest
        workflow = self.workflow_var.get()
        try:
            profile = self._profile_from_fields()
            files = selected_sync_files(manifest, workflow)
            destination = cache_directory(
                _local_app_root() / "remote-cache",
                profile.profile_id,
                manifest,
            )
        except (OSError, ValueError) as exc:
            self._show_inline_error(
                str(exc),
                retry=lambda: self._sync(preview=preview),
            )
            return

        self._clear_inline_error()
        self.status_var.set(f"准备同步 {len(files)} 个文件")
        self.task_detail_var.set(
            f"目标：{destination}；工作流：{WORKFLOW_LABELS.get(workflow, workflow)}"
        )

        def perform():
            for index, item in enumerate(files, 1):
                self._queue.put(
                    ("status", format_task_progress(item.name, index, len(files)))
                )
                session.download_file(
                    item.path,
                    destination / item.name,
                    max_bytes=max(item.size, 1),
                )
            return destination, workflow, preview

        self._async(
            perform,
            self._synced,
            retry=lambda: self._sync(preview=preview),
        )

    def _remote_preview(self) -> None:
        if self.session is None or self.manifest is None or self.remote_runner is None:
            return
        workflow = self.workflow_var.get()
        if not workflow_supports_remote_analysis(workflow):
            self._show_inline_error("该工作流尚未开放远端原地解析。")
            return
        runner = self.remote_runner
        manifest = self.manifest
        self._clear_inline_error()
        self.status_var.set("正在检查远端科学环境")
        self.task_detail_var.set("检查 Python、NumPy、Matplotlib、Pillow 与缓存目录")

        def perform():
            report = runner.prepare()
            if not report.compatible:
                raise RuntimeError("远端科学环境不可用：" + " ".join(report.reasons))
            self._queue.put(("status", "远端环境通过，正在提交解析任务"))
            job_id = runner.start(
                workflow,
                manifest.directory,
                manifest.fingerprint,
                {},
            )
            self.remote_job_id = job_id
            self._queue.put(("remote-job-started", job_id))
            while not self._closing:
                event = runner.poll(job_id)
                if event is None:
                    time.sleep(0.25)
                    continue
                if event.status is not None and event.status.progress is not None:
                    progress = event.status.progress
                    self._queue.put(
                        ("status", f"{progress.phase}：{progress.detail}".rstrip("："))
                    )
                if event.state is RemoteTaskState.SUCCEEDED and event.result is not None:
                    return workflow, runner.fetch_preview_bytes(event.result)
                if event.state is RemoteTaskState.FAILED:
                    raise RuntimeError(runner.read_error(job_id))
                if event.state is RemoteTaskState.CANCELLED:
                    raise RuntimeError("远端任务已取消。")
                if event.state is RemoteTaskState.UNKNOWN:
                    raise RuntimeError("连接中断，远端任务状态未知。")
                time.sleep(0.1)
            raise RuntimeError("远程窗口已关闭。")

        self._async(perform, self._remote_preview_ready, retry=self._remote_preview)

    def _remote_preview_ready(self, result) -> None:
        workflow, payload = result
        self.remote_job_id = None
        suffix = "band" if workflow in {"band", "projected_band"} else "dos"
        target = Path(self._remote_preview_directory.name) / f"remote-{suffix}.png"
        target.write_bytes(payload)
        self.status_var.set("远端解析与预览完成")
        self.task_detail_var.set("原始 VASP 文件与解析缓存均保留在 Linux 服务器")
        show = getattr(self.parent_app, "_show_external_preview", None)
        if callable(show):
            show(target, suffix)
        self._refresh_action_states()

    def _cancel_remote_job(self) -> None:
        if self.remote_job_id is None or self.remote_runner is None:
            return
        try:
            requested = self.remote_runner.cancel(self.remote_job_id)
        except Exception as exc:
            self._show_inline_error(str(exc))
            return
        if requested:
            self.status_var.set("已请求取消；等待服务器确认")
            self.task_detail_var.set("取消请求已发送，尚未确认任务终止")

    def _synced(self, result) -> None:
        destination, workflow, preview = result
        self.status_var.set(f"同步完成：{destination}")
        self.task_detail_var.set(f"同步完成：{destination}")
        if not preview:
            return
        app = self.parent_app
        app.read_mode_var.set("lite")
        if workflow in {"band", "projected_band"}:
            app.band_folder_var.set(str(destination))
            app._choose_band_folder(False)
            app.after(600, app._run_band)
        elif workflow == "dos":
            app.dos_folder_var.set(str(destination))
            app._choose_dos_folder(False)
            app.after(600, app._run_dos)
        elif workflow == "fermi_surface":
            app.fermi_folder_var.set(str(destination))
            app._choose_fermi_folder(False)
            app.after(600, app._run_fermi)
        else:
            self.status_var.set(f"同步完成：{destination}；请在对应模块中打开")

    def _async(self, function, on_success, *, retry=None) -> None:
        if self._busy:
            return
        self._retry_action = retry
        self._set_busy(True)

        def run() -> None:
            try:
                result = function()
            except BaseException as exc:
                if not self._closing:
                    self._queue.put(("error", exc, retry))
            else:
                if not self._closing:
                    self._queue.put(("success", on_success, result))
            finally:
                with self._threads_lock:
                    self._threads.discard(threading.current_thread())

        thread = Thread(target=run, name="LyraRemoteOperation", daemon=True)
        with self._threads_lock:
            self._threads.add(thread)
        thread.start()

    def _schedule_poll(self) -> None:
        if self._closing:
            return
        try:
            while True:
                event = self._queue.get_nowait()
                if event[0] == "status":
                    self.status_var.set(event[1])
                    self.task_detail_var.set(event[1])
                elif event[0] == "remote-job-started":
                    self.remote_job_id = event[1]
                    self._refresh_action_states()
                elif event[0] == "error":
                    self.remote_job_id = None
                    self._set_busy(False)
                    if self.session is None:
                        self.connection_var.set("未连接")
                    self._show_inline_error(str(event[1]), retry=event[2])
                elif event[0] == "success":
                    self.remote_job_id = None
                    self._set_busy(False)
                    self._clear_inline_error()
                    event[1](event[2])
        except Empty:
            pass
        self._poll_id = self.after(100, self._schedule_poll)

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{size} B"

    def close(self) -> None:
        if self._closing:
            return
        self._save_workspace_preferences()
        self._closing = True
        self._request_generation += 1
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        if self.session is not None:
            self.session.close()
            self.session = None
        self.remote_runner = None
        self.remote_job_id = None
        if self.transport is not None:
            self.transport.cancel_all()
        with self._threads_lock:
            threads = tuple(self._threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.2)
        preview_directory = getattr(self, "_remote_preview_directory", None)
        if preview_directory is not None:
            preview_directory.cleanup()
        try:
            self.destroy()
        except tk.TclError:
            pass

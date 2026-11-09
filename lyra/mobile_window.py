from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from lyra.mobile.auth import AuthStore
from lyra.mobile.resources import ResourceRegistry
from lyra.mobile.server import MobileService, ServerConfig
from lyra.theme import COLORS, FONTS
from lyra.tool_capabilities import ToolCapabilities


def _default_state_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    return (Path(local) if local else Path.home() / ".lyra") / "Lyra"


class MobileServiceController:
    def __init__(
        self,
        *,
        state_root: Path | None = None,
        port: int = 8765,
        trusted_origins: tuple[str, ...] = (),
        backend=None,
        resources: ResourceRegistry | None = None,
    ) -> None:
        self.state_root = Path(state_root or _default_state_root()).resolve()
        self.auth = AuthStore(state_file=self.state_root / "mobile" / "auth.json")
        self.requested_port = int(port)
        self.trusted_origins = tuple(trusted_origins)
        self.backend = backend
        self.resources = resources or ResourceRegistry()
        self.service: MobileService | None = None
        self.server = None
        self.thread: threading.Thread | None = None
        self._backend_closed = False

    @property
    def running(self) -> bool:
        return bool(self.thread is not None and self.thread.is_alive())

    @property
    def port(self) -> int:
        if self.server is not None:
            return int(self.server.server_address[1])
        return self.requested_port

    def start(self) -> str:
        if self.running:
            return f"http://127.0.0.1:{self.port}"
        self.service = MobileService(
            backend=self.backend,
            auth=self.auth,
            resources=self.resources,
            config=ServerConfig(
                host="127.0.0.1",
                port=self.requested_port,
                trusted_origins=self.trusted_origins,
            ),
        )
        try:
            self.server = self.service.create_server()
        except BaseException:
            self._close_backend()
            raise
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="LyraMobileService",
            daemon=True,
        )
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        server, thread = self.server, self.thread
        self.server = None
        self.thread = None
        self.service = None
        if server is not None:
            server.shutdown()
            server.server_close()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        self._close_backend()

    def _close_backend(self) -> None:
        if self._backend_closed:
            return
        self._backend_closed = True
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()

    def issue_pairing_code(self) -> str:
        return self.auth.issue_pairing_code()


class MobileControlWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        settings_store,
        capabilities: ToolCapabilities,
    ) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        self.capabilities = capabilities
        shared = settings_store.load_namespace("shared")
        saved = shared.get("mobile", {}) if isinstance(shared, dict) else {}
        if not isinstance(saved, dict):
            saved = {}
        self.port_var = tk.StringVar(value=str(saved.get("port", 8765)))
        self.origin_var = tk.StringVar(value=str(saved.get("trusted_origin", "")))
        self.status_var = tk.StringVar(value="手机服务尚未启动")
        self.code_var = tk.StringVar(value="------")
        self.controller: MobileServiceController | None = None
        self._tailscale_enabled = False
        from lyra.mobile.lyra_backend import AuthorizedDirectoryRegistry

        self.directories = AuthorizedDirectoryRegistry(
            state_file=_default_state_root() / "mobile" / "directories.json"
        )
        self._authorize_saved_folders()

        self.title("Lyra - 手机控制中心")
        self.geometry("780x720")
        self.minsize(700, 620)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        shell = ttk.Frame(self, padding=18)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="手机遥控", font=FONTS["section"]).pack(anchor="w")
        ttk.Label(
            shell,
            text="服务只监听本机；远程访问由 Tailscale 私有网络转发。手机不会取得 Linux 密钥。",
            foreground=COLORS["muted"],
            wraplength=700,
        ).pack(anchor="w", pady=(4, 14))

        settings = ttk.LabelFrame(shell, text="本地服务", padding=12)
        settings.pack(fill="x")
        row = ttk.Frame(settings)
        row.pack(fill="x")
        ttk.Label(row, text="端口").pack(side="left")
        ttk.Entry(row, textvariable=self.port_var, width=9).pack(side="left", padx=(8, 18))
        ttk.Label(row, text="可信 HTTPS 地址").pack(side="left")
        ttk.Entry(row, textvariable=self.origin_var).pack(side="left", fill="x", expand=True, padx=(8, 0))
        buttons = ttk.Frame(settings)
        buttons.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(buttons, text="启动服务", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(6, 0))
        self.open_button = ttk.Button(buttons, text="在电脑浏览器打开", command=self._open, state="disabled")
        self.open_button.pack(side="left", padx=(6, 0))
        ttk.Label(settings, textvariable=self.status_var, foreground=COLORS["accent"]).pack(anchor="w", pady=(10, 0))

        pair = ttk.LabelFrame(shell, text="手机配对", padding=12)
        pair.pack(fill="x", pady=(12, 0))
        ttk.Label(pair, textvariable=self.code_var, font=("Segoe UI", 24, "bold")).pack(side="left")
        self.code_button = ttk.Button(pair, text="生成一次性配对码", command=self._issue_code, state="disabled")
        self.code_button.pack(side="left", padx=(18, 0))
        ttk.Label(pair, text="5 分钟有效，使用一次立即失效", foreground=COLORS["muted"]).pack(side="right")

        private = ttk.LabelFrame(shell, text="私有网络", padding=12)
        private.pack(fill="x", pady=(12, 0))
        ttk.Label(private, text=self.capabilities.mobile_summary, wraplength=680).pack(anchor="w")
        self.tailscale_button = ttk.Button(
            private,
            text="启用 Tailscale 私有访问",
            command=self._enable_tailscale,
            state="normal" if self.capabilities.tailscale else "disabled",
        )
        self.tailscale_button.pack(anchor="w", pady=(8, 0))

        folders = ttk.LabelFrame(shell, text="允许手机使用的计算目录", padding=10)
        folders.pack(fill="x", pady=(12, 0))
        self.folder_list = ttk.Treeview(
            folders,
            columns=("label", "id"),
            show="headings",
            height=4,
        )
        self.folder_list.heading("label", text="名称")
        self.folder_list.heading("id", text="手机端编号")
        self.folder_list.column("label", width=350)
        self.folder_list.column("id", width=220)
        self.folder_list.pack(side="left", fill="x", expand=True)
        folder_actions = ttk.Frame(folders)
        folder_actions.pack(side="right", padx=(8, 0), anchor="n")
        ttk.Button(folder_actions, text="添加目录", command=self._add_folder).pack(fill="x")
        ttk.Button(folder_actions, text="撤销所选", command=self._revoke_folder).pack(fill="x", pady=(6, 0))
        self._refresh_folders()

        devices = ttk.LabelFrame(shell, text="已授权设备", padding=10)
        devices.pack(fill="both", expand=True, pady=(12, 0))
        self.device_list = ttk.Treeview(devices, columns=("name", "state"), show="headings", height=6)
        self.device_list.heading("name", text="设备")
        self.device_list.heading("state", text="状态")
        self.device_list.column("name", width=360)
        self.device_list.column("state", width=120, anchor="center")
        self.device_list.pack(side="left", fill="both", expand=True)
        ttk.Button(devices, text="撤销所选设备", command=self._revoke).pack(side="right", anchor="n", padx=(8, 0))
        self._refresh_devices()

    def _start(self) -> None:
        try:
            port = int(self.port_var.get())
            origin = self.origin_var.get().strip().rstrip("/")
            origins = (origin,) if origin else ()
            from lyra.mobile.lyra_backend import LyraTaskBackend

            mobile_root = _default_state_root() / "mobile"
            resources = ResourceRegistry(state_file=mobile_root / "resources.json")
            backend = LyraTaskBackend(
                directories=self.directories,
                resources=resources,
                state_file=mobile_root / "tasks.json",
            )
            self.controller = MobileServiceController(
                port=port,
                trusted_origins=origins,
                backend=backend,
                resources=resources,
            )
            address = self.controller.start()
            self.settings_store.save_namespace(
                "shared",
                {"mobile": {"port": port, "trusted_origin": origin}},
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("手机服务启动失败", str(exc), parent=self)
            return
        self.status_var.set(f"运行中：{address}")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.code_button.configure(state="normal")

    def _stop(self) -> None:
        self._disable_tailscale()
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self.status_var.set("手机服务已停止")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.code_button.configure(state="disabled")
        self.code_var.set("------")

    def _open(self) -> None:
        if self.controller is not None:
            webbrowser.open(f"http://127.0.0.1:{self.controller.port}")

    def _issue_code(self) -> None:
        if self.controller is None:
            return
        self.code_var.set(self.controller.issue_pairing_code())

    def _refresh_devices(self) -> None:
        for item in self.device_list.get_children():
            self.device_list.delete(item)
        auth = self.controller.auth if self.controller else AuthStore(
            state_file=_default_state_root() / "mobile" / "auth.json"
        )
        for device in auth.list_devices():
            self.device_list.insert(
                "",
                "end",
                iid=device.device_id,
                values=(device.name, "已撤销" if device.revoked else "已授权"),
            )

    def _authorize_saved_folders(self) -> None:
        candidates = (
            ("Band", getattr(self.master, "band_folder_var", None)),
            ("DOS", getattr(self.master, "dos_folder_var", None)),
            ("Combined Band", getattr(self.master, "combined_band_folder_var", None)),
            ("Combined DOS", getattr(self.master, "combined_dos_folder_var", None)),
            ("Fermi", getattr(self.master, "fermi_folder_var", None)),
        )
        known: set[Path] = set()
        for label, variable in candidates:
            if variable is None:
                continue
            path = Path(variable.get()).expanduser()
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in known:
                continue
            known.add(resolved)
            self.directories.authorize(resolved, label=f"{label}: {resolved.name}")

    def _refresh_folders(self) -> None:
        for item in self.folder_list.get_children():
            self.folder_list.delete(item)
        for directory in self.directories.list_public():
            self.folder_list.insert(
                "",
                "end",
                iid=directory.directory_id,
                values=(directory.label, directory.directory_id),
            )

    def _add_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="选择允许手机使用的计算目录")
        if not selected:
            return
        path = Path(selected).resolve()
        self.directories.authorize(path, label=path.name or str(path))
        self._refresh_folders()

    def _revoke_folder(self) -> None:
        selected = self.folder_list.selection()
        if selected:
            self.directories.revoke(selected[0])
            self._refresh_folders()

    def _revoke(self) -> None:
        selected = self.device_list.selection()
        if not selected:
            return
        auth = self.controller.auth if self.controller else AuthStore(
            state_file=_default_state_root() / "mobile" / "auth.json"
        )
        auth.revoke(selected[0])
        self._refresh_devices()

    def _enable_tailscale(self) -> None:
        if self.capabilities.tailscale is None or self.controller is None:
            messagebox.showinfo("Tailscale", "请先启动手机服务。", parent=self)
            return
        command = [
            str(self.capabilities.tailscale),
            "serve",
            "--bg",
            f"http://127.0.0.1:{self.controller.port}",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            messagebox.showerror("Tailscale 启动失败", str(exc), parent=self)
            return
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode:
            messagebox.showerror("Tailscale 启动失败", output or "命令执行失败", parent=self)
        else:
            match = re.search(r"https://[^\s/]+\.ts\.net", output)
            if match and self.controller.service is not None:
                origin = match.group(0).rstrip("/")
                self.controller.service.trust_origin(origin)
                self.origin_var.set(origin)
                self.settings_store.save_namespace(
                    "shared",
                    {"mobile": {"trusted_origin": origin}},
                )
                self.status_var.set(f"手机访问地址：{origin}")
                self._tailscale_enabled = True
            else:
                self.status_var.set(output or "Tailscale 私有访问已启用")
                self._tailscale_enabled = True

    def _disable_tailscale(self) -> None:
        if not self._tailscale_enabled or self.capabilities.tailscale is None:
            return
        try:
            subprocess.run(
                [str(self.capabilities.tailscale), "serve", "off"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._tailscale_enabled = False

    def close(self) -> None:
        self._stop()
        try:
            self.destroy()
        except tk.TclError:
            pass

from __future__ import annotations

from pathlib import Path
import tkinter as tk

import lyra.remote_window as remote_window_module
from lyra.incar_window import IncarAssistantWindow
from lyra.mobile_window import MobileControlWindow
from lyra.remote.profiles import ProfileStore, RemoteProfile
from lyra.remote_window import RemoteWorkspaceWindow
from lyra.settings_store import SettingsStore
from lyra.tool_capabilities import ToolCapabilities


def test_new_feature_windows_construct_and_close_without_background_leaks(
    tmp_path: Path,
) -> None:
    root = tk.Tk()
    root.withdraw()
    store = SettingsStore(tmp_path / "settings.json")
    capabilities = ToolCapabilities(winscp=None, openssh=Path("ssh.exe"), tailscale=None)
    windows = []
    try:
        windows = [
            IncarAssistantWindow(root, settings_store=store),
            MobileControlWindow(
                root,
                settings_store=store,
                capabilities=capabilities,
            ),
            RemoteWorkspaceWindow(
                root,
                settings_store=store,
                capabilities=capabilities,
            ),
        ]
        root.update_idletasks()

        assert all(window.winfo_exists() for window in windows)
        assert windows[0].title() == "Lyra - INCAR 助手"
        assert windows[1].title() == "Lyra - 手机控制中心"
        assert windows[2].title() == "Lyra - 远程 Linux 工作台"
        assert hasattr(windows[2], "password_var")
        assert hasattr(windows[2], "key_passphrase_var")
        assert hasattr(windows[2], "remember_credentials_var")
    finally:
        for window in reversed(windows):
            window.close()
        root.destroy()


def test_remote_profile_rail_scrolls_and_exposes_missing_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_store = ProfileStore(tmp_path / "remote_profiles.json")
    profile_store.save(
        (
            RemoteProfile(
                "cluster-a",
                "Linux 集群",
                "192.0.2.10",
                22,
                "researcher",
                "ssh-ed25519 255 SHA256:abcdefghijklmnopqrstuvwx1234567890ABCDE",
            ),
        )
    )
    monkeypatch.setattr(remote_window_module, "ProfileStore", lambda: profile_store)
    monkeypatch.setattr(
        remote_window_module,
        "_local_app_root",
        lambda: tmp_path / "local-app",
    )
    root = tk.Tk()
    root.withdraw()
    window = RemoteWorkspaceWindow(
        root,
        settings_store=SettingsStore(tmp_path / "settings.json"),
        capabilities=ToolCapabilities(
            winscp=Path("WinSCP.exe"),
            openssh=Path("ssh.exe"),
            tailscale=None,
        ),
    )
    try:
        root.update_idletasks()

        assert window.server_scrollbar.winfo_manager() == "grid"
        assert tuple(window.profile_box.cget("values")) == ("Linux 集群",)
        assert window.editor_expanded_var.get() is True
        assert "SSH 密码" in window.credential_status_var.get()
    finally:
        window.close()
        root.destroy()

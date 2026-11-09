from __future__ import annotations

import ctypes
from ctypes import wintypes
import platform


_DWMWA_SYSTEMBACKDROP_TYPE = 38
_DWMSBT_MAINWINDOW = 2


def apply_system_backdrop(window: object) -> bool:
    """Request the Windows 11 main-window backdrop when the platform supports it."""
    try:
        if platform.system() != "Windows":
            return False

        top_level = window.winfo_toplevel()
        top_level.update_idletasks()
        hwnd = top_level.winfo_id()
        if not hwnd:
            return False

        dwmapi = ctypes.WinDLL("dwmapi")
        set_window_attribute = dwmapi.DwmSetWindowAttribute
        set_window_attribute.argtypes = (
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        set_window_attribute.restype = ctypes.c_long

        backdrop = ctypes.c_int(_DWMSBT_MAINWINDOW)
        result = set_window_attribute(
            wintypes.HWND(hwnd),
            _DWMWA_SYSTEMBACKDROP_TYPE,
            ctypes.byref(backdrop),
            ctypes.sizeof(backdrop),
        )
        return result == 0
    except Exception:
        return False

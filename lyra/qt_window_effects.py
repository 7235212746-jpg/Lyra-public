from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys


class _Margins(ctypes.Structure):
    _fields_ = (
        ("left", ctypes.c_int),
        ("right", ctypes.c_int),
        ("top", ctypes.c_int),
        ("bottom", ctypes.c_int),
    )


class _AccentPolicy(ctypes.Structure):
    _fields_ = (
        ("state", ctypes.c_int),
        ("flags", ctypes.c_int),
        ("gradient_color", ctypes.c_uint32),
        ("animation_id", ctypes.c_int),
    )


class _CompositionAttributeData(ctypes.Structure):
    _fields_ = (
        ("attribute", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
    )


def apply_acrylic(window) -> tuple[int, int, int] | None:
    if sys.platform != "win32":
        return None
    hwnd = wintypes.HWND(int(window.winId()))
    dwm = ctypes.WinDLL("dwmapi")
    set_attribute = dwm.DwmSetWindowAttribute
    set_attribute.argtypes = (
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    set_attribute.restype = ctypes.c_long
    backdrop = ctypes.c_int(3)
    backdrop_result = int(
        set_attribute(hwnd, 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
    )

    extend = dwm.DwmExtendFrameIntoClientArea
    extend.argtypes = (wintypes.HWND, ctypes.POINTER(_Margins))
    extend.restype = ctypes.c_long
    margins = _Margins(-1, -1, -1, -1)
    extend_result = int(extend(hwnd, ctypes.byref(margins)))

    accent = _AccentPolicy(4, 2, 0x50F4F1EE, 0)
    composition = _CompositionAttributeData(
        19,
        ctypes.cast(ctypes.byref(accent), ctypes.c_void_p),
        ctypes.sizeof(accent),
    )
    set_composition = ctypes.WinDLL("user32").SetWindowCompositionAttribute
    set_composition.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(_CompositionAttributeData),
    )
    set_composition.restype = wintypes.BOOL
    acrylic_result = int(set_composition(hwnd, ctypes.byref(composition)))
    return backdrop_result, extend_result, acrylic_result

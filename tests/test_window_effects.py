from __future__ import annotations

import importlib
import unittest
from unittest.mock import MagicMock, patch


class _TopLevel:
    def __init__(self) -> None:
        self.update_calls = 0
        self.id_calls = 0

    def update_idletasks(self) -> None:
        self.update_calls += 1

    def winfo_id(self) -> int:
        self.id_calls += 1
        return 101


class _ChildWidget:
    def __init__(self, top_level: _TopLevel | None = None) -> None:
        self.top_level = top_level or _TopLevel()
        self.toplevel_calls = 0
        self.update_calls = 0
        self.id_calls = 0

    def winfo_toplevel(self) -> _TopLevel:
        self.toplevel_calls += 1
        return self.top_level

    def update_idletasks(self) -> None:
        self.update_calls += 1

    def winfo_id(self) -> int:
        self.id_calls += 1
        return 202


class _BrokenTopLevelWidget(_ChildWidget):
    def winfo_toplevel(self) -> _TopLevel:
        raise RuntimeError("top-level unavailable")


def _window_effects():
    return importlib.import_module("lyra.window_effects")


class SystemBackdropTests(unittest.TestCase):
    def test_non_windows_returns_false(self) -> None:
        try:
            window_effects = _window_effects()
        except ModuleNotFoundError as error:
            self.fail(f"Native backdrop adapter is missing: {error}")

        with patch.object(window_effects.platform, "system", return_value="Linux"):
            self.assertFalse(window_effects.apply_system_backdrop(_ChildWidget()))

    def test_platform_detection_failure_is_contained(self) -> None:
        window_effects = _window_effects()
        with patch.object(
            window_effects.platform,
            "system",
            side_effect=OSError("platform unavailable"),
        ):
            try:
                result = window_effects.apply_system_backdrop(_ChildWidget())
            except OSError as error:
                self.fail(f"Platform failure escaped: {error}")
        self.assertFalse(result)

    def test_dwm_load_failure_is_contained(self) -> None:
        try:
            window_effects = _window_effects()
        except ModuleNotFoundError as error:
            self.fail(f"Native backdrop adapter is missing: {error}")

        with (
            patch.object(window_effects.platform, "system", return_value="Windows"),
            patch.object(window_effects.ctypes, "WinDLL", side_effect=OSError("DWM unavailable")),
        ):
            self.assertFalse(window_effects.apply_system_backdrop(_ChildWidget()))

    def test_child_widget_uses_top_level_hwnd(self) -> None:
        try:
            window_effects = _window_effects()
        except ModuleNotFoundError as error:
            self.fail(f"Native backdrop adapter is missing: {error}")

        top_level = _TopLevel()
        child = _ChildWidget(top_level)
        dwmapi = MagicMock()
        dwmapi.DwmSetWindowAttribute.return_value = 0
        with (
            patch.object(window_effects.platform, "system", return_value="Windows"),
            patch.object(window_effects.ctypes, "WinDLL", return_value=dwmapi),
        ):
            self.assertTrue(window_effects.apply_system_backdrop(child))

        self.assertEqual(child.toplevel_calls, 1)
        self.assertEqual(child.update_calls, 0)
        self.assertEqual(child.id_calls, 0)
        self.assertEqual(top_level.update_calls, 1)
        self.assertEqual(top_level.id_calls, 1)
        hwnd = dwmapi.DwmSetWindowAttribute.call_args.args[0]
        self.assertEqual(hwnd.value, 101)

    def test_windows_declares_dwm_signature_and_documented_backdrop_values(self) -> None:
        window_effects = _window_effects()
        dwmapi = MagicMock()
        dwmapi.DwmSetWindowAttribute.return_value = 0
        with (
            patch.object(window_effects.platform, "system", return_value="Windows"),
            patch.object(window_effects.ctypes, "WinDLL", return_value=dwmapi),
        ):
            self.assertTrue(window_effects.apply_system_backdrop(_ChildWidget()))

        arguments = dwmapi.DwmSetWindowAttribute.call_args.args
        self.assertEqual(
            dwmapi.DwmSetWindowAttribute.argtypes,
            (
                window_effects.wintypes.HWND,
                window_effects.wintypes.DWORD,
                window_effects.ctypes.c_void_p,
                window_effects.wintypes.DWORD,
            ),
        )
        self.assertIs(dwmapi.DwmSetWindowAttribute.restype, window_effects.ctypes.c_long)
        self.assertEqual(arguments[1], 38)
        self.assertEqual(arguments[3], window_effects.ctypes.sizeof(window_effects.ctypes.c_int))
        self.assertEqual(
            window_effects.ctypes.cast(
                arguments[2], window_effects.ctypes.POINTER(window_effects.ctypes.c_int)
            ).contents.value,
            2,
        )

    def test_nonzero_hresult_returns_false(self) -> None:
        window_effects = _window_effects()
        dwmapi = MagicMock()
        dwmapi.DwmSetWindowAttribute.return_value = -2147024809
        with (
            patch.object(window_effects.platform, "system", return_value="Windows"),
            patch.object(window_effects.ctypes, "WinDLL", return_value=dwmapi),
        ):
            self.assertFalse(window_effects.apply_system_backdrop(_ChildWidget()))

    def test_top_level_resolution_failure_is_contained(self) -> None:
        window_effects = _window_effects()
        dwmapi = MagicMock()
        dwmapi.DwmSetWindowAttribute.return_value = 0
        with (
            patch.object(window_effects.platform, "system", return_value="Windows"),
            patch.object(window_effects.ctypes, "WinDLL", return_value=dwmapi),
        ):
            self.assertFalse(window_effects.apply_system_backdrop(_BrokenTopLevelWidget()))

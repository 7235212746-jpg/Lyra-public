from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk

import pytest


_GUI_TEST_FILES = {
    "test_app.py",
    "test_band_inspector_window.py",
    "test_feature_windows_smoke.py",
    "test_gui_test_harness.py",
    "test_preview_viewer.py",
}


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("LYRA_RUN_GUI_TESTS") == "1":
        return
    skip = pytest.mark.skip(
        reason=(
            "real Tk windows are disabled in normal background tests; "
            "set LYRA_RUN_GUI_TESTS=1 only on an isolated test desktop"
        )
    )
    for item in items:
        if Path(str(item.fspath)).name in _GUI_TEST_FILES:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def keep_tk_test_windows_invisible(monkeypatch):
    """Preserve Tk mapping semantics without flashing test windows on desktop."""

    for window_type in (tk.Tk, tk.Toplevel):
        original_init = window_type.__init__

        def invisible_init(self, *args, _original=original_init, **kwargs):
            _original(self, *args, **kwargs)
            try:
                self.attributes("-alpha", 0.0)
            except tk.TclError:
                self.withdraw()

        monkeypatch.setattr(window_type, "__init__", invisible_init)

from __future__ import annotations

import tkinter as tk


def test_tk_test_windows_are_invisible_while_remaining_mappable() -> None:
    root = tk.Tk()
    child = tk.Toplevel(root)
    try:
        root.update_idletasks()

        assert float(root.attributes("-alpha")) == 0.0
        assert float(child.attributes("-alpha")) == 0.0
        assert root.state() == "normal"
        assert child.state() == "normal"
    finally:
        child.destroy()
        root.destroy()

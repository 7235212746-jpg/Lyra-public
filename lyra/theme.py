from __future__ import annotations


FONT_STACK = (
    "Segoe UI",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Arial",
    "sans-serif",
)

COLORS = {
    "background": "#edf3f4",
    "shell": "#e3eef0",
    "panel": "#ffffff",
    "panel_alt": "#f6fafb",
    "border": "#c9d7da",
    "border_soft": "#dce6e8",
    "text": "#16252a",
    "muted": "#566b72",
    "subtle": "#52666d",
    "accent": "#006b6d",
    "accent_hover": "#00595b",
    "accent_soft": "#d8f0ef",
    "warning": "#b74a43",
    "caution": "#946000",
    "success": "#187044",
    "danger": "#b42318",
    "input": "#ffffff",
    "plot_background": "#ffffff",
    "preview_background": "#ffffff",
    "export_background": "#ffffff",
}

FONTS = {
    "body": (FONT_STACK[1], 10),
    "small": (FONT_STACK[1], 9),
    "title": (FONT_STACK[0], 20, "bold"),
    "section": (FONT_STACK[1], 12, "bold"),
    "button": (FONT_STACK[1], 10, "bold"),
}

SPACING = {
    "window": 18,
    "panel": 16,
    "row": 10,
    "small": 6,
}

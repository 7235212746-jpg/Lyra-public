from __future__ import annotations

from lyra.models import ExportSettings
from lyra.theme import COLORS


def publication_rc(export: ExportSettings) -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [export.font_family, "Microsoft YaHei", "Segoe UI"],
        "font.size": export.font_size,
        "figure.facecolor": COLORS["plot_background"],
        "axes.facecolor": COLORS["plot_background"],
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.facecolor": COLORS["export_background"],
        "savefig.transparent": False,
    }

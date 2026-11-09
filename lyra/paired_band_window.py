from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
import math
import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lyra.band_inspector_window import nearest_band_state, state_detail_rows
from lyra.models import BandElementIndex, BandStateDetail, BandStateSelection
from lyra.paired_band import PairedBandSession, rank_projection_candidates
from lyra.theme import COLORS, FONTS


DetailQuery = Callable[[str, BandStateSelection], BandStateDetail | None]


class PairedBandInspector(tk.Toplevel):
    """Compare two cached calculations without asserting wavefunction identity."""

    def __init__(
        self,
        parent: tk.Misc,
        session: PairedBandSession,
        *,
        query_detail: DetailQuery,
        energy_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.query_detail = query_detail
        self.energy_range = energy_range
        self.title("Lyra - 能带对照与逐点成分")
        self.geometry("1500x900")
        self.minsize(1120, 700)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self.left_spin_var = tk.StringVar(value=session.left.spin_names[0])
        self.right_spin_var = tk.StringVar(value=session.right.spin_names[0])
        self.scientific_status_var = tk.StringVar()
        self.left_detail_status_var = tk.StringVar(value="单击左图中的能带点")
        self.right_detail_status_var = tk.StringVar(value="单击右图或选择候选")
        self.left_selection: BandStateSelection | None = None
        self.right_selection: BandStateSelection | None = None
        self._candidates = ()
        self._candidate_source_side: str | None = None
        self._markers: dict[str, object] = {}
        self._detail_generation = {"left": 0, "right": 0}
        self._detail_queue: Queue = Queue()
        self._poll_id = None
        self._build_layout()
        self._draw_all()
        self._set_compatibility_status()
        self._poll_id = self.after(80, self._poll_details)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_layout(self) -> None:
        controls = ttk.Frame(self, padding=(12, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="左侧自旋", font=FONTS["body"]).pack(side="left")
        left_spin = ttk.Combobox(
            controls,
            textvariable=self.left_spin_var,
            values=self.session.left.spin_names,
            state="readonly",
            width=10,
        )
        left_spin.pack(side="left", padx=(6, 24))
        ttk.Label(controls, text="右侧自旋", font=FONTS["body"]).pack(side="left")
        right_spin = ttk.Combobox(
            controls,
            textvariable=self.right_spin_var,
            values=self.session.right.spin_names,
            state="readonly",
            width=10,
        )
        right_spin.pack(side="left", padx=(6, 24))
        left_spin.bind("<<ComboboxSelected>>", lambda _e: self._spin_changed("left"))
        right_spin.bind("<<ComboboxSelected>>", lambda _e: self._spin_changed("right"))
        ttk.Label(
            controls,
            text="每侧以自己的 E-fermi 为零；候选不是波函数身份判定",
            foreground=COLORS["muted"],
        ).pack(side="right")

        plot_holder = ttk.Frame(self, padding=(12, 0, 12, 4))
        plot_holder.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(13.8, 5.2), dpi=100, layout="constrained")
        self.left_axes, self.right_axes = self.figure.subplots(1, 2, sharey=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_holder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.navigation = NavigationToolbar2Tk(self.canvas, plot_holder, pack_toolbar=False)
        self.navigation.update()
        self.navigation.pack(fill="x")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        ttk.Label(
            self,
            textvariable=self.scientific_status_var,
            padding=(14, 4),
            anchor="w",
            foreground=COLORS["muted"],
        ).pack(fill="x")

        details = ttk.Panedwindow(self, orient="horizontal")
        details.pack(fill="both", expand=False, padx=12, pady=(0, 10))
        left_panel = ttk.Frame(details, padding=6)
        candidate_panel = ttk.Frame(details, padding=6)
        right_panel = ttk.Frame(details, padding=6)
        details.add(left_panel, weight=2)
        details.add(candidate_panel, weight=2)
        details.add(right_panel, weight=2)
        self.left_detail = self._detail_panel(
            left_panel, "左侧所选点组成", self.left_detail_status_var, self.session.left.elements
        )
        self.right_detail = self._detail_panel(
            right_panel, "右侧所选点组成", self.right_detail_status_var, self.session.right.elements
        )
        ttk.Label(candidate_panel, text="投影相似候选", font=FONTS["section"]).pack(anchor="w")
        ttk.Label(
            candidate_panel,
            text="综合元素投影余弦相似度与能量接近程度，仅作候选排序。",
            foreground=COLORS["muted"],
            wraplength=430,
        ).pack(fill="x", pady=(2, 5))
        self.candidate_tree = ttk.Treeview(
            candidate_panel,
            columns=("band", "k", "energy", "delta", "similarity", "score"),
            show="headings",
            height=8,
            selectmode="browse",
        )
        for key, text, width in (
            ("band", "Band", 58), ("k", "k 点", 54), ("energy", "E (eV)", 72),
            ("delta", "ΔE", 64), ("similarity", "成分相似", 76), ("score", "综合分", 64),
        ):
            self.candidate_tree.heading(key, text=text)
            self.candidate_tree.column(key, width=width, anchor="center")
        self.candidate_tree.pack(fill="both", expand=True)
        self.candidate_tree.bind("<<TreeviewSelect>>", self._candidate_selected)

    @staticmethod
    def _detail_panel(parent, title, status_var, elements):
        ttk.Label(parent, text=title, font=FONTS["section"]).pack(anchor="w")
        ttk.Label(parent, textvariable=status_var, foreground=COLORS["muted"]).pack(fill="x", pady=(2, 5))
        tree = ttk.Treeview(parent, columns=elements, show="tree headings", height=8)
        tree.heading("#0", text="分量")
        tree.column("#0", width=130, anchor="w")
        for element in elements:
            tree.heading(element, text=element)
            tree.column(element, width=68, anchor="center")
        tree.pack(fill="both", expand=True)
        return tree

    def _draw_index(self, axes, index: BandElementIndex, spin: str, side: str) -> None:
        axes.clear()
        energies = np.asarray(index.energies[spin], dtype=float)
        for band_index in range(energies.shape[1]):
            for start, stop in index.segment_ranges:
                axes.plot(
                    index.distances[start:stop], energies[start:stop, band_index],
                    color="#252a34", linewidth=0.9,
                )
        for tick in index.tick_positions:
            axes.axvline(tick, color="#c8ccd3", linewidth=0.65, zorder=0)
        axes.axhline(0.0, color="#cf5147", linewidth=0.8, linestyle="--")
        axes.set_xticks(index.tick_positions, index.tick_labels)
        axes.set_xlim(float(index.distances[0]), float(index.distances[-1]))
        if self.energy_range is not None:
            axes.set_ylim(*self.energy_range)
        axes.set_xlabel("k path")
        folder = self.session.left_folder if side == "left" else self.session.right_folder
        axes.set_title(f"{folder.name} | E-fermi = {index.reference_energy:.6f} eV", fontsize=11)
        axes.tick_params(direction="in")

    def _draw_all(self) -> None:
        self._markers.clear()
        self._draw_index(self.left_axes, self.session.left, self.left_spin_var.get(), "left")
        self._draw_index(self.right_axes, self.session.right, self.right_spin_var.get(), "right")
        self.left_axes.set_ylabel(r"$E - E_\mathrm{F}$ (eV)")
        self.canvas.draw_idle()

    def _set_compatibility_status(self) -> None:
        compatibility = self.session.compatibility
        if compatibility.sync_enabled:
            self.scientific_status_var.set("路径逐点兼容：点击后同步 k 点；另一侧显示投影相似候选。")
        else:
            reason = "；".join(compatibility.reasons) or "路径证据不足"
            self.scientific_status_var.set(f"近似候选模式：{reason}。不声明两侧为同一物理能带。")

    def _spin_changed(self, side: str) -> None:
        if side == "left":
            self.left_selection = None
        else:
            self.right_selection = None
        self._draw_all()

    def _event_selection(self, event, side: str) -> BandStateSelection | None:
        index = self.session.left if side == "left" else self.session.right
        axes = self.left_axes if side == "left" else self.right_axes
        spin = self.left_spin_var.get() if side == "left" else self.right_spin_var.get()
        if event.inaxes is not axes or event.xdata is None or event.ydata is None:
            return None
        selection = nearest_band_state(index, float(event.xdata), float(event.ydata), spin)
        x = index.distances[selection.kpoint_index]
        y = index.energies[spin][selection.kpoint_index, selection.band_index]
        pixel = axes.transData.transform((x, y))
        return selection if math.hypot(pixel[0] - event.x, pixel[1] - event.y) <= 12 else None

    def _on_plot_click(self, event) -> None:
        side = "left" if event.inaxes is self.left_axes else "right"
        selection = self._event_selection(event, side)
        if selection is not None:
            self.select_state(side, selection)

    def select_state(self, side: str, selection: BandStateSelection) -> None:
        if side not in {"left", "right"}:
            raise ValueError("Side must be 'left' or 'right'.")
        if side == "left":
            self.left_selection = selection
            candidates = rank_projection_candidates(
                self.session.left, self.session.right, selection, self.right_spin_var.get()
            )
        else:
            self.right_selection = selection
            candidates = rank_projection_candidates(
                self.session.right, self.session.left, selection, self.left_spin_var.get()
            )
        self._candidates = candidates
        self._candidate_source_side = side
        self._fill_candidates(candidates)
        if candidates:
            opposite = "right" if side == "left" else "left"
            if opposite == "right":
                self.right_selection = candidates[0].selection
            else:
                self.left_selection = candidates[0].selection
        self._draw_markers()
        self._request_detail(side, selection)
        if candidates:
            self._request_detail("right" if side == "left" else "left", candidates[0].selection)
        self.scientific_status_var.set(
            "已固定离散 k 点并列出投影相似候选；候选不是 WAVECAR 波函数身份判定。"
        )

    def _fill_candidates(self, candidates) -> None:
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        for position, candidate in enumerate(candidates):
            similarity = candidate.composition_similarity
            self.candidate_tree.insert(
                "", "end", iid=str(position),
                values=(
                    candidate.band_index + 1, candidate.kpoint_index + 1,
                    f"{candidate.energy:.4f}", f"{candidate.energy_delta:+.4f}",
                    "—" if not np.isfinite(similarity) else f"{similarity * 100:.1f}%",
                    f"{candidate.score:.3f}",
                ),
            )
        if candidates:
            self.candidate_tree.selection_set("0")

    def _candidate_selected(self, _event=None) -> None:
        selected = self.candidate_tree.selection()
        if not selected:
            return
        candidate = self._candidates[int(selected[0])]
        if self._candidate_source_side == "left":
            self.right_selection = candidate.selection
            side = "right"
        else:
            self.left_selection = candidate.selection
            side = "left"
        self._draw_markers()
        self._request_detail(side, candidate.selection)

    def _draw_markers(self) -> None:
        for marker in self._markers.values():
            try:
                marker.remove()
            except ValueError:
                pass
        self._markers.clear()
        for side, selection in (("left", self.left_selection), ("right", self.right_selection)):
            if selection is None:
                continue
            index = self.session.left if side == "left" else self.session.right
            axes = self.left_axes if side == "left" else self.right_axes
            energy = index.energies[selection.spin][selection.kpoint_index, selection.band_index]
            marker, = axes.plot(
                [index.distances[selection.kpoint_index]], [energy], marker="o",
                markersize=8, markerfacecolor="#ffffff", markeredgecolor="#d3483f",
                markeredgewidth=2, zorder=10,
            )
            self._markers[side] = marker
        self.canvas.draw_idle()

    def _request_detail(self, side: str, selection: BandStateSelection) -> None:
        available = (
            self.session.left_detail_available
            if side == "left"
            else self.session.right_detail_available
        )
        if not available:
            status = self.left_detail_status_var if side == "left" else self.right_detail_status_var
            status.set("此侧无投影数据；仍可查看能量与 k 点候选")
            return
        self._detail_generation[side] += 1
        generation = self._detail_generation[side]

        def query() -> None:
            try:
                detail = self.query_detail(side, selection)
                self._detail_queue.put((side, generation, selection, detail, None))
            except Exception as exc:
                self._detail_queue.put((side, generation, selection, None, f"{type(exc).__name__}: {exc}"))

        Thread(target=query, name=f"lyra-paired-{side}-detail", daemon=True).start()

    def _poll_details(self) -> None:
        while True:
            try:
                side, generation, selection, detail, error = self._detail_queue.get_nowait()
            except Empty:
                break
            if generation != self._detail_generation[side]:
                continue
            status = self.left_detail_status_var if side == "left" else self.right_detail_status_var
            if error:
                status.set(error)
            elif detail is None:
                status.set(f"k {selection.kpoint_index + 1} | Band {selection.band_index + 1}")
            else:
                self.set_detail(side, detail)
        if self.winfo_exists():
            self._poll_id = self.after(80, self._poll_details)

    def set_detail(self, side: str, detail: BandStateDetail) -> None:
        tree = self.left_detail if side == "left" else self.right_detail
        status = self.left_detail_status_var if side == "left" else self.right_detail_status_var
        tree.delete(*tree.get_children())
        for label, values in state_detail_rows(detail):
            tree.insert("", "end", text=label, values=values)
        selection = detail.selection
        status.set(
            f"k {selection.kpoint_index + 1} | Band {selection.band_index + 1} | "
            f"投影总量 {detail.raw_projection_total:.4f}"
        )

    def close(self) -> None:
        for side in self._detail_generation:
            self._detail_generation[side] += 1
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        idle_draw_id = getattr(self.canvas, "_idle_draw_id", None)
        if idle_draw_id is not None:
            try:
                self.after_cancel(idle_draw_id)
            except tk.TclError:
                pass
            self.canvas._idle_draw_id = None
        try:
            self.canvas.get_tk_widget().destroy()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass

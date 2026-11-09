from __future__ import annotations

from dataclasses import replace
import math
import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lyra.band_inspector import near_degenerate_bands
from lyra.models import (
    BandElementIndex,
    BandPlotSettings,
    BandStateDetail,
    BandStateSelection,
    ParsedBandData,
)
from lyra.plotting.band import draw_band_axes
from lyra.theme import COLORS, FONTS


DetailRequest = Callable[[BandStateSelection, bool, tuple[int, ...]], None]
SPIN_COMPONENT_NAMES = ("Mx", "My", "Mz")


def nearest_band_state(
    index: BandElementIndex,
    distance: float,
    energy: float,
    spin: str,
) -> BandStateSelection:
    energies = np.asarray(index.energies[spin], dtype=float)
    distance_errors = np.abs(index.distances - distance)
    nearest_distance = float(np.nanmin(distance_errors))
    kpoint_candidates = np.flatnonzero(
        np.isclose(distance_errors, nearest_distance, atol=1e-12, rtol=0.0)
    )
    candidate_errors = np.abs(energies[kpoint_candidates] - energy)
    candidate_index, band_index = np.unravel_index(
        int(np.nanargmin(candidate_errors)),
        candidate_errors.shape,
    )
    kpoint_index = int(kpoint_candidates[candidate_index])
    return BandStateSelection(spin, kpoint_index, band_index)


def _format_percentage(value: float, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def element_percentage_rows(
    index: BandElementIndex,
    spin: str,
    kpoint_index: int,
    energy_range: tuple[float, float] | None = None,
) -> list[tuple[int, float, tuple[str, ...]]]:
    spin_index = index.spin_names.index(spin)
    energies = np.asarray(index.energies[spin][kpoint_index], dtype=float)
    percentages = np.asarray(
        index.relative_element_percentages[spin_index, :, kpoint_index, :],
        dtype=float,
    )
    rows = []
    for band_index, energy in enumerate(energies):
        if energy_range is not None and not energy_range[0] <= energy <= energy_range[1]:
            continue
        rows.append(
            (
                band_index + 1,
                float(energy),
                tuple(
                    _format_percentage(percentages[element_index, band_index])
                    for element_index in range(len(index.elements))
                ),
            )
        )
    return rows


def state_detail_rows(
    detail: BandStateDetail,
) -> list[tuple[str, tuple[str, ...]]]:
    relative = np.asarray(detail.relative_orbital_percentages, dtype=float)
    element_totals = np.where(
        np.any(np.isfinite(relative), axis=1),
        np.nansum(relative, axis=1),
        np.nan,
    )
    rows = [
        (
            "元素合计",
            tuple(
                _format_percentage(value)
                for value in element_totals
            ),
        )
    ]
    rows.extend(
        (
            orbital,
            tuple(
                _format_percentage(relative[element_index, orbital_index])
                for element_index in range(len(detail.elements))
            ),
        )
        for orbital_index, orbital in enumerate(detail.orbitals)
    )
    if detail.spin_components is not None and detail.raw_projection_total > 1e-8:
        components = (
            np.sum(np.asarray(detail.spin_components, dtype=float), axis=2)
            * (100.0 / detail.raw_projection_total)
        )
        rows.extend(
            (
                f"{component_name} / 投影总量",
                tuple(
                    _format_percentage(value, signed=True)
                    for value in components[component_index]
                ),
            )
            for component_index, component_name in enumerate(SPIN_COMPONENT_NAMES)
        )
    return rows


class BandProjectionInspector(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        index: BandElementIndex,
        request_detail: DetailRequest,
        energy_range: tuple[float, float] | None = None,
        display_data: ParsedBandData | None = None,
        plot_settings: BandPlotSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.request_detail = request_detail
        self.energy_range = energy_range
        self.display_data = display_data
        self.plot_settings = plot_settings
        self.title("Lyra - 放大能带与成分分析")
        self.geometry("1320x820")
        self.minsize(960, 640)
        self.configure(bg=COLORS["background"])
        self.transient(parent)

        self.spin_var = tk.StringVar(value=index.spin_names[0])
        self.whole_band_var = tk.BooleanVar(value=False)
        self.degenerate_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(
            value="悬停仅预览元素占比；单击曲线或表格才分析该点。"
        )
        self.detail_status_var = tk.StringVar(value="尚未选择能带状态")
        self._selection: BandStateSelection | None = None
        self._hover_selection: BandStateSelection | None = None
        self._line_artists = []
        self._selection_artists = []

        self._build_layout()
        self._draw_bands()
        self._set_kpoint_overview(0)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self, padding=(12, 9))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="自旋", font=FONTS["body"]).pack(side="left")
        spin_box = ttk.Combobox(
            toolbar,
            textvariable=self.spin_var,
            values=self.index.spin_names,
            state="readonly",
            width=10,
        )
        spin_box.pack(side="left", padx=(6, 18))
        spin_box.bind("<<ComboboxSelected>>", self._spin_changed)
        ttk.Checkbutton(
            toolbar,
            text="整条能带统计",
            variable=self.whole_band_var,
            command=self._request_current_detail,
        ).pack(side="left")
        ttk.Checkbutton(
            toolbar,
            text="合并 5 meV 内近简并带",
            variable=self.degenerate_var,
            command=self._request_current_detail,
        ).pack(side="left", padx=(14, 0))
        ttk.Label(
            toolbar,
            text="百分比口径：已解析投影内占比",
            foreground=COLORS["muted"],
        ).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        plot_panel = ttk.Frame(body, padding=8)
        table_panel = ttk.Frame(body, padding=(8, 4))
        body.add(plot_panel, weight=3)
        body.add(table_panel, weight=2)

        self.figure = Figure(figsize=(7.2, 6.2), dpi=100, layout="constrained")
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.navigation = NavigationToolbar2Tk(
            self.canvas,
            plot_panel,
            pack_toolbar=False,
        )
        self.navigation.update()
        self.navigation.pack(fill="x", pady=(4, 0))
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        ttk.Label(
            plot_panel,
            textvariable=self.status_var,
            anchor="w",
            font=FONTS["small"],
        ).pack(fill="x", pady=(5, 0))

        table_panel.columnconfigure(0, weight=1)
        table_panel.rowconfigure(1, weight=3)
        table_panel.rowconfigure(4, weight=2)
        ttk.Label(
            table_panel,
            text="当前 k 点：各能带元素占比 (%)",
            font=FONTS["section"],
        ).grid(row=0, column=0, sticky="w")
        overview_holder, self.overview = self._make_tree(
            table_panel,
            "Band",
            self.index.elements,
        )
        overview_holder.grid(row=1, column=0, sticky="nsew", pady=(6, 12))
        self.overview.bind("<<TreeviewSelect>>", self._overview_selected)

        ttk.Label(
            table_panel,
            text="所选状态：元素、轨道与自旋分量",
            font=FONTS["section"],
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(
            table_panel,
            textvariable=self.detail_status_var,
            foreground=COLORS["muted"],
            font=FONTS["small"],
            wraplength=440,
        ).grid(row=3, column=0, sticky="ew", pady=(3, 5))
        detail_holder, self.detail = self._make_tree(
            table_panel,
            "分量",
            self.index.elements,
        )
        detail_holder.grid(row=4, column=0, sticky="nsew")

    @staticmethod
    def _make_tree(
        parent: tk.Misc,
        first_heading: str,
        columns: tuple[str, ...],
    ) -> tuple[ttk.Frame, ttk.Treeview]:
        holder = ttk.Frame(parent)
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        tree = ttk.Treeview(
            holder,
            columns=columns,
            show="tree headings",
            selectmode="browse",
            height=12,
        )
        vertical = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.heading("#0", text=first_heading)
        tree.column("#0", width=116, minwidth=86, stretch=False, anchor="w")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=74, minwidth=58, anchor="center")
        return holder, tree

    def _draw_bands(self) -> None:
        self.axes.clear()
        self._line_artists.clear()
        self._selection_artists.clear()
        spin = self.spin_var.get()
        energies = np.asarray(self.index.energies[spin], dtype=float)
        if self.display_data is not None and self.plot_settings is not None:
            display_data = replace(
                self.display_data,
                energies={spin: self.display_data.energies[spin]},
                projection_total=(
                    {spin: self.display_data.projection_total[spin]}
                    if spin in self.display_data.projection_total
                    else {}
                ),
            )
            draw_band_axes(
                self.axes,
                display_data,
                self.plot_settings,
                self.index.reference_energy,
            )
        else:
            for band_index in range(energies.shape[1]):
                for start, stop in self.index.segment_ranges:
                    (line,) = self.axes.plot(
                        self.index.distances[start:stop],
                        energies[start:stop, band_index],
                        color="#252a34",
                        linewidth=0.85,
                    )
                    self._line_artists.append(line)
            for tick in self.index.tick_positions:
                self.axes.axvline(tick, color="#c8ccd3", linewidth=0.65, zorder=0)
            self.axes.axhline(0.0, color="#cf5147", linewidth=0.8, linestyle="--")
            self.axes.set_xticks(self.index.tick_positions, self.index.tick_labels)
            self.axes.set_ylabel(r"$E - E_\mathrm{ref}$ (eV)")
            self.axes.set_xlim(
                float(self.index.distances[0]),
                float(self.index.distances[-1]),
            )
            if self.energy_range is not None:
                self.axes.set_ylim(*self.energy_range)
            self.axes.tick_params(direction="in")
        self.canvas.draw_idle()

    def _spin_changed(self, _event=None) -> None:
        self._selection = None
        self._draw_bands()
        self._set_kpoint_overview(0)
        self._clear_detail()

    def _candidate_from_event(self, event) -> BandStateSelection | None:
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return None
        selection = nearest_band_state(
            self.index,
            float(event.xdata),
            float(event.ydata),
            self.spin_var.get(),
        )
        energy = self.index.energies[selection.spin][
            selection.kpoint_index, selection.band_index
        ]
        x = self.index.distances[selection.kpoint_index]
        pixel_x, pixel_y = self.axes.transData.transform((x, energy))
        if math.hypot(pixel_x - event.x, pixel_y - event.y) > 12.0:
            return None
        return selection

    def _on_motion(self, event) -> None:
        selection = self._candidate_from_event(event)
        if selection is None or selection == self._hover_selection:
            return
        self._hover_selection = selection
        self._set_hover_status(selection)

    def _set_hover_status(self, selection: BandStateSelection) -> None:
        spin_index = self.index.spin_names.index(selection.spin)
        percentages = self.index.relative_element_percentages[
            spin_index, :, selection.kpoint_index, selection.band_index
        ]
        contributions = "  ".join(
            f"{element} {_format_percentage(value)}%"
            for element, value in zip(self.index.elements, percentages)
        )
        energy = self.index.energies[selection.spin][
            selection.kpoint_index, selection.band_index
        ]
        self.status_var.set(
            f"k 点 {selection.kpoint_index + 1} | Band {selection.band_index + 1} | "
            f"{energy:.4f} eV | {contributions}"
        )

    def _on_plot_click(self, event) -> None:
        selection = self._candidate_from_event(event)
        if selection is not None:
            self.select_state(selection)

    def _set_kpoint_overview(
        self,
        kpoint_index: int,
        selected_band_index: int | None = None,
    ) -> None:
        self.overview.delete(*self.overview.get_children())
        y_min, y_max = self.axes.get_ylim()
        for band_number, energy, values in element_percentage_rows(
            self.index,
            self.spin_var.get(),
            kpoint_index,
            (y_min, y_max),
        ):
            band_index = band_number - 1
            iid = f"{kpoint_index}:{band_index}"
            self.overview.insert(
                "",
                "end",
                iid=iid,
                text=f"{band_number}  {energy:.3f}",
                values=values,
            )
            if band_index == selected_band_index:
                self.overview.selection_set(iid)
                self.overview.see(iid)

    def _overview_selected(self, _event=None) -> None:
        selected = self.overview.selection()
        if not selected:
            return
        kpoint_index, band_index = (
            int(value) for value in selected[0].split(":", maxsplit=1)
        )
        selection = BandStateSelection(
            self.spin_var.get(),
            kpoint_index,
            band_index,
        )
        if selection != self._selection:
            self.select_state(selection)

    def select_state(self, selection: BandStateSelection) -> None:
        self._selection = selection
        self._highlight_selection(selection)
        self._set_kpoint_overview(selection.kpoint_index, selection.band_index)
        self._request_current_detail()

    def _highlight_selection(self, selection: BandStateSelection) -> None:
        for artist in self._selection_artists:
            artist.remove()
        self._selection_artists.clear()
        energies = np.asarray(self.index.energies[selection.spin], dtype=float)
        y = energies[:, selection.band_index]
        for start, stop in self.index.segment_ranges:
            (line,) = self.axes.plot(
                self.index.distances[start:stop],
                y[start:stop],
                color="#f2b134",
                linewidth=2.4,
                alpha=0.9,
                zorder=20,
            )
            self._selection_artists.append(line)
        marker = self.axes.scatter(
            [self.index.distances[selection.kpoint_index]],
            [y[selection.kpoint_index]],
            s=58,
            facecolor="#ffffff",
            edgecolor="#d33f49",
            linewidth=1.8,
            zorder=21,
        )
        self._selection_artists.append(marker)
        self.canvas.draw_idle()

    def _request_current_detail(self) -> None:
        if self._selection is None:
            return
        members = (
            near_degenerate_bands(self.index, self._selection)
            if self.degenerate_var.get()
            else (self._selection.band_index,)
        )
        self.detail_status_var.set("正在读取所选状态的轨道明细…")
        self.request_detail(
            self._selection,
            bool(self.whole_band_var.get()),
            members,
        )

    def set_detail(self, detail: BandStateDetail) -> None:
        self.detail.delete(*self.detail.get_children())
        for row_index, (label, values) in enumerate(state_detail_rows(detail)):
            self.detail.insert("", "end", iid=f"detail:{row_index}", text=label, values=values)
        scope = "整条能带路径加权平均" if detail.whole_band else "当前 k 点"
        members = ", ".join(str(index + 1) for index in detail.band_indices)
        self.detail_status_var.set(
            f"{scope} | Band {members} | 原始投影总量 {detail.raw_projection_total:.6g} | "
            "轨道百分比按已解析投影总量归一化"
        )

    def _clear_detail(self) -> None:
        self.detail.delete(*self.detail.get_children())
        self.detail_status_var.set("尚未选择能带状态")

    def close(self) -> None:
        self.destroy()

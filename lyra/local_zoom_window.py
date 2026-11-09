from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from lyra.models import (
    BandPlotSettings,
    CombinedPlotSettings,
    DosPlotSettings,
    ExportSettings,
    ParsedBandData,
    ParsedDosData,
)
from lyra.plotting.band import draw_band_axes
from lyra.plotting.dos import draw_dos_axes
from lyra.plotting.style import publication_rc
from lyra.theme import COLORS, FONTS
from lyra.band_gap_measurement import (
    format_gap_measurement,
    measure_selected_gap,
    nearest_band_point,
)
from lyra.band_edge_analysis import (
    BandEdgeAnalysis,
    analyze_band_edges,
    format_band_edge_summary,
)


def validate_font_size(value) -> float:
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("图中文字必须是数字。") from exc
    if not math.isfinite(size) or not 8.0 <= size <= 48.0:
        raise ValueError("图中文字应在 8 至 48 pt 之间。")
    return size


def validate_line_width(value) -> float:
    try:
        width = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("能带线宽必须是数字。") from exc
    if not math.isfinite(width) or not 0.3 <= width <= 6.0:
        raise ValueError("能带线宽应在 0.3 至 6.0 pt 之间。")
    return width


def local_zoom_rc(
    export: ExportSettings,
    font_size: float = 22.0,
) -> dict[str, object]:
    style = publication_rc(export)
    base_size = validate_font_size(font_size)
    style.update(
        {
            "font.size": base_size,
            "axes.labelsize": base_size + 2.0,
            "axes.titlesize": base_size + 2.0,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
        }
    )
    return style


def validate_axis_limits(values) -> tuple[float, float, float, float]:
    try:
        x_min, x_max, y_min, y_max = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("横纵坐标范围必须是数字。") from exc
    if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        raise ValueError("横纵坐标范围必须是有限数。")
    if x_min >= x_max:
        raise ValueError("横坐标最小值必须小于最大值。")
    if y_min >= y_max:
        raise ValueError("纵坐标最小值必须小于最大值。")
    return x_min, x_max, y_min, y_max


class _BandGapMeasurementMixin:
    def _init_gap_measurement(self, data: ParsedBandData, reference: float) -> None:
        self._gap_data = data
        self._gap_reference = float(reference)
        self._gap_points = []
        self._gap_measurement = None
        self._gap_artists = []
        self._gap_mode = False
        self._edge_analysis = None
        self._edge_artists = []
        self._edge_report_window = None

    def _attach_gap_measurement(self, axes) -> None:
        self._gap_axes = axes
        self._gap_artists = []
        self._edge_artists = []
        self._draw_gap_measurement()
        self._draw_edge_analysis()

    def _run_edge_analysis(self, *, show_report: bool = True) -> BandEdgeAnalysis:
        result = analyze_band_edges(
            self._gap_data,
            reference_energy=self._gap_reference,
        )
        self._edge_analysis = result
        self._draw_edge_analysis()
        self.canvas.draw_idle()
        self.status_var.set(format_band_edge_summary(result))
        if show_report:
            if self._edge_report_window is not None:
                try:
                    self._edge_report_window.destroy()
                except tk.TclError:
                    pass
            self._edge_report_window = BandEdgeReportWindow(self, result)
        return result

    def _draw_edge_analysis(self) -> None:
        for artist in self._edge_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._edge_artists = []
        result = self._edge_analysis
        if result is None or result.vbm is None or result.cbm is None:
            return
        vbm_x = float(self._gap_data.distances[result.vbm.kpoint_index])
        cbm_x = float(self._gap_data.distances[result.cbm.kpoint_index])
        vbm_y = float(result.vbm.energy - result.reference_energy)
        cbm_y = float(result.cbm.energy - result.reference_energy)
        for x, y, colour, marker, label in (
            (vbm_x, vbm_y, "#2563eb", "v", "VBM"),
            (cbm_x, cbm_y, "#dc2626", "^", "CBM"),
        ):
            self._edge_artists.append(
                self._gap_axes.scatter(
                    [x],
                    [y],
                    s=150,
                    marker=marker,
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=33,
                    label=label,
                )
            )
        line, = self._gap_axes.plot(
            [vbm_x, cbm_x],
            [vbm_y, cbm_y],
            color="#374151",
            linestyle=":",
            linewidth=max(1.5, self._plot_line_width),
            zorder=32,
        )
        self._edge_artists.append(line)

    def _clear_edge_analysis(self) -> None:
        self._edge_analysis = None
        self._draw_edge_analysis()
        self.canvas.draw_idle()
        self.status_var.set("已清除自动带边标记。")

    def _close_edge_report(self) -> None:
        report = self._edge_report_window
        self._edge_report_window = None
        if report is None:
            return
        try:
            report.destroy()
        except tk.TclError:
            pass

    def _toggle_gap_measurement(self) -> None:
        self._gap_mode = not self._gap_mode
        self.measure_gap_button.configure(
            text="测量中：点击两点" if self._gap_mode else "测量能隙"
        )
        self.status_var.set(
            "请依次点击两个带边；将自动吸附到最近的真实能带点。"
            if self._gap_mode
            else "能隙测量已暂停；已有标记仍会保留在保存图片中。"
        )

    def _on_gap_click(self, event) -> None:
        if not self._gap_mode or event.inaxes is not self._gap_axes or event.button != 1:
            return
        if getattr(self.navigation, "mode", ""):
            self.status_var.set("请先退出工具栏的缩放或平移模式，再点击带边。")
            return
        point = nearest_band_point(
            self._gap_data,
            self._gap_reference,
            click_display=(event.x, event.y),
            transform=self._gap_axes.transData.transform,
        )
        if point is None:
            self.status_var.set("附近没有可吸附的真实能带点，请更靠近曲线点击。")
            return
        if len(self._gap_points) >= 2:
            self._gap_points = []
            self._gap_measurement = None
        self._gap_points.append(point)
        if len(self._gap_points) == 2:
            self._gap_measurement = measure_selected_gap(
                self._gap_points[0], self._gap_points[1], self._gap_data.kpoints
            )
            self.status_var.set(format_gap_measurement(self._gap_measurement))
        else:
            self.status_var.set(
                f"已选第一个点：{point.spin} / Band {point.band_index + 1} / "
                f"k {point.kpoint_index + 1} / {point.energy:.6f} eV；请点击第二点。"
            )
        self._draw_gap_measurement()
        self.canvas.draw_idle()

    def _draw_gap_measurement(self) -> None:
        for artist in self._gap_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._gap_artists = []
        if not self._gap_points or not hasattr(self, "_gap_axes"):
            return
        points = self._gap_points
        colours = (
            tuple(
                "#2563eb" if point == self._gap_measurement.lower else "#dc2626"
                for point in points
            )
            if self._gap_measurement is not None
            else ("#f59e0b",)
        )
        for point, colour in zip(points, colours):
            artist = self._gap_axes.scatter(
                [point.distance],
                [point.energy],
                s=95,
                marker="o",
                facecolor=colour,
                edgecolor="white",
                linewidth=1.4,
                zorder=30,
            )
            self._gap_artists.append(artist)
        if self._gap_measurement is None:
            return
        lower = self._gap_measurement.lower
        upper = self._gap_measurement.upper
        line, = self._gap_axes.plot(
            [lower.distance, upper.distance],
            [lower.energy, upper.energy],
            color="#111827",
            linewidth=max(1.5, self._plot_line_width),
            linestyle="--",
            zorder=29,
        )
        self._gap_artists.append(line)
        annotation = self._gap_axes.annotate(
            f"ΔE = {self._gap_measurement.gap:.4f} eV\n"
            f"{'direct' if self._gap_measurement.is_direct else 'indirect'}",
            xy=((lower.distance + upper.distance) / 2.0, (lower.energy + upper.energy) / 2.0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=self._plot_font_size,
            color="#111827",
            bbox={"boxstyle": "square,pad=0.2", "facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.92},
            zorder=31,
        )
        self._gap_artists.append(annotation)

    def _clear_gap_measurement(self, _event=None) -> None:
        self._gap_points = []
        self._gap_measurement = None
        self._draw_gap_measurement()
        self.canvas.draw_idle()
        self.status_var.set("已清除所选两点和能量差标记。")


class BandEdgeReportWindow(tk.Toplevel):
    _STATUS_LABELS = {
        "gapped": "有带隙",
        "metallic": "金属",
        "zero_gap": "零带隙",
        "ambiguous": "展宽下不确定",
        "unavailable": "无法判定",
    }

    def __init__(self, parent: tk.Misc, result: BandEdgeAnalysis) -> None:
        super().__init__(parent)
        self.title("Lyra - 自动带边证据")
        self.geometry("860x430")
        self.minsize(700, 360)
        self.transient(parent)
        self.configure(bg=COLORS["background"])

        ttk.Label(
            self,
            text=format_band_edge_summary(result),
            font=FONTS["heading"],
            wraplength=810,
            justify="left",
            padding=(16, 14, 16, 10),
        ).pack(fill="x")

        columns = ("spin", "status", "gap", "vbm", "cbm")
        table = ttk.Treeview(self, columns=columns, show="headings", height=5)
        for key, label, width in (
            ("spin", "自旋通道", 100),
            ("status", "判定", 100),
            ("gap", "通道带隙 (eV)", 130),
            ("vbm", "VBM: Band / k / E-Eref", 210),
            ("cbm", "CBM: Band / k / E-Eref", 210),
        ):
            table.heading(key, text=label)
            table.column(key, width=width, anchor="center")
        for channel in result.channels:
            vbm = self._edge_text(channel.vbm, result.reference_energy)
            cbm = self._edge_text(channel.cbm, result.reference_energy)
            table.insert(
                "",
                "end",
                values=(
                    channel.spin,
                    self._STATUS_LABELS.get(channel.status, channel.status),
                    "-" if channel.gap is None else f"{channel.gap:.6f}",
                    vbm,
                    cbm,
                ),
            )
        table.pack(fill="x", padx=16)

        evidence = tk.Text(self, height=7, wrap="word", relief="flat", padx=10, pady=8)
        evidence.insert(
            "1.0",
            "判据：直接读取 VASP 占据数；未取得可靠占据数时不会按 E-fermi 猜测。"
            "结果只覆盖文件中实际采样的 k 点。\n\n"
            + "\n".join(
                f"{channel.spin}: {channel.evidence}" for channel in result.channels
            )
            + (
                "\n" + "\n".join(f"注意：{warning}" for warning in result.warnings)
                if result.warnings
                else ""
            ),
        )
        evidence.configure(state="disabled")
        evidence.pack(fill="both", expand=True, padx=16, pady=(10, 8))
        ttk.Button(self, text="关闭", command=self.destroy).pack(anchor="e", padx=16, pady=(0, 14))

    @staticmethod
    def _edge_text(edge, reference: float) -> str:
        if edge is None:
            return "-"
        return (
            f"{edge.band_index + 1} / {edge.kpoint_index + 1} / "
            f"{edge.energy - reference:+.6f}"
        )


class BandLocalZoomWindow(_BandGapMeasurementMixin, tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        data: ParsedBandData,
        settings: BandPlotSettings,
        output_directory: str | Path,
    ) -> None:
        super().__init__(parent)
        self.data = data
        self.settings = settings
        self.output_directory = Path(output_directory)
        self.title("Lyra - 能带局部放大图")
        self.geometry("1220x820")
        self.minsize(900, 620)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self.status_var = tk.StringVar(
            value="使用工具栏框选/平移，或输入坐标范围；保存的是当前真实坐标视图。"
        )
        self.x_min_var = tk.StringVar()
        self.x_max_var = tk.StringVar()
        self.y_min_var = tk.StringVar()
        self.y_max_var = tk.StringVar()
        self.font_size_var = tk.StringVar(value="22")
        self._plot_font_size = 22.0
        self._plot_line_width = max(1.8, float(settings.export.line_width))
        self.line_width_var = tk.StringVar(value=f"{self._plot_line_width:g}")
        self._closed = False
        self._build_layout()
        reference = data.efermi if settings.reference_energy is None else settings.reference_energy
        self._init_gap_measurement(data, reference)
        self._attach_gap_measurement(self.axes)
        self._gap_click_id = self.canvas.mpl_connect("button_press_event", self._on_gap_click)
        self._draw()
        self._original_limits = (*self.axes.get_xlim(), *self.axes.get_ylim())
        self._sync_limit_entries()
        self.axes.callbacks.connect("xlim_changed", lambda _axes: self._sync_limit_entries())
        self.axes.callbacks.connect("ylim_changed", lambda _axes: self._sync_limit_entries())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", self._clear_gap_measurement)

    def _build_layout(self) -> None:
        controls = ttk.Frame(self, padding=(12, 9))
        controls.pack(fill="x")
        self._range_controls(controls, "X", self.x_min_var, self.x_max_var).pack(
            side="left"
        )
        self._range_controls(controls, "Y", self.y_min_var, self.y_max_var).pack(
            side="left", padx=(18, 0)
        )
        ttk.Button(controls, text="应用范围", command=self.apply_limits).pack(
            side="left", padx=(18, 4)
        )
        ttk.Button(controls, text="复位", command=self.reset_limits).pack(
            side="left", padx=4
        )
        font_controls = ttk.Frame(controls)
        font_controls.pack(side="left", padx=(14, 0))
        ttk.Label(font_controls, text="图中文字").pack(side="left")
        ttk.Spinbox(
            font_controls,
            from_=8,
            to=48,
            increment=1,
            width=5,
            textvariable=self.font_size_var,
        ).pack(side="left", padx=(6, 4))
        ttk.Label(font_controls, text="pt").pack(side="left")
        ttk.Label(font_controls, text="线宽").pack(side="left", padx=(12, 0))
        ttk.Spinbox(
            font_controls,
            from_=0.3,
            to=6.0,
            increment=0.1,
            width=5,
            textvariable=self.line_width_var,
        ).pack(side="left", padx=(6, 4))
        ttk.Label(font_controls, text="pt").pack(side="left")
        ttk.Button(
            font_controls,
            text="应用样式",
            command=self.apply_plot_style,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls,
            text="保存当前视图",
            style="Primary.TButton",
            command=self.save_current_view,
        ).pack(side="right")
        self.clear_gap_button = ttk.Button(
            controls, text="清除测量", command=self._clear_gap_measurement
        )
        self.clear_gap_button.pack(side="right", padx=(4, 0))
        self.measure_gap_button = ttk.Button(
            controls, text="测量能隙", command=self._toggle_gap_measurement
        )
        self.measure_gap_button.pack(side="right", padx=(8, 4))
        ttk.Button(
            controls,
            text="自动找带边",
            command=self._run_edge_analysis,
        ).pack(side="right", padx=(8, 4))

        plot_panel = ttk.Frame(self, padding=(12, 0, 12, 6))
        plot_panel.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(9.0, 6.2), dpi=100, layout="constrained")
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.navigation = NavigationToolbar2Tk(self.canvas, plot_panel, pack_toolbar=False)
        self.navigation.update()
        self.navigation.pack(fill="x", pady=(4, 0))
        ttk.Label(
            self,
            textvariable=self.status_var,
            foreground=COLORS["muted"],
            font=FONTS["small"],
            padding=(14, 4, 14, 9),
            anchor="w",
        ).pack(fill="x")

    @staticmethod
    def _range_controls(parent, label, minimum_var, maximum_var):
        holder = ttk.Frame(parent)
        ttk.Label(holder, text=f"{label} 范围").pack(side="left")
        ttk.Entry(holder, textvariable=minimum_var, width=9).pack(side="left", padx=(6, 3))
        ttk.Label(holder, text="至").pack(side="left")
        ttk.Entry(holder, textvariable=maximum_var, width=9).pack(side="left", padx=(3, 0))
        return holder

    def _draw(self) -> None:
        style = local_zoom_rc(self.settings.export, self._plot_font_size)
        plot_settings = replace(
            self.settings,
            export=replace(
                self.settings.export,
                line_width=self._plot_line_width,
            ),
        )
        with matplotlib.rc_context(style):
            self.axes.clear()
            draw_band_axes(
                self.axes,
                self.data,
                plot_settings,
                plot_settings.reference_energy,
            )
        self._apply_plot_typography(style)
        self._attach_gap_measurement(self.axes)
        self.canvas.draw_idle()

    def _apply_plot_typography(self, style: dict[str, object]) -> None:
        self.axes.xaxis.label.set_fontsize(style["axes.labelsize"])
        self.axes.yaxis.label.set_fontsize(style["axes.labelsize"])
        self.axes.tick_params(
            axis="both",
            which="major",
            labelsize=style["xtick.labelsize"],
        )
        legend = self.axes.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(style["legend.fontsize"])

    def apply_plot_style(self) -> None:
        try:
            font_size = validate_font_size(self.font_size_var.get())
            line_width = validate_line_width(self.line_width_var.get())
        except ValueError as exc:
            messagebox.showerror("样式错误", str(exc), parent=self)
            return
        current_limits = (*self.axes.get_xlim(), *self.axes.get_ylim())
        self._plot_font_size = font_size
        self._plot_line_width = line_width
        self._draw()
        self.axes.set_xlim(current_limits[0], current_limits[1])
        self.axes.set_ylim(current_limits[2], current_limits[3])
        self.canvas.draw_idle()
        self.status_var.set(
            f"样式已更新：文字 {font_size:g} pt；线宽 {line_width:g} pt。"
        )

    @staticmethod
    def _format_limit(value: float) -> str:
        return f"{value:.8g}"

    def _sync_limit_entries(self) -> None:
        if self._closed:
            return
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        self.x_min_var.set(self._format_limit(x_min))
        self.x_max_var.set(self._format_limit(x_max))
        self.y_min_var.set(self._format_limit(y_min))
        self.y_max_var.set(self._format_limit(y_max))

    def apply_limits(self) -> None:
        try:
            x_min, x_max, y_min, y_max = validate_axis_limits(
                (
                    self.x_min_var.get(),
                    self.x_max_var.get(),
                    self.y_min_var.get(),
                    self.y_max_var.get(),
                )
            )
        except ValueError as exc:
            messagebox.showerror("范围错误", str(exc), parent=self)
            return
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)
        self.canvas.draw_idle()
        self.status_var.set(
            f"当前视图：X {x_min:.5g} 至 {x_max:.5g}；Y {y_min:.5g} 至 {y_max:.5g} eV"
        )

    def reset_limits(self) -> None:
        x_min, x_max, y_min, y_max = self._original_limits
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)
        self.canvas.draw_idle()
        self.status_var.set("已恢复完整能带范围。")

    def save_current_view(self) -> Path | None:
        initial = self.output_directory
        initial.mkdir(parents=True, exist_ok=True)
        selected = filedialog.asksaveasfilename(
            parent=self,
            initialdir=str(initial),
            initialfile=f"{self.settings.export.basename}_local.png",
            defaultextension=".png",
            filetypes=(
                ("PNG image", "*.png"),
                ("PDF vector", "*.pdf"),
                ("SVG vector", "*.svg"),
            ),
        )
        if not selected:
            return None
        path = Path(selected)
        if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            path = path.with_suffix(".png")
        try:
            self.figure.savefig(
                path,
                dpi=self.settings.export.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return None
        self.status_var.set(f"已保存当前局部视图：{path}")
        return path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_edge_report()
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


class CombinedLocalZoomWindow(_BandGapMeasurementMixin, tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        band_data: ParsedBandData,
        dos_data: ParsedDosData,
        settings: CombinedPlotSettings,
        output_directory: str | Path,
    ) -> None:
        super().__init__(parent)
        self.band_data = band_data
        self.dos_data = dos_data
        self.settings = settings
        self.output_directory = Path(output_directory)
        self.title("Lyra - 能带 + DOS 局部放大图")
        self.geometry("1320x840")
        self.minsize(980, 640)
        self.configure(bg=COLORS["background"])
        self.transient(parent)
        self.status_var = tk.StringVar(value="工具栏可框选和平移整张双面板图；输入范围后保存的仍是真实坐标图。")
        self.energy_min_var = tk.StringVar()
        self.energy_max_var = tk.StringVar()
        self.dos_min_var = tk.StringVar()
        self.dos_max_var = tk.StringVar()
        self.font_size_var = tk.StringVar(value="22")
        self.line_width_var = tk.StringVar(value=f"{max(1.8, float(settings.export.line_width)):g}")
        self._plot_font_size = 22.0
        self._plot_line_width = max(1.8, float(settings.export.line_width))
        self._closed = False
        self._build_layout()
        reference = dos_data.efermi if settings.reference_energy is None else settings.reference_energy
        self._init_gap_measurement(band_data, reference)
        self._draw()
        self._gap_click_id = self.canvas.mpl_connect("button_press_event", self._on_gap_click)
        self._original_energy = self.band_axes.get_ylim()
        self._original_dos = self.dos_axes.get_xlim()
        self._sync_limit_entries()
        self.band_axes.callbacks.connect("ylim_changed", lambda _axes: self._sync_limit_entries())
        self.dos_axes.callbacks.connect("xlim_changed", lambda _axes: self._sync_limit_entries())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", self._clear_gap_measurement)

    def _build_layout(self) -> None:
        controls = ttk.Frame(self, padding=(12, 9))
        controls.pack(fill="x")
        BandLocalZoomWindow._range_controls(
            controls, "能量 Y", self.energy_min_var, self.energy_max_var
        ).pack(side="left")
        BandLocalZoomWindow._range_controls(
            controls, "DOS X", self.dos_min_var, self.dos_max_var
        ).pack(side="left", padx=(18, 0))
        ttk.Button(controls, text="应用范围", command=self.apply_limits).pack(side="left", padx=(16, 4))
        ttk.Button(controls, text="复位", command=self.reset_limits).pack(side="left", padx=4)
        ttk.Label(controls, text="字号").pack(side="left", padx=(16, 0))
        ttk.Spinbox(controls, from_=8, to=48, width=5, textvariable=self.font_size_var).pack(side="left", padx=(5, 2))
        ttk.Label(controls, text="线宽").pack(side="left", padx=(10, 0))
        ttk.Spinbox(controls, from_=0.3, to=6.0, increment=0.1, width=5, textvariable=self.line_width_var).pack(side="left", padx=(5, 2))
        ttk.Button(controls, text="应用样式", command=self.apply_plot_style).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="保存当前视图", style="Primary.TButton", command=self.save_current_view).pack(side="right")
        self.clear_gap_button = ttk.Button(controls, text="清除测量", command=self._clear_gap_measurement)
        self.clear_gap_button.pack(side="right", padx=(4, 0))
        self.measure_gap_button = ttk.Button(controls, text="测量能隙", command=self._toggle_gap_measurement)
        self.measure_gap_button.pack(side="right", padx=(8, 4))
        ttk.Button(
            controls,
            text="自动找带边",
            command=self._run_edge_analysis,
        ).pack(side="right", padx=(8, 4))

        panel = ttk.Frame(self, padding=(12, 0, 12, 6))
        panel.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(10.2, 6.4), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.navigation = NavigationToolbar2Tk(self.canvas, panel, pack_toolbar=False)
        self.navigation.update()
        self.navigation.pack(fill="x", pady=(4, 0))
        ttk.Label(self, textvariable=self.status_var, foreground=COLORS["muted"], font=FONTS["small"], padding=(14, 4, 14, 9), anchor="w").pack(fill="x")

    def _draw(self) -> None:
        self.figure.clear()
        self.band_axes, self.dos_axes = self.figure.subplots(
            1,
            2,
            sharey=True,
            gridspec_kw={"width_ratios": (self.settings.panel_ratio, 1.0), "wspace": 0.06},
        )
        export = replace(self.settings.export, line_width=self._plot_line_width)
        settings = replace(self.settings, export=export)
        reference = self.dos_data.efermi if settings.reference_energy is None else settings.reference_energy
        band_settings = BandPlotSettings(
            settings.energy_range,
            reference,
            settings.band_mode,
            settings.band_projection_channels,
            settings.band_complete_colour,
            settings.band_fat_scale,
            export,
        )
        dos_settings = DosPlotSettings(
            settings.energy_range,
            reference,
            settings.dos_projection_channels,
            settings.plot_total_dos,
            export,
            settings.dos_density_range,
        )
        style = local_zoom_rc(export, self._plot_font_size)
        with matplotlib.rc_context(style):
            draw_band_axes(self.band_axes, self.band_data, band_settings, reference)
            draw_dos_axes(self.dos_axes, self.dos_data, dos_settings, reference, energy_axis="y", legend_outside=True)
            self.dos_axes.tick_params(axis="y", labelleft=False)
        for axes in (self.band_axes, self.dos_axes):
            axes.xaxis.label.set_fontsize(style["axes.labelsize"])
            axes.yaxis.label.set_fontsize(style["axes.labelsize"])
            axes.tick_params(axis="both", labelsize=style["xtick.labelsize"])
            legend = axes.get_legend()
            if legend is not None:
                for label in legend.get_texts():
                    label.set_fontsize(style["legend.fontsize"])
        self._attach_gap_measurement(self.band_axes)
        self.canvas.draw_idle()

    @staticmethod
    def _pair(minimum, maximum, label: str) -> tuple[float, float]:
        try:
            result = float(minimum), float(maximum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}范围必须是数字。") from exc
        if not all(math.isfinite(value) for value in result) or result[0] >= result[1]:
            raise ValueError(f"{label}最小值必须小于最大值。")
        return result

    def _sync_limit_entries(self) -> None:
        if self._closed:
            return
        energy = self.band_axes.get_ylim()
        density = self.dos_axes.get_xlim()
        self.energy_min_var.set(f"{energy[0]:.8g}")
        self.energy_max_var.set(f"{energy[1]:.8g}")
        self.dos_min_var.set(f"{density[0]:.8g}")
        self.dos_max_var.set(f"{density[1]:.8g}")

    def apply_limits(self) -> None:
        try:
            energy = self._pair(self.energy_min_var.get(), self.energy_max_var.get(), "能量")
            density = self._pair(self.dos_min_var.get(), self.dos_max_var.get(), "DOS")
        except ValueError as exc:
            messagebox.showerror("范围错误", str(exc), parent=self)
            return
        self.band_axes.set_ylim(*energy)
        self.dos_axes.set_xlim(*density)
        self.canvas.draw_idle()
        self.status_var.set(f"当前视图：能量 {energy[0]:g} 至 {energy[1]:g} eV；DOS {density[0]:g} 至 {density[1]:g}")

    def reset_limits(self) -> None:
        self.band_axes.set_ylim(*self._original_energy)
        self.dos_axes.set_xlim(*self._original_dos)
        self.canvas.draw_idle()
        self.status_var.set("已恢复完整 Band + DOS 范围。")

    def apply_plot_style(self) -> None:
        try:
            font_size = validate_font_size(self.font_size_var.get())
            line_width = validate_line_width(self.line_width_var.get())
        except ValueError as exc:
            messagebox.showerror("样式错误", str(exc), parent=self)
            return
        energy, density = self.band_axes.get_ylim(), self.dos_axes.get_xlim()
        self._plot_font_size, self._plot_line_width = font_size, line_width
        self._draw()
        self.band_axes.set_ylim(*energy)
        self.dos_axes.set_xlim(*density)
        self.canvas.draw_idle()

    def save_current_view(self) -> Path | None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        selected = filedialog.asksaveasfilename(
            parent=self,
            initialdir=str(self.output_directory),
            initialfile=f"{self.settings.export.basename}_local.png",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"), ("PDF vector", "*.pdf"), ("SVG vector", "*.svg")),
        )
        if not selected:
            return None
        path = Path(selected)
        if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            path = path.with_suffix(".png")
        self.figure.savefig(path, dpi=self.settings.export.dpi, bbox_inches="tight", facecolor="white")
        self.status_var.set(f"已保存当前双面板视图：{path}")
        return path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_edge_report()
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

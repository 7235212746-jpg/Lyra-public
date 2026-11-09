from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Property, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

from lyra.adapters.common import read_outcar_metadata, read_poscar
from lyra.adapters.inspection import (
    inspect_band_folder,
    inspect_dos_folder,
    inspect_fermi_folder,
)
from lyra.models import (
    BandPlotSettings,
    CapabilityReport,
    CombinedPlotSettings,
    DosPlotSettings,
    ExportSettings,
    FermiPlotSettings,
    PlotRange,
    ReadMode,
)
from lyra.settings_store import SettingsStore
from lyra.qt_models import ProjectionChannelModel


def _capability_summary(report: CapabilityReport) -> str:
    mode = "Lite Mode" if report.mode is ReadMode.LITE else "Full Mode"
    detected = ", ".join(report.detected_files) or "无"
    used = ", ".join(report.used_files) or "无"
    missing = ", ".join(report.missing_files) or "无"
    ignored = ", ".join(report.ignored_files) or "无"
    available = ", ".join(key for key, value in report.features.items() if value) or "无"
    limitations = "; ".join(dict.fromkeys(report.disabled_reasons.values())) or "无"
    return (
        f"{mode} | 已检测：{detected}\n"
        f"读取：{used} | 缺少：{missing} | 忽略：{ignored}\n"
        f"可用功能：{available} | 限制：{limitations}"
    )


def _progress_status(event: dict) -> str:
    labels = {
        "validate": "核验数据源",
        "cache-hit": "读取文件夹缓存",
        "memory-hit": "读取内存缓存",
        "cache-miss": "缓存未命中",
        "cache-write": "写入缓存",
        "cache-written": "缓存已写入",
        "cache-write-failed": "缓存写入失败（仅本次内存可用）",
        "source-parse": "解析数据源",
        "metadata": "读取投影元数据",
        "parse": "解析投影表",
        "complete": "投影解析完成",
        "parse-projections": "解析投影记录",
        "preview-render": "渲染预览图",
        "export": "保存图片",
    }
    phase = str(event.get("phase") or "任务进行中")
    parts = [labels.get(phase, phase)]
    detail = event.get("detail")
    if isinstance(detail, str) and detail:
        parts.append(detail)
    current, total = event.get("current"), event.get("total")
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        parts.append(f"{current:,} / {total:,}")
    elapsed = event.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed >= 0:
        parts.append(f"耗时 {elapsed:.1f} 秒")
    return " | ".join(parts)


def _quality_summary(report) -> str:
    if report is None:
        return "计算质量：尚无分析结果"
    errors = sum(issue.severity == "error" for issue in report.issues)
    warnings = sum(issue.severity == "warning" for issue in report.issues)
    return (
        f"计算质量：{report.verdict}\n"
        f"阻断 {errors} | 警告 {warnings} | 可测指标 {len(report.metrics)} 项"
    )


def _string_property(key: str):
    def getter(self):
        return str(self._state[key])

    def setter(self, value):
        self._set_value(key, str(value))

    return Property(str, getter, setter, notify=LyraControllerState.changed)


def _bool_property(key: str):
    def getter(self):
        return bool(self._state[key])

    return Property(bool, getter, notify=LyraControllerState.changed)


def _list_property(key: str):
    def getter(self):
        return list(self._state[key])

    return Property("QVariantList", getter, notify=LyraControllerState.changed)


class LyraControllerState(QObject):
    changed = Signal()


class LyraController(LyraControllerState):
    _DEFAULTS: dict[str, Any] = {
        "currentPage": "band",
        "readMode": "lite",
        "bandFolder": "",
        "bandEnergyMin": "-5",
        "bandEnergyMax": "5",
        "bandStatus": "请选择 Band 文件夹",
        "bandAvailable": False,
        "bandProjectionAvailable": False,
        "bandProjectionReason": "需要 PROCAR 才能启用元素/轨道投影。",
        "bandElements": (),
        "bandProjected": False,
        "bandProjectionStyle": "RGB",
        "dosFolder": "",
        "dosEnergyMin": "-5",
        "dosEnergyMax": "5",
        "dosDensityMin": "",
        "dosDensityMax": "",
        "dosStatus": "请选择 DOS/SCF 文件夹",
        "dosAvailable": False,
        "dosProjectionAvailable": False,
        "dosElements": (),
        "dosProjected": False,
        "combinedBandFolder": "",
        "combinedDosFolder": "",
        "combinedBandReadMode": "lite",
        "combinedDosReadMode": "lite",
        "combinedEnergyMin": "-5",
        "combinedEnergyMax": "5",
        "combinedDosDensityMin": "",
        "combinedDosDensityMax": "",
        "combinedBandStatus": "请选择 Band 文件夹",
        "combinedDosStatus": "请选择 DOS/SCF 文件夹",
        "combinedAvailable": False,
        "combinedBandProjectionAvailable": False,
        "combinedDosProjectionAvailable": False,
        "combinedReference": "共同能量参考：DOS/SCF E-fermi",
        "combinedBandProjected": False,
        "combinedDosProjected": False,
        "combinedProjectionStyle": "RGB",
        "combinedElements": (),
        "fermiFolder": "",
        "fermiViewMode": "surface",
        "fermiEnergyOffset": "0",
        "fermiSpin": "all",
        "fermiSliceAxis": "kz",
        "fermiSlicePosition": "0",
        "fermiElevation": "24",
        "fermiAzimuth": "-55",
        "fermiColour": "#3974d8",
        "fermiStatus": "请选择费米面计算文件夹",
        "fermiAvailable": False,
        "outputFolder": str(Path.home() / "Desktop" / "Lyra Outputs"),
        "dpi": "600",
        "widthMm": "180",
        "statusText": "就绪",
        "busy": False,
        "canSave": False,
        "previewUrl": "",
        "outputFiles": (),
        "qualitySummary": "计算质量：尚无分析结果",
        "qualityDetails": "尚无分析结果。",
        "qualityAvailable": False,
        "errorText": "",
        "magneticFolder": "",
        "magneticStatus": "请选择包含 POSCAR 与 OUTCAR 的文件夹",
        "canExportMagnetic": False,
        "magneticOutputFiles": (),
    }

    currentPage = _string_property("currentPage")
    readMode = _string_property("readMode")
    bandFolder = _string_property("bandFolder")
    bandEnergyMin = _string_property("bandEnergyMin")
    bandEnergyMax = _string_property("bandEnergyMax")
    bandStatus = _string_property("bandStatus")
    bandAvailable = _bool_property("bandAvailable")
    bandProjectionAvailable = _bool_property("bandProjectionAvailable")
    bandProjectionReason = _string_property("bandProjectionReason")
    bandElements = _list_property("bandElements")
    bandProjected = _bool_property("bandProjected")
    bandProjectionStyle = _string_property("bandProjectionStyle")
    dosFolder = _string_property("dosFolder")
    dosEnergyMin = _string_property("dosEnergyMin")
    dosEnergyMax = _string_property("dosEnergyMax")
    dosDensityMin = _string_property("dosDensityMin")
    dosDensityMax = _string_property("dosDensityMax")
    dosStatus = _string_property("dosStatus")
    dosAvailable = _bool_property("dosAvailable")
    dosProjectionAvailable = _bool_property("dosProjectionAvailable")
    dosElements = _list_property("dosElements")
    dosProjected = _bool_property("dosProjected")
    combinedBandFolder = _string_property("combinedBandFolder")
    combinedDosFolder = _string_property("combinedDosFolder")
    combinedBandReadMode = _string_property("combinedBandReadMode")
    combinedDosReadMode = _string_property("combinedDosReadMode")
    combinedEnergyMin = _string_property("combinedEnergyMin")
    combinedEnergyMax = _string_property("combinedEnergyMax")
    combinedDosDensityMin = _string_property("combinedDosDensityMin")
    combinedDosDensityMax = _string_property("combinedDosDensityMax")
    combinedBandStatus = _string_property("combinedBandStatus")
    combinedDosStatus = _string_property("combinedDosStatus")
    combinedAvailable = _bool_property("combinedAvailable")
    combinedBandProjectionAvailable = _bool_property("combinedBandProjectionAvailable")
    combinedDosProjectionAvailable = _bool_property("combinedDosProjectionAvailable")
    combinedReference = _string_property("combinedReference")
    combinedBandProjected = _bool_property("combinedBandProjected")
    combinedDosProjected = _bool_property("combinedDosProjected")
    combinedProjectionStyle = _string_property("combinedProjectionStyle")
    combinedElements = _list_property("combinedElements")
    fermiFolder = _string_property("fermiFolder")
    fermiViewMode = _string_property("fermiViewMode")
    fermiEnergyOffset = _string_property("fermiEnergyOffset")
    fermiSpin = _string_property("fermiSpin")
    fermiSliceAxis = _string_property("fermiSliceAxis")
    fermiSlicePosition = _string_property("fermiSlicePosition")
    fermiElevation = _string_property("fermiElevation")
    fermiAzimuth = _string_property("fermiAzimuth")
    fermiColour = _string_property("fermiColour")
    fermiStatus = _string_property("fermiStatus")
    fermiAvailable = _bool_property("fermiAvailable")
    outputFolder = _string_property("outputFolder")
    dpi = _string_property("dpi")
    widthMm = _string_property("widthMm")
    statusText = _string_property("statusText")
    busy = _bool_property("busy")
    canSave = _bool_property("canSave")
    previewUrl = _string_property("previewUrl")
    outputFiles = _list_property("outputFiles")
    qualitySummary = _string_property("qualitySummary")
    qualityDetails = _string_property("qualityDetails")
    qualityAvailable = _bool_property("qualityAvailable")
    errorText = _string_property("errorText")
    magneticFolder = _string_property("magneticFolder")
    magneticStatus = _string_property("magneticStatus")
    canExportMagnetic = _bool_property("canExportMagnetic")
    magneticOutputFiles = _list_property("magneticOutputFiles")

    def __init__(self, settings_store: SettingsStore, worker) -> None:
        super().__init__()
        self._settings_store = settings_store
        self.worker = worker
        self._state = dict(self._DEFAULTS)
        self._reports: dict[str, CapabilityReport] = {}
        self._combined_band_elements: tuple[str, ...] = ()
        self._combined_dos_elements: tuple[str, ...] = ()
        self._projection_models = {
            "band": ProjectionChannelModel(self),
            "dos": ProjectionChannelModel(self),
            "combined": ProjectionChannelModel(self),
        }
        for model in self._projection_models.values():
            model.changed.connect(self._projection_changed)
        self._active_job: dict[str, str] | None = None
        self._job_notice = ""
        self._preview_sources = ()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(120)
        self._poll_timer.timeout.connect(self.pollWorker)
        self._restore()
        self._inspect_restored_folders()

    @Property(QObject, constant=True)
    def bandProjectionModel(self):
        return self._projection_models["band"]

    @Property(QObject, constant=True)
    def dosProjectionModel(self):
        return self._projection_models["dos"]

    @Property(QObject, constant=True)
    def combinedProjectionModel(self):
        return self._projection_models["combined"]

    def _set_value(self, key: str, value: Any, *, notify: bool = True) -> bool:
        if key not in self._state:
            return False
        if self._state[key] == value:
            return False
        self._state[key] = value
        if notify:
            self.changed.emit()
        return True

    @Slot(str, str)
    def setValue(self, key: str, value: str) -> None:
        if key == "readMode" and value not in {"lite", "full"}:
            return
        if key in {"combinedBandReadMode", "combinedDosReadMode"} and value not in {"lite", "full"}:
            return
        changed = self._set_value(key, value)
        if not changed:
            return
        if key in self._plot_keys():
            self._invalidate_preview()
        if key == "readMode":
            for target in ("band", "dos", "fermi"):
                if self._state[f"{target}Folder"]:
                    self._inspect(target)
        elif key == "combinedBandReadMode" and self.combinedBandFolder:
            self._inspect("combinedBand")
        elif key == "combinedDosReadMode" and self.combinedDosFolder:
            self._inspect("combinedDos")

    @Slot(str, bool)
    def setBoolValue(self, key: str, value: bool) -> None:
        if key not in {
            "bandProjected",
            "dosProjected",
            "combinedBandProjected",
            "combinedDosProjected",
        }:
            return
        if self._set_value(key, bool(value)):
            self._invalidate_preview()

    @Slot(str)
    def addProjection(self, target: str) -> None:
        model = self._projection_models.get(target)
        if model is not None:
            model.add()

    @Slot(str, int)
    def removeProjection(self, target: str, row: int) -> None:
        model = self._projection_models.get(target)
        if model is not None:
            model.remove(row)

    @Slot(str, int, str, "QVariant")
    def setProjection(self, target: str, row: int, field: str, value) -> None:
        model = self._projection_models.get(target)
        if model is not None:
            model.setField(row, field, value)

    def _projection_changed(self) -> None:
        self._invalidate_preview()
        self.changed.emit()

    @Slot(str, str)
    def setFolder(self, target: str, folder: str) -> None:
        keys = {
            "band": "bandFolder",
            "dos": "dosFolder",
            "combinedBand": "combinedBandFolder",
            "combinedDos": "combinedDosFolder",
            "fermi": "fermiFolder",
            "output": "outputFolder",
            "magnetic": "magneticFolder",
        }
        key = keys.get(target)
        if key is None:
            return
        folder_url = QUrl(folder)
        local_folder = folder_url.toLocalFile() if folder_url.isLocalFile() else folder
        value = str(Path(local_folder).resolve()) if local_folder else ""
        self._set_value(key, value)
        if target != "output":
            self._invalidate_preview()
        if not value or target == "output":
            return
        if target == "magnetic":
            self._inspect_magnetic(Path(value))
            return
        self._inspect(target)

    def _inspect_magnetic(self, folder: Path) -> None:
        present = tuple(
            name for name in ("POSCAR", "OUTCAR") if (folder / name).is_file()
        )
        missing = tuple(name for name in ("POSCAR", "OUTCAR") if name not in present)
        self._state["canExportMagnetic"] = not missing
        if missing:
            self._state["magneticStatus"] = (
                f"已检测：{', '.join(present) or '无'}\n"
                f"缺少：{', '.join(missing)}"
            )
        else:
            self._state["magneticStatus"] = (
                "已检测：POSCAR、OUTCAR\n"
                "可导出：VESTA magnetic CIF + 磁矩证据文本"
            )
        self.changed.emit()

    def _mode_for(self, target: str) -> ReadMode:
        if target == "combinedBand":
            return ReadMode(self.combinedBandReadMode)
        if target == "combinedDos":
            return ReadMode(self.combinedDosReadMode)
        return ReadMode(self.readMode)

    @staticmethod
    def _elements(folder: Path) -> tuple[str, ...]:
        try:
            _reciprocal, elements, _counts = read_poscar(folder / "POSCAR")
        except (OSError, UnicodeError, ValueError):
            return ()
        return tuple(elements)

    def _inspect(self, target: str) -> None:
        folder_key = {
            "band": "bandFolder",
            "dos": "dosFolder",
            "combinedBand": "combinedBandFolder",
            "combinedDos": "combinedDosFolder",
            "fermi": "fermiFolder",
        }[target]
        folder = Path(str(self._state[folder_key]))
        mode = self._mode_for(target)
        if target in {"band", "combinedBand"}:
            report = inspect_band_folder(folder, mode)
        elif target in {"dos", "combinedDos"}:
            report = inspect_dos_folder(folder, mode)
        else:
            report = inspect_fermi_folder(folder, mode)
        self._reports[target] = report
        elements = self._elements(folder)

        if target == "band":
            projected = report.is_available("projected_band")
            if not projected:
                self._state["bandProjected"] = False
            self._state.update(
                bandStatus=_capability_summary(report),
                bandAvailable=report.is_available("standard_band"),
                bandProjectionAvailable=projected,
                bandProjectionReason=(
                    "可选择元素，再细分总轨道、s/p/d/f 或具体轨道。"
                    if projected
                    else report.reason("projected_band")
                ),
                bandElements=elements,
            )
        elif target == "dos":
            if not report.is_available("projected_dos"):
                self._state["dosProjected"] = False
            self._state.update(
                dosStatus=_capability_summary(report),
                dosAvailable=report.is_available("total_dos"),
                dosProjectionAvailable=report.is_available("projected_dos"),
                dosElements=elements,
            )
        elif target == "combinedBand":
            self._combined_band_elements = elements
            if not report.is_available("projected_band"):
                self._state["combinedBandProjected"] = False
            self._state.update(
                combinedBandStatus=_capability_summary(report),
                combinedBandProjectionAvailable=report.is_available("projected_band"),
            )
        elif target == "combinedDos":
            self._combined_dos_elements = elements
            if not report.is_available("projected_dos"):
                self._state["combinedDosProjected"] = False
            self._state.update(
                combinedDosStatus=_capability_summary(report),
                combinedDosProjectionAvailable=report.is_available("projected_dos"),
            )
            self._state["combinedReference"] = self._combined_reference(folder, mode, report)
        else:
            self._state.update(
                fermiStatus=_capability_summary(report),
                fermiAvailable=report.is_available("fermi_surface"),
            )
        self._state["combinedAvailable"] = bool(
            self._reports.get("combinedBand")
            and self._reports["combinedBand"].is_available("standard_band")
            and self._reports.get("combinedDos")
            and self._reports["combinedDos"].is_available("total_dos")
        )
        common_elements = tuple(
            element
            for element in self._combined_band_elements
            if element in self._combined_dos_elements
        )
        self._state["combinedElements"] = (
            common_elements
            or self._combined_band_elements
            or self._combined_dos_elements
        )
        self.changed.emit()

    @staticmethod
    def _plot_keys() -> set[str]:
        return {
            "readMode",
            "bandFolder",
            "bandEnergyMin",
            "bandEnergyMax",
            "dosFolder",
            "dosEnergyMin",
            "dosEnergyMax",
            "dosDensityMin",
            "dosDensityMax",
            "combinedBandFolder",
            "combinedDosFolder",
            "combinedBandReadMode",
            "combinedDosReadMode",
            "combinedEnergyMin",
            "combinedEnergyMax",
            "combinedDosDensityMin",
            "combinedDosDensityMax",
            "fermiFolder",
            "fermiViewMode",
            "fermiEnergyOffset",
            "fermiSpin",
            "fermiSliceAxis",
            "fermiSlicePosition",
            "fermiElevation",
            "fermiAzimuth",
            "fermiColour",
            "widthMm",
            "bandProjectionStyle",
            "combinedProjectionStyle",
        }

    def _invalidate_preview(self) -> None:
        if not self._state["canSave"]:
            return
        self._state["canSave"] = False
        self._preview_sources = ()
        self._state["statusText"] = "预览已过期"
        self.changed.emit()

    def _export_settings(self, basename: str) -> ExportSettings:
        dpi = int(self.dpi)
        width = float(self.widthMm)
        if dpi <= 0 or width <= 0:
            raise ValueError("DPI 和图宽必须大于 0。")
        return ExportSettings(
            basename=basename,
            formats=("png", "pdf", "svg"),
            dpi=dpi,
            width_mm=width,
        )

    @staticmethod
    def _range(minimum: str, maximum: str, label: str) -> PlotRange:
        try:
            result = PlotRange(float(minimum), float(maximum))
        except ValueError as exc:
            raise ValueError(f"{label}必须填写数字。") from exc
        result.validate()
        return result

    @classmethod
    def _optional_range(
        cls, minimum: str, maximum: str, label: str
    ) -> PlotRange | None:
        minimum, maximum = minimum.strip(), maximum.strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError(f"{label}需要同时填写最小值和最大值。")
        return cls._range(minimum, maximum, label)

    def _build_job(self, intent: str) -> tuple[str, dict]:
        page = self.currentPage
        common = {"mode": self.readMode, "output": self.outputFolder}
        if page == "band":
            channels = (
                self.bandProjectionModel.to_channels(tuple(self.bandElements))
                if self.bandProjected
                else ()
            )
            mode = "standard"
            switched_to_fat = False
            if self.bandProjected:
                if not channels:
                    raise ValueError("投影能带至少需要一个元素或轨道通道。")
                mode = "rgb" if self.bandProjectionStyle == "RGB" else "fat"
                if mode == "rgb" and len(channels) > 3:
                    mode, switched_to_fat = "fat", True
            settings = BandPlotSettings(
                energy_range=self._range(
                    self.bandEnergyMin, self.bandEnergyMax, "能量范围"
                ),
                mode=mode,
                projection_channels=channels,
                export=self._export_settings("lyra_band"),
            )
            payload = {**common, "folder": self.bandFolder, "settings": settings}
            if switched_to_fat:
                self._job_notice = "RGB 最多 3 条通道，已自动改用 Fat band"
        elif page == "dos":
            channels = (
                self.dosProjectionModel.to_channels(tuple(self.dosElements))
                if self.dosProjected
                else ()
            )
            settings = DosPlotSettings(
                energy_range=self._range(
                    self.dosEnergyMin, self.dosEnergyMax, "能量范围"
                ),
                density_range=self._optional_range(
                    self.dosDensityMin, self.dosDensityMax, "DOS 范围"
                ),
                projection_channels=channels,
                export=self._export_settings("lyra_dos"),
            )
            payload = {**common, "folder": self.dosFolder, "settings": settings}
        elif page == "combined":
            common_elements = tuple(
                element
                for element in self._combined_band_elements
                if element in self._combined_dos_elements
            )
            if not common_elements:
                common_elements = self._combined_band_elements or self._combined_dos_elements
            channels = self.combinedProjectionModel.to_channels(common_elements)
            band_channels = channels if self.combinedBandProjected else ()
            dos_channels = channels if self.combinedDosProjected else ()
            band_mode = "standard"
            switched_to_fat = False
            if self.combinedBandProjected:
                if not band_channels:
                    raise ValueError("Combined 能带投影至少需要一个元素或轨道通道。")
                band_mode = "rgb" if self.combinedProjectionStyle == "RGB" else "fat"
                if band_mode == "rgb" and len(band_channels) > 3:
                    band_mode, switched_to_fat = "fat", True
            if self.combinedDosProjected and not dos_channels:
                raise ValueError("Combined DOS 投影至少需要一个元素或轨道通道。")
            settings = CombinedPlotSettings(
                energy_range=self._range(
                    self.combinedEnergyMin,
                    self.combinedEnergyMax,
                    "共同能量范围",
                ),
                reference_energy=None,
                band_mode=band_mode,
                band_projection_channels=band_channels,
                dos_projection_channels=dos_channels,
                dos_density_range=self._optional_range(
                    self.combinedDosDensityMin,
                    self.combinedDosDensityMax,
                    "DOS 横轴范围",
                ),
                export=self._export_settings("lyra_band_dos"),
            )
            payload = {
                **common,
                "band_folder": self.combinedBandFolder,
                "dos_folder": self.combinedDosFolder,
                "band_mode": self.combinedBandReadMode,
                "dos_mode": self.combinedDosReadMode,
                "settings": settings,
            }
            if switched_to_fat:
                self._job_notice = "RGB 最多 3 条通道，Combined 已自动改用 Fat band"
        elif page == "fermi":
            settings = FermiPlotSettings(
                energy_offset=float(self.fermiEnergyOffset),
                view_mode=self.fermiViewMode,
                spin_channel=self.fermiSpin,
                slice_axis={"kx": 0, "ky": 1, "kz": 2}[self.fermiSliceAxis],
                slice_position=float(self.fermiSlicePosition),
                elevation=float(self.fermiElevation),
                azimuth=float(self.fermiAzimuth),
                colour=self.fermiColour,
                export=self._export_settings("lyra_fermi_surface"),
            )
            settings.validate()
            payload = {**common, "folder": self.fermiFolder, "settings": settings}
        else:
            raise ValueError("当前页面还没有可运行的绘图任务。")
        if intent == "export":
            payload["preview_sources"] = self._preview_sources
        return f"{intent}-{page}", payload

    def _start_current(self, intent: str) -> None:
        if self.worker.running:
            self._show_error("已有绘图任务正在运行，请等待完成。")
            return
        self._job_notice = ""
        try:
            operation, payload = self._build_job(intent)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self._show_error(str(exc))
            return
        if intent == "preview":
            self._state["canSave"] = False
            self._preview_sources = ()
        self._state.update(
            busy=True,
            errorText="",
            statusText=self._job_notice or "任务已开始",
        )
        self._active_job = {"page": self.currentPage, "intent": intent}
        self.changed.emit()
        self.worker.start(operation, payload)

    @Slot()
    def previewCurrent(self) -> None:
        self._start_current("preview")

    @Slot()
    def saveCurrent(self) -> None:
        if not self.canSave or not self._preview_sources:
            self._show_error("请先生成仍然有效的预览。")
            return
        self._start_current("export")

    @Slot()
    def startPolling(self) -> None:
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    @Slot()
    def pollWorker(self) -> None:
        message = self.worker.poll()
        if message is None:
            return
        kind, payload = message
        if kind == "progress":
            self._state["statusText"] = _progress_status(payload)
            self.changed.emit()
            return
        self._state["busy"] = False
        if kind == "success":
            self._handle_success(payload)
        elif kind == "cache-cleared":
            self._state["statusText"] = str(payload.get("message", "缓存已清除"))
            self._invalidate_preview()
        else:
            self._show_error(str(payload))
        self._active_job = None
        self.changed.emit()

    def _handle_success(self, result) -> None:
        intent = self._active_job["intent"] if self._active_job else "export"
        if intent == "magnetic-export":
            files = tuple(str(path) for path in result["output_files"])
            self._state.update(
                magneticOutputFiles=files,
                outputFiles=files,
                statusText="磁矩结构已导出",
            )
            return
        self._state["previewUrl"] = QUrl.fromLocalFile(
            str(Path(result.preview_file).resolve())
        ).toString()
        self._state["qualitySummary"] = _quality_summary(result.quality_report)
        self._state["qualityDetails"] = self._format_quality_details(
            result.quality_report
        )
        self._state["qualityAvailable"] = result.quality_report is not None
        if intent == "preview":
            self._preview_sources = tuple(result.source_identities)
            self._state.update(
                canSave=bool(self._preview_sources),
                outputFiles=(),
                statusText="预览完成",
            )
        else:
            files = (*result.output_files, *result.report_files)
            self._state.update(
                outputFiles=tuple(str(path) for path in files),
                statusText="图片已保存",
            )

    @staticmethod
    def _format_quality_details(report) -> str:
        if report is None:
            return "尚无分析结果。"
        lines = [f"结论：{report.verdict}", f"方法：{report.method}"]
        if report.metrics:
            lines.append("\n可测指标")
            lines.extend(
                f"• {metric.name}: {metric.value} {metric.unit}".rstrip()
                + (f"\n  {metric.interpretation}" if metric.interpretation else "")
                for metric in report.metrics
            )
        if report.issues:
            lines.append("\n问题与限制")
            lines.extend(
                f"• [{issue.severity}] {issue.summary}\n  证据：{issue.evidence}\n  建议：{issue.recommendation}"
                for issue in report.issues
            )
        if report.sources:
            lines.append("\n数据来源")
            lines.extend(f"• {source}" for source in report.sources)
        return "\n".join(lines)

    @Slot()
    def exportMagnetic(self) -> None:
        if self.worker.running:
            self._show_error("已有任务正在运行，请等待完成。")
            return
        if not self.canExportMagnetic:
            self._show_error("磁矩导出需要同一文件夹中的 POSCAR 与 OUTCAR。")
            return
        payload = {
            "folder": self.magneticFolder,
            "output": str(Path(self.outputFolder).resolve()),
        }
        self._state.update(busy=True, errorText="", statusText="正在解析最终磁矩")
        self._active_job = {"page": "magnetic", "intent": "magnetic-export"}
        self.changed.emit()
        self.worker.start("export-magnetic", payload)

    @Slot()
    def openPreviewExternal(self) -> None:
        if self.previewUrl:
            QDesktopServices.openUrl(QUrl(self.previewUrl))

    @Slot()
    def openOutputFolder(self) -> None:
        files = list(self.outputFiles)
        folder = Path(files[0]).parent if files else Path(self.outputFolder)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _show_error(self, message: str) -> None:
        self._state.update(errorText=message, busy=False, statusText="任务失败")
        self.changed.emit()

    @Slot()
    def clearError(self) -> None:
        self._set_value("errorText", "")

    @staticmethod
    def _combined_reference(
        folder: Path, mode: ReadMode, report: CapabilityReport
    ) -> str:
        if not report.is_available("total_dos"):
            return "DOS/SCF 数据不可用"
        if mode is ReadMode.FULL:
            return "共同能量参考：DOS/SCF XML E-fermi（读取时确定）"
        try:
            efermi = read_outcar_metadata(folder / "OUTCAR").efermi
        except (OSError, UnicodeError, ValueError):
            return "共同能量参考：DOS/SCF E-fermi"
        return f"共同能量参考：DOS/SCF E-fermi = {efermi:.6f} eV"

    def _restore(self) -> None:
        settings = self._settings_store.load_namespace("qt")
        if not settings:
            return
        global_settings = settings.get("global", {})
        self._restore_values(global_settings, {"current_page": "currentPage", "read_mode": "readMode"})
        self._restore_values(
            settings.get("band", {}),
            {"folder": "bandFolder", "energy_min": "bandEnergyMin", "energy_max": "bandEnergyMax"},
        )
        band = settings.get("band", {})
        if isinstance(band, dict):
            self._state["bandProjected"] = bool(band.get("projected", False))
            if band.get("projection_style") in {"RGB", "Fat band"}:
                self._state["bandProjectionStyle"] = band["projection_style"]
            self.bandProjectionModel.restore(band.get("channels"))
        self._restore_values(
            settings.get("dos", {}),
            {
                "folder": "dosFolder",
                "energy_min": "dosEnergyMin",
                "energy_max": "dosEnergyMax",
                "density_min": "dosDensityMin",
                "density_max": "dosDensityMax",
            },
        )
        dos = settings.get("dos", {})
        if isinstance(dos, dict):
            self._state["dosProjected"] = bool(dos.get("projected", False))
            self.dosProjectionModel.restore(dos.get("channels"))
        self._restore_values(
            settings.get("combined", {}),
            {
                "band_folder": "combinedBandFolder",
                "dos_folder": "combinedDosFolder",
                "band_read_mode": "combinedBandReadMode",
                "dos_read_mode": "combinedDosReadMode",
                "energy_min": "combinedEnergyMin",
                "energy_max": "combinedEnergyMax",
                "dos_density_min": "combinedDosDensityMin",
                "dos_density_max": "combinedDosDensityMax",
            },
        )
        combined = settings.get("combined", {})
        if isinstance(combined, dict):
            self._state["combinedBandProjected"] = bool(combined.get("band_projected", False))
            self._state["combinedDosProjected"] = bool(combined.get("dos_projected", False))
            if combined.get("projection_style") in {"RGB", "Fat band"}:
                self._state["combinedProjectionStyle"] = combined["projection_style"]
            self.combinedProjectionModel.restore(combined.get("channels"))
        self._restore_values(
            settings.get("fermi", {}),
            {
                "folder": "fermiFolder",
                "view_mode": "fermiViewMode",
                "energy_offset": "fermiEnergyOffset",
                "spin": "fermiSpin",
                "slice_axis": "fermiSliceAxis",
                "slice_position": "fermiSlicePosition",
                "elevation": "fermiElevation",
                "azimuth": "fermiAzimuth",
                "colour": "fermiColour",
            },
        )
        self._restore_values(
            settings.get("export", {}),
            {"folder": "outputFolder", "dpi": "dpi", "width_mm": "widthMm"},
        )
        self._restore_values(
            settings.get("magnetic", {}), {"folder": "magneticFolder"}
        )

    def _restore_values(self, section: Any, mapping: dict[str, str]) -> None:
        if not isinstance(section, dict):
            return
        for source, target in mapping.items():
            value = section.get(source)
            if isinstance(value, str):
                self._state[target] = value

    def _inspect_restored_folders(self) -> None:
        for target, key in (
            ("band", "bandFolder"),
            ("dos", "dosFolder"),
            ("combinedBand", "combinedBandFolder"),
            ("combinedDos", "combinedDosFolder"),
            ("fermi", "fermiFolder"),
        ):
            value = str(self._state[key])
            if value and Path(value).is_dir():
                self._inspect(target)
        magnetic = str(self._state["magneticFolder"])
        if magnetic and Path(magnetic).is_dir():
            self._inspect_magnetic(Path(magnetic))

    def _snapshot(self) -> dict[str, object]:
        return {
            "global": {"current_page": self.currentPage, "read_mode": self.readMode},
            "band": {
                "folder": self.bandFolder,
                "energy_min": self.bandEnergyMin,
                "energy_max": self.bandEnergyMax,
                "projected": self.bandProjected,
                "projection_style": self.bandProjectionStyle,
                "channels": self.bandProjectionModel.snapshot(),
            },
            "dos": {
                "folder": self.dosFolder,
                "energy_min": self.dosEnergyMin,
                "energy_max": self.dosEnergyMax,
                "density_min": self.dosDensityMin,
                "density_max": self.dosDensityMax,
                "projected": self.dosProjected,
                "channels": self.dosProjectionModel.snapshot(),
            },
            "combined": {
                "band_folder": self.combinedBandFolder,
                "dos_folder": self.combinedDosFolder,
                "band_read_mode": self.combinedBandReadMode,
                "dos_read_mode": self.combinedDosReadMode,
                "energy_min": self.combinedEnergyMin,
                "energy_max": self.combinedEnergyMax,
                "dos_density_min": self.combinedDosDensityMin,
                "dos_density_max": self.combinedDosDensityMax,
                "band_projected": self.combinedBandProjected,
                "dos_projected": self.combinedDosProjected,
                "projection_style": self.combinedProjectionStyle,
                "channels": self.combinedProjectionModel.snapshot(),
            },
            "fermi": {
                "folder": self.fermiFolder,
                "view_mode": self.fermiViewMode,
                "energy_offset": self.fermiEnergyOffset,
                "spin": self.fermiSpin,
                "slice_axis": self.fermiSliceAxis,
                "slice_position": self.fermiSlicePosition,
                "elevation": self.fermiElevation,
                "azimuth": self.fermiAzimuth,
                "colour": self.fermiColour,
            },
            "export": {
                "folder": self.outputFolder,
                "dpi": self.dpi,
                "width_mm": self.widthMm,
                "formats": {"png": True, "pdf": True, "svg": True},
            },
            "magnetic": {"folder": self.magneticFolder},
        }

    @Slot()
    def save(self) -> None:
        self._settings_store.save_namespace("qt", self._snapshot())

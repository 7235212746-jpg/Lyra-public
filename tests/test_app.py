import tempfile
import threading
import time
import tkinter as tk
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch
from tkinter import ttk

from PIL import Image

from lyra.adapters.inspection import inspect_band_folder
from lyra.cache import CacheIdentity, CacheSource
from lyra.app import (
    ORBITALS,
    LyraApp,
    cache_status_text,
    capability_summary,
    format_progress_status,
    initial_window_geometry,
    quality_summary,
    required_files_text,
    workbench_sections,
)
from lyra.data_service import ProjectionDataService
from lyra.models import (
    BandPlotSettings,
    BandStateSelection,
    CapabilityReport,
    CombinedPlotSettings,
    GeneratedResult,
    PlotRange,
    ReadMode,
)
from lyra.settings_store import SettingsStore
from tests.fixtures import (
    write_band_fixture,
    write_dos_fixture,
    write_fermi_fixture,
    write_poscar,
    write_procar,
)


class AppTests(unittest.TestCase):
    def test_local_zoom_button_explains_missing_preview_instead_of_staying_disabled(self):
        app = LyraApp()
        try:
            self.assertEqual(
                str(app.zoom_preview_button.cget("state")),
                tk.NORMAL,
            )
            with patch("lyra.app.messagebox.showinfo") as showinfo:
                app._open_preview_viewer()

            showinfo.assert_called_once()
            self.assertIn("生成一次预览", showinfo.call_args.args[1])
        finally:
            app.destroy()

    def test_settings_language_selector_switches_main_workspace_and_persists(self):
        app = LyraApp()
        try:
            self.assertEqual(app.language_var.get(), "zh_CN")
            self.assertEqual(
                tuple(app.language_box.cget("values")),
                ("中文", "English"),
            )

            app.language_display_var.set("English")
            app._language_changed()

            self.assertEqual(app.language_var.get(), "en_US")
            self.assertEqual(app.nav_buttons["band"].cget("text"), "Band")
            self.assertEqual(app.nav_buttons["settings"].cget("text"), "Settings")
            self.assertEqual(app.sidebar_title_label.cget("text"), "Plot Modules")
            self.assertEqual(
                app._settings_snapshot()["global"]["language"],
                "en_US",
            )
        finally:
            app.destroy()

    def test_classic_feature_entries_receive_a_real_settings_store(self):
        app = LyraApp()
        try:
            cases = (
                ("lyra.remote_window.RemoteWorkspaceWindow", app._open_remote_workspace),
                ("lyra.incar_window.IncarAssistantWindow", app._open_incar_assistant),
                ("lyra.mobile_window.MobileControlWindow", app._open_mobile_control),
            )
            for target, opener in cases:
                window = Mock()
                with patch(target, return_value=window) as constructor:
                    opener()
                self.assertIsInstance(
                    constructor.call_args.kwargs["settings_store"], SettingsStore
                )
        finally:
            app.destroy()

    @staticmethod
    def app_with_worker_message(message):
        app = LyraApp()
        app.worker.close()
        job_id = uuid.uuid4()
        if isinstance(message, tuple):
            message = {"job_id": job_id, "kind": message[0], "payload": message[1]}
        worker = Mock()
        worker.poll.return_value = message
        worker.running = True
        app.worker = worker
        app._current_worker_job_id = message["job_id"] if message is not None else None
        return app

    def test_cancel_button_tracks_cancellable_job_and_disables_repeat(self):
        app = LyraApp()
        try:
            app.update_idletasks()
            self.assertFalse(app.cancel_job_button.winfo_ismapped())
            app.worker.close()
            worker = Mock()
            worker.running = False
            worker.start.return_value = uuid.uuid4()
            worker.cancel.return_value = True
            app.worker = worker

            app._start_job("preview-band", {}, page="band", intent="preview")
            app.update_idletasks()
            self.assertEqual(app._current_worker_job_id, worker.start.return_value)
            self.assertTrue(app.cancel_job_button.winfo_ismapped())
            self.assertEqual(str(app.cancel_job_button.cget("state")), tk.NORMAL)

            app.cancel_job_button.invoke()

            worker.cancel.assert_called_once_with()
            self.assertEqual(app.task_status_var.get(), "正在取消")
            self.assertEqual(str(app.cancel_job_button.cget("state")), tk.DISABLED)
        finally:
            app.destroy()

    def test_clear_cache_never_offers_cancel(self):
        app = LyraApp()
        try:
            app.worker.close()
            worker = Mock()
            worker.running = False
            worker.start.return_value = uuid.uuid4()
            app.worker = worker
            app.band_folder_var.set("C:/calculation")

            app._clear_cache("band")
            app.update_idletasks()

            self.assertEqual(app._current_worker_job_id, worker.start.return_value)
            self.assertFalse(app.cancel_job_button.winfo_ismapped())
        finally:
            app.destroy()

    def test_stale_uuid_terminal_cannot_restore_controls_or_change_status(self):
        active_id = uuid.uuid4()
        app = self.app_with_worker_message(
            {"job_id": uuid.uuid4(), "kind": "success", "payload": Mock()}
        )
        try:
            app._current_worker_job_id = active_id
            app._active_job = {"page": "band", "intent": "preview"}
            app.task_status_var.set("still running")
            with (
                patch.object(app, "_set_job_controls") as controls,
                patch.object(app, "_show_result") as show_result,
                patch.object(app, "_schedule_poll"),
            ):
                app._poll_worker()

            self.assertEqual(app.task_status_var.get(), "still running")
            self.assertEqual(app._current_worker_job_id, active_id)
            controls.assert_not_called()
            show_result.assert_not_called()
        finally:
            app.destroy()

    def test_matching_cancelled_terminal_restores_controls_without_error_dialog(self):
        job_id = uuid.uuid4()
        app = self.app_with_worker_message(
            {"job_id": job_id, "kind": "cancelled", "payload": None}
        )
        try:
            app._active_job = {"page": "band", "intent": "preview"}
            with (
                patch.object(app, "_set_job_controls") as controls,
                patch.object(app, "_schedule_poll"),
                patch("lyra.app.messagebox.showerror") as showerror,
            ):
                app._poll_worker()

            self.assertEqual(app.task_status_var.get(), "任务已取消")
            self.assertIsNone(app._current_worker_job_id)
            controls.assert_called_once_with("normal")
            showerror.assert_not_called()
        finally:
            app.destroy()

    def test_required_files_text_covers_each_workflow_and_mode(self):
        lite_band = required_files_text("band", ReadMode.LITE)
        full_band = required_files_text("band", ReadMode.FULL)
        lite_dos = required_files_text("dos", ReadMode.LITE)
        full_dos = required_files_text("dos", ReadMode.FULL)
        lite_fermi = required_files_text("fermi", ReadMode.LITE)
        full_fermi = required_files_text("fermi", ReadMode.FULL)
        combined = required_files_text("combined", ReadMode.LITE)

        self.assertIn("EIGENVAL + KPOINTS + POSCAR + OUTCAR", lite_band)
        self.assertIn("额外需要 PROCAR", lite_band)
        self.assertIn("vasprun.xml(.gz) + KPOINTS", full_band)
        self.assertIn("DOSCAR + POSCAR + OUTCAR", lite_dos)
        self.assertIn("DOS/PDOS：vasprun.xml(.gz)", full_dos)
        self.assertIn("均匀", lite_fermi)
        self.assertIn("ISYM=-1", lite_fermi)
        self.assertIn("vasprun.xml(.gz)", full_fermi)
        self.assertIn("Band 文件夹", combined)
        self.assertIn("DOS/SCF 文件夹", combined)

    def test_quality_summary_reports_evidence_counts_without_score(self):
        from lyra.models import AnalysisQualityReport, QualityIssue, QualityMetric

        report = AnalysisQualityReport(
            verdict="可分析，但需核查警告",
            method="evidence",
            metrics=(QualityMetric("k 点数", "100"),),
            issues=(
                QualityIssue("warning", "x", "warning", "evidence", "check"),
            ),
        )

        summary = quality_summary(report)

        self.assertIn("警告 1", summary)
        self.assertIn("可测指标 1 项", summary)
        self.assertNotIn("%", summary)

    def test_required_file_labels_refresh_when_mode_changes(self):
        app = LyraApp()
        try:
            self.assertIn("EIGENVAL", app.band_required_var.get())
            self.assertIn("DOSCAR", app.dos_required_var.get())
            self.assertIn("ISYM=-1", app.fermi_required_var.get())

            app.read_mode_var.set(ReadMode.FULL.value)
            app._mode_changed()

            self.assertIn("vasprun.xml(.gz)", app.band_required_var.get())
            self.assertNotIn("EIGENVAL", app.band_required_var.get())
            self.assertIn("vasprun.xml(.gz)", app.fermi_required_var.get())
            self.assertIn("Band 文件夹", app.combined_required_var.get())
        finally:
            app.destroy()

    def test_initial_window_geometry_fits_laptop_and_desktop(self):
        laptop = initial_window_geometry(1366, 768)
        desktop = initial_window_geometry(1920, 1080)
        laptop_width, laptop_height = (
            int(value) for value in laptop.split("+", 1)[0].split("x")
        )
        desktop_width, desktop_height = (
            int(value) for value in desktop.split("+", 1)[0].split("x")
        )

        self.assertLessEqual(laptop_width, 1266)
        self.assertLessEqual(laptop_height, 680)
        self.assertGreater(desktop_width, laptop_width)
        self.assertGreater(desktop_height, laptop_height)

    def test_navigation_exposes_current_and_future_modules(self):
        sections = workbench_sections()

        self.assertEqual(
            [section["label"] for section in sections],
            [
                "能带",
                "态密度",
                "能带 + 态密度",
                "能带对照",
                "费米面",
                "成键分析 · 规划中",
                "价态与电荷 · 规划中",
                "远程 Linux",
                "INCAR 助手",
                "手机遥控",
                "设置",
            ],
        )
        enabled = {section["key"]: section["enabled"] for section in sections}
        self.assertTrue(enabled["fermi"])
        self.assertTrue(enabled["paired_band"])
        self.assertFalse(enabled["bonding"])
        self.assertFalse(enabled["valence_charge"])
        self.assertTrue(enabled["remote"])
        self.assertTrue(enabled["incar"])
        self.assertTrue(enabled["mobile"])
        tooltips = {
            section["key"]: section["tooltip"]
            for section in sections
            if not section["enabled"]
        }
        self.assertIn("COHPCAR.lobster", tooltips["bonding"])
        self.assertIn("ACF.dat", tooltips["valence_charge"])

    def test_paired_band_page_has_two_folders_and_independent_modes(self):
        app = LyraApp()
        try:
            self.assertIn("paired_band", app.pages)
            self.assertEqual(app.paired_left_mode_var.get(), "lite")
            self.assertEqual(app.paired_right_mode_var.get(), "lite")
            self.assertEqual(str(app.paired_run_button.cget("state")), tk.DISABLED)
        finally:
            app.destroy()

    def test_planned_navigation_has_no_page_or_command_and_stays_disabled(self):
        app = LyraApp()
        try:
            for key in ("bonding", "valence_charge"):
                button = app.nav_buttons[key]
                self.assertEqual(str(button.cget("state")), tk.DISABLED)
                self.assertEqual(button.cget("command"), "")
                self.assertNotIn(key, app.pages)
                with patch.object(app, "_select_page") as select_page:
                    button.invoke()
                select_page.assert_not_called()

            app._set_job_controls("disabled")
            app._set_job_controls("normal")
            for key in ("remote", "incar", "mobile"):
                self.assertNotEqual(str(app.nav_buttons[key].cget("state")), tk.DISABLED)
                self.assertIn(key, app.pages)
        finally:
            app.destroy()

    def test_active_navigation_and_segmented_mode_have_distinct_styles(self):
        app = LyraApp()
        try:
            self.assertEqual(app.nav_buttons["band"].cget("style"), "ActiveNav.TButton")
            self.assertEqual(app.nav_buttons["dos"].cget("style"), "Nav.TButton")
            app._select_page("dos")
            self.assertEqual(app.nav_buttons["band"].cget("style"), "Nav.TButton")
            self.assertEqual(app.nav_buttons["dos"].cget("style"), "ActiveNav.TButton")
            self.assertEqual(
                app.read_mode_buttons["lite"].cget("style"),
                "Segmented.TRadiobutton",
            )
            self.assertEqual(
                app.style.lookup("Segmented.TRadiobutton", "background", ("selected",)),
                "#006b6d",
            )
        finally:
            app.destroy()

    def test_backdrop_is_scheduled_after_construction_and_cancelled_on_destroy(self):
        with patch("lyra.app.apply_system_backdrop", return_value=False) as apply:
            app = LyraApp()
            try:
                self.assertIsNotNone(app._backdrop_after_id)
                app.update()
                apply.assert_called_once_with(app)
            finally:
                app.destroy()
            self.assertIsNone(app._backdrop_after_id)

    def test_backdrop_failure_is_silent(self):
        with patch("lyra.app.apply_system_backdrop", side_effect=OSError("DWM unavailable")):
            app = LyraApp()
            try:
                app.update()
                self.assertTrue(app.winfo_exists())
            finally:
                app.destroy()

    def test_repeated_lifecycle_has_no_delayed_theme_or_stale_after_exceptions(self):
        errors = []
        for _ in range(3):
            app = LyraApp()
            app.report_callback_exception = lambda *exc_info: errors.append(exc_info)
            app.update()
            app.destroy()
            tk._default_root = None

        self.assertEqual(errors, [])

    def test_direct_worker_poll_cancels_the_recorded_pending_callback(self):
        app = self.app_with_worker_message(None)
        try:
            pending_id = app._poll_after_id
            self.assertIsNotNone(pending_id)
            self.assertIn(pending_id, app.tk.call("after", "info"))

            def assert_cancelled_before_reschedule():
                self.assertNotIn(pending_id, app.tk.call("after", "info"))

            with patch.object(
                app,
                "_schedule_poll",
                side_effect=assert_cancelled_before_reschedule,
            ) as schedule_poll:
                app._poll_worker()

            self.assertNotIn(pending_id, app.tk.call("after", "info"))
            schedule_poll.assert_called_once_with()
        finally:
            app.destroy()

    def test_tk_dispatched_worker_poll_leaves_exactly_one_replacement(self):
        app = self.app_with_worker_message(None)
        errors = []
        try:
            app.report_callback_exception = lambda *exc_info: errors.append(exc_info)
            app.after_cancel(app._poll_after_id)
            app._poll_after_id = app.after(0, app._poll_worker)

            app.update()

            pending_ids = app.tk.call("after", "info")
            self.assertIn(app._poll_after_id, pending_ids)
            poll_ids = [
                after_id
                for after_id in pending_ids
                if "_poll_worker" in str(app.tk.call("after", "info", after_id))
            ]
            self.assertEqual(poll_ids, [app._poll_after_id])
            self.assertEqual(errors, [])
        finally:
            app.destroy()

    def test_minimum_geometry_uses_resizable_workspace_preview_panes(self):
        app = LyraApp()
        try:
            app.geometry("1080x640+0+0")
            app.update()
            self.assertEqual(app.minsize(), (1080, 640))
            self.assertIsInstance(app.content_panes, ttk.Panedwindow)
            self.assertEqual(
                tuple(app.content_panes.panes()),
                (str(app.workspace), str(app.preview_panel)),
            )
            self.assertEqual(app.style.lookup("Preview.TLabel", "background"), "#ffffff")
        finally:
            app.destroy()

    def test_capability_summary_reports_used_missing_and_ignored_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            (folder / "vasprun.xml").write_text("xml", encoding="utf-8")
            report = inspect_band_folder(folder, ReadMode.LITE)

            summary = capability_summary(report)

        self.assertIn("Lite Mode", summary)
        self.assertIn("EIGENVAL", summary)
        self.assertIn("vasprun.xml", summary)
        self.assertIn("PROCAR", summary)

    def test_app_defaults_to_lite_and_keeps_projection_unavailable_until_detected(self):
        app = LyraApp()
        try:
            app.geometry("1200x650+0+0")
            app.update()

            self.assertEqual(app.read_mode_var.get(), ReadMode.LITE.value)
            self.assertIn("band", app.nav_buttons)
            self.assertIn("dos", app.nav_buttons)
            self.assertIn("combined", app.nav_buttons)
            self.assertEqual(str(app.nav_buttons["fermi"].cget("state")), tk.NORMAL)
            self.assertEqual(str(app.band_projection_mode.cget("state")), tk.DISABLED)
            self.assertTrue(app.band_min_var.get())
            self.assertTrue(app.dos_max_var.get())
            self.assertEqual(app.combined_reference_var.get(), "等待 DOS/SCF 文件夹")
            self.assertEqual(app.combined_band_read_mode_var.get(), ReadMode.LITE.value)
            self.assertEqual(app.combined_dos_read_mode_var.get(), ReadMode.LITE.value)
        finally:
            app.destroy()

    def test_fermi_page_accepts_uniform_grid_and_sends_slice_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_fermi_fixture(folder)
            app = LyraApp()
            try:
                app.fermi_folder_var.set(str(folder))
                app._choose_fermi_folder(False)

                self.assertEqual(str(app.fermi_run_button.cget("state")), tk.NORMAL)
                self.assertIn("fermi_surface", app.fermi_status_var.get())

                app.fermi_view_mode_var.set("slice")
                app.fermi_energy_offset_var.set("0.15")
                app.fermi_spin_var.set("total")
                app.fermi_slice_axis_var.set("kz")
                app.fermi_slice_position_var.set("0.1")
                with patch.object(app, "_start_job") as start_job:
                    app._run_fermi()

                operation, payload = start_job.call_args.args[:2]
                settings = payload["settings"]
                self.assertEqual(operation, "preview-fermi")
                self.assertEqual(settings.view_mode, "slice")
                self.assertEqual(settings.energy_offset, 0.15)
                self.assertEqual(settings.spin_channel, "total")
                self.assertEqual(settings.slice_axis, 2)
                self.assertEqual(settings.slice_position, 0.1)
            finally:
                app.destroy()

    def test_resolved_dx2_y2_orbital_uses_the_parser_name(self):
        self.assertIn("dx2-y2", ORBITALS)
        self.assertNotIn("dx2", ORBITALS)

    def test_projection_swatch_opens_picker_and_updates_channel_colour(self):
        app = LyraApp()
        try:
            row = app.band_projection_rows[0]
            colour = row[3]
            swatch = row[7]
            row[0].set(True)
            row[1].set("Ta")

            with patch(
                "lyra.app.colorchooser.askcolor",
                return_value=((18, 52, 86), "#123456"),
            ):
                swatch.invoke()

            self.assertEqual(colour.get(), "#123456")
            self.assertEqual(swatch.cget("bg"), "#123456")
            self.assertEqual(app._channels(app.band_projection_rows)[0].colour, "#123456")
        finally:
            app.destroy()

    def test_complete_band_colour_defaults_to_black_and_is_sent_to_worker(self):
        app = LyraApp()
        try:
            self.assertEqual(app.band_complete_colour_var.get(), "#000000")
            with patch.object(app, "_start_job") as start_job:
                app._run_band()
            settings = start_job.call_args.args[1]["settings"]
            self.assertEqual(settings.complete_band_colour, "#000000")
        finally:
            app.destroy()

    def test_band_page_uses_scrollable_content_like_other_long_plot_pages(self):
        app = LyraApp()
        try:
            self.assertTrue(hasattr(app, "band_scroll_canvas"))
            self.assertTrue(app.band_scroll_canvas.cget("yscrollcommand"))
            self.assertTrue(app.band_scroll_canvas.bind("<MouseWheel>"))
        finally:
            app.destroy()

    def test_combined_complete_band_colour_defaults_to_black_and_is_sent_to_worker(self):
        app = LyraApp()
        try:
            self.assertEqual(app.combined_complete_colour_var.get(), "#000000")
            with patch.object(app, "_start_job") as start_job:
                app._run_combined()
            settings = start_job.call_args.args[1]["settings"]
            self.assertEqual(settings.band_complete_colour, "#000000")
        finally:
            app.destroy()

    def test_dos_density_range_is_sent_to_plot_worker(self):
        app = LyraApp()
        try:
            app.dos_density_min_var.set("0")
            app.dos_density_max_var.set("100")

            with patch.object(app, "_start_job") as start_job:
                app._run_dos()

            settings = start_job.call_args.args[1]["settings"]
            self.assertEqual(settings.density_range, PlotRange(0.0, 100.0))
        finally:
            app.destroy()

    def test_combined_dos_density_range_is_sent_to_plot_worker(self):
        app = LyraApp()
        try:
            app.combined_dos_density_min_var.set("-100")
            app.combined_dos_density_max_var.set("100")

            with patch.object(app, "_start_job") as start_job:
                app._run_combined()

            settings = start_job.call_args.args[1]["settings"]
            self.assertEqual(settings.dos_density_range, PlotRange(-100.0, 100.0))
        finally:
            app.destroy()

    def test_combined_owns_modes_and_projection_controls(self):
        app = LyraApp()
        try:
            app.combined_band_read_mode_var.set(ReadMode.FULL.value)
            app.combined_dos_read_mode_var.set(ReadMode.LITE.value)
            app.combined_band_projected_var.set(True)
            app.combined_dos_projected_var.set(True)
            app.combined_projection_style_var.set("Fat band")
            app.combined_projection_rows[0][0].set(True)
            app.combined_projection_rows[0][1].set("Ta")
            app.combined_projection_rows[0][2].set("d")

            standalone_signature = app._plot_signature("combined")
            app.band_folder_var.set("C:/unrelated-band")
            app.dos_folder_var.set("C:/unrelated-dos")
            self.assertEqual(app._plot_signature("combined"), standalone_signature)

            with patch.object(app, "_start_job") as start_job:
                app._run_combined()

            payload = start_job.call_args.args[1]
            settings = payload["settings"]
            self.assertEqual(payload["band_mode"], ReadMode.FULL.value)
            self.assertEqual(payload["dos_mode"], ReadMode.LITE.value)
            self.assertEqual(settings.band_mode, "fat")
            self.assertEqual(settings.band_projection_channels[0].element, "Ta")
            self.assertEqual(settings.dos_projection_channels[0].orbital, "d")
        finally:
            app.destroy()

    def test_combined_rgb_with_four_channels_requires_explicit_style_change(self):
        app = LyraApp()
        try:
            app._add_projection_channel("combined")
            for row, element in zip(
                app.combined_projection_rows,
                ("Zr", "Sn", "Ge", "V"),
            ):
                row[0].set(True)
                row[1].set(element)
            app.combined_band_projected_var.set(True)
            app.combined_dos_projected_var.set(True)
            app.combined_projection_style_var.set("RGB")

            with patch.object(app, "_start_job") as start_job, patch(
                "lyra.app.messagebox.showerror"
            ) as showerror:
                app._run_combined()

            start_job.assert_not_called()
            self.assertEqual(app.combined_projection_style_var.get(), "RGB")
            self.assertIn("最多 3", showerror.call_args.args[1])
        finally:
            app.destroy()

    def test_combined_controls_scroll_into_view_on_minimum_window(self):
        app = LyraApp()
        try:
            app.geometry("1080x640+0+0")
            app._select_page("combined")
            app.update()

            canvas = app.combined_scroll_canvas
            content_bottom = canvas.bbox("all")[3]
            self.assertGreater(content_bottom, canvas.winfo_height())

            canvas.yview_moveto(1.0)
            app.update()
            viewport_top = canvas.winfo_rooty()
            viewport_bottom = viewport_top + canvas.winfo_height()
            button_top = app.combined_run_button.winfo_rooty()
            button_bottom = button_top + app.combined_run_button.winfo_height()
            self.assertGreaterEqual(button_top, viewport_top)
            self.assertLessEqual(button_bottom, viewport_bottom)
        finally:
            app.destroy()

    def test_dos_controls_scroll_into_view_on_minimum_window(self):
        app = LyraApp()
        try:
            app.geometry("1080x640+0+0")
            app._select_page("dos")
            app.update()

            canvas = app.dos_scroll_canvas
            content_bottom = canvas.bbox("all")[3]
            self.assertGreater(content_bottom, canvas.winfo_height())

            canvas.yview_moveto(1.0)
            app.update()
            viewport_top = canvas.winfo_rooty()
            viewport_bottom = viewport_top + canvas.winfo_height()
            button_top = app.dos_run_button.winfo_rooty()
            button_bottom = button_top + app.dos_run_button.winfo_height()
            self.assertGreaterEqual(button_top, viewport_top)
            self.assertLessEqual(button_bottom, viewport_bottom)
        finally:
            app.destroy()

    def test_quaternary_dos_exposes_at_least_four_projection_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_dos_fixture(folder)
            write_poscar(
                folder,
                elements=("Zr", "Sn", "Ge", "Ta"),
                counts=(1, 1, 1, 1),
            )
            app = LyraApp()
            try:
                app.dos_folder_var.set(str(folder))
                app._choose_dos_folder(False)

                self.assertGreaterEqual(len(app.dos_projection_rows), 4)
                self.assertEqual(
                    app.dos_projection_rows[3][6].cget("text"),
                    "通道 4",
                )
                self.assertEqual(
                    [row[1].get() for row in app.dos_projection_rows[:4]],
                    ["Zr", "Sn", "Ge", "Ta"],
                )
            finally:
                app.destroy()

    def test_preview_hides_empty_output_list_and_fits_visible_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            preview = folder / "preview.png"
            Image.new("RGB", (1600, 1000), "white").save(preview)
            app = LyraApp()
            try:
                app.geometry("1438x774+0+0")
                app.update()
                app._active_job = {"intent": "preview", "page": "dos"}
                app._show_result(
                    GeneratedResult((preview,), preview, "preview")
                )
                app.update()

                self.assertFalse(app.output_list.winfo_ismapped())
                self.assertLessEqual(
                    app._preview_photo.height(),
                    app.preview_label.winfo_height(),
                )
                self.assertLessEqual(
                    app._preview_photo.width(),
                    app.preview_label.winfo_width(),
                )
            finally:
                app.destroy()

    def test_saved_output_list_expands_and_opens_selected_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            preview = folder / "preview.png"
            Image.new("RGB", (800, 500), "white").save(preview)
            app = LyraApp()
            try:
                app.update()
                app._active_job = {"intent": "export", "page": "dos"}
                app._show_result(
                    GeneratedResult((preview,), preview, "export")
                )
                app.update()
                app.output_list.selection_set(0)

                self.assertTrue(app.output_list.winfo_ismapped())
                with patch("lyra.app.os.startfile") as startfile:
                    app._open_selected_output()

                startfile.assert_called_once_with(str(preview))
            finally:
                app.destroy()

    def test_active_progress_elapsed_time_keeps_ticking_between_worker_events(self):
        app = LyraApp()
        try:
            with patch("lyra.app.time.monotonic", side_effect=(100.0, 102.5)):
                app._apply_progress(
                    {
                        "phase": "source-parse",
                        "detail": "Parsing Full Mode XML.",
                        "elapsed_seconds": 0.0,
                    }
                )
                app._refresh_active_progress()

            self.assertIn("耗时 2.5 秒", app.task_status_var.get())
        finally:
            app.destroy()

    def test_large_full_xml_projection_detection_runs_off_the_gui_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "KPOINTS").write_text("fixture", encoding="utf-8")
            with (folder / "vasprun.xml").open("wb") as handle:
                handle.seek(9 * 1024 * 1024)
                handle.write(b"x")
            pending = CapabilityReport(
                folder=folder,
                mode=ReadMode.FULL,
                detected_files=("KPOINTS", "vasprun.xml"),
                used_files=("KPOINTS", "vasprun.xml"),
                ignored_files=(),
                missing_files=(),
                features={"standard_band": True, "projected_band": False},
                disabled_reasons={"projected_band": "Projection data is being inspected."},
            )
            projected = CapabilityReport(
                folder=folder,
                mode=ReadMode.FULL,
                detected_files=pending.detected_files,
                used_files=pending.used_files,
                ignored_files=(),
                missing_files=(),
                features={"standard_band": True, "projected_band": True},
            )
            started = threading.Event()
            release = threading.Event()
            full_scans = []

            def inspect(_folder, _mode, *, scan_projection_sections=True):
                if not scan_projection_sections:
                    return pending
                full_scans.append((_folder, _mode))
                started.set()
                release.wait(2.0)
                return projected

            app = LyraApp()
            try:
                app._projection_preferences["band_data_type"] = "projected"
                app._projection_preferences["combined_band_projected"] = True
                app.read_mode_var.set(ReadMode.FULL.value)
                app.band_folder_var.set(str(folder))
                app.combined_band_read_mode_var.set(ReadMode.FULL.value)
                app.combined_band_folder_var.set(str(folder))
                with patch("lyra.app.inspect_band_folder", side_effect=inspect):
                    before = time.perf_counter()
                    app._choose_band_folder(False)
                    app._choose_combined_band(False)
                    elapsed = time.perf_counter() - before
                    self.assertLess(elapsed, 0.5)
                    self.assertTrue(started.wait(1.0))
                    self.assertEqual(len(full_scans), 1)
                    self.assertEqual(str(app.band_projected_radio.cget("state")), tk.DISABLED)

                    release.set()
                    deadline = time.monotonic() + 2.0
                    while (
                        str(app.band_projected_radio.cget("state")) != tk.NORMAL
                        or str(app.combined_band_projection_toggle.cget("state"))
                        != tk.NORMAL
                    ):
                        app._poll_inspection_results()
                        if time.monotonic() >= deadline:
                            self.fail("Background Full XML inspection did not update the GUI.")
                        time.sleep(0.01)
                    self.assertEqual(app.band_data_type_var.get(), "projected")
                    self.assertTrue(app.combined_band_projected_var.get())
                    self.assertEqual(len(full_scans), 1)
            finally:
                release.set()
                app.destroy()

    def test_background_inspection_waits_for_active_plot_job_before_updating_controls(self):
        app = LyraApp()
        try:
            app.worker.close()
            app.worker = Mock()
            app.worker.running = True
            key = (
                "band",
                "C:/calculation",
                "full",
                "band",
                "vasprun.xml",
                9 * 1024 * 1024,
                1,
            )
            app._inspection_queue.put((key, None))

            with patch.object(app, "_choose_band_folder") as choose:
                app._poll_inspection_results()
                choose.assert_not_called()

                app.worker.running = False
                app._poll_inspection_results()
                choose.assert_called_once_with(False)
        finally:
            app.destroy()

    def test_full_mode_allows_manual_element_when_xml_folder_has_no_poscar(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "vasprun.xml").write_text(
                "<modeling><partial><array /></partial></modeling>",
                encoding="utf-8",
            )
            app = LyraApp()
            try:
                app.read_mode_var.set(ReadMode.FULL.value)
                app.dos_folder_var.set(str(folder))
                app._choose_dos_folder(False)

                element_box = app.dos_projection_rows[0][4]
                self.assertEqual(str(app.dos_projection_toggle.cget("state")), tk.NORMAL)
                self.assertEqual(str(element_box.cget("state")), tk.NORMAL)
            finally:
                app.destroy()

    def test_band_defaults_to_complete_data_independently_of_projection_style(self):
        app = LyraApp()
        try:
            self.assertEqual(app.band_data_type_var.get(), "complete")
            self.assertEqual(app.band_projection_style_var.get(), "RGB")
            self.assertEqual(app._band_render_mode(), "standard")
            self.assertEqual(str(app.band_projection_style.cget("state")), tk.DISABLED)
            self.assertEqual(str(app.band_complete_radio.cget("style")), "Panel.TRadiobutton")
        finally:
            app.destroy()

    def test_band_rgb_with_four_channels_requires_explicit_style_change(self):
        app = LyraApp()
        try:
            app._add_projection_channel("band")
            for row, element in zip(
                app.band_projection_rows,
                ("Zr", "Sn", "Ge", "V"),
            ):
                row[0].set(True)
                row[1].set(element)
            app.band_data_type_var.set("projected")
            app.band_projection_style_var.set("RGB")

            with patch.object(app, "_start_job") as start_job, patch(
                "lyra.app.messagebox.showerror"
            ) as showerror:
                app._run_band()

            start_job.assert_not_called()
            self.assertEqual(app.band_projection_style_var.get(), "RGB")
            self.assertIn("最多 3", showerror.call_args.args[1])
        finally:
            app.destroy()

    def test_procar_enables_element_then_orbital_projection_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            app = LyraApp()
            try:
                app.band_projection_rows[0][1].set("Ta")
                app.band_folder_var.set(str(folder))
                app._choose_band_folder(False)

                self.assertEqual(str(app.band_projected_radio.cget("state")), tk.NORMAL)
                self.assertEqual(app.band_projection_rows[0][1].get(), "X")
                self.assertEqual(
                    tuple(app.band_projection_rows[0][4].cget("values")),
                    ("全部元素", "X"),
                )
                self.assertEqual(str(app.band_projection_style.cget("state")), tk.DISABLED)
                self.assertEqual(str(app.band_projection_rows[0][4].cget("state")), tk.DISABLED)

                app.band_data_type_var.set("projected")
                app._refresh_band_projection_state()
                app.band_projection_style_var.set("Fat band")
                app.band_projection_rows[0][2].set("dxy")

                self.assertEqual(app._band_render_mode(), "fat")
                self.assertEqual(str(app.band_projection_style.cget("state")), "readonly")
                self.assertEqual(str(app.band_projection_rows[0][4].cget("state")), "readonly")
                self.assertEqual(app._channels(app.band_projection_rows)[0].orbital, "dxy")
            finally:
                app.destroy()

    def test_settings_round_trip_restores_valid_folders_and_dynamic_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "能带"
            dos_folder = root / "态密度"
            band_folder.mkdir()
            dos_folder.mkdir()
            write_band_fixture(band_folder)
            write_procar(band_folder)
            write_dos_fixture(dos_folder)
            store = SettingsStore(root / "config" / "settings.json")

            first = LyraApp(settings_store=store)
            try:
                first.band_folder_var.set(str(band_folder))
                first._choose_band_folder(False)
                first.band_data_type_var.set("projected")
                first.band_projection_style_var.set("Fat band")
                first._refresh_band_projection_state()
                first._add_projection_channel("band")
                for index, row in enumerate(first.band_projection_rows):
                    row[0].set(index != 2)
                    row[1].set("X")
                    row[2].set(("s", "p", "d", "f")[index])
                    row[3].set(("#111111", "#224466", "#3974d8", "#d94841")[index])
                    row[7].configure(
                        bg=row[3].get(),
                        activebackground=row[3].get(),
                    )
                first.dos_folder_var.set(str(dos_folder))
                first._choose_dos_folder(False)
                first.dos_projection_enabled.set(True)
                first._set_projection_rows(
                    first.dos_projection_rows,
                    True,
                    ("X",),
                )
                first.band_min_var.set("-3.25")
                first.band_max_var.set("4.75")
                first.band_complete_colour_var.set("#123456")
                first.combined_complete_colour_var.set("#234567")
                first.language_var.set("en_US")
                first.dos_density_min_var.set("0")
                first.dos_density_max_var.set("35")
                first.output_dir_var.set(str(root / "论文图片"))
                first.dpi_var.set("450")
                first.width_var.set("165")
                first.format_vars["pdf"].set(False)
                first._save_settings()
            finally:
                first.destroy()

            restored = LyraApp(settings_store=store)
            try:
                self.assertEqual(restored.band_folder_var.get(), str(band_folder))
                self.assertTrue(restored.band_projection_available)
                self.assertEqual(restored.band_data_type_var.get(), "projected")
                self.assertEqual(restored.band_projection_style_var.get(), "Fat band")
                self.assertEqual(len(restored.band_projection_rows), 4)
                self.assertEqual(
                    [
                        (
                            row[0].get(),
                            row[1].get(),
                            row[2].get(),
                            row[3].get(),
                        )
                        for row in restored.band_projection_rows
                    ],
                    [
                        (True, "X", "s", "#111111"),
                        (True, "X", "p", "#224466"),
                        (False, "X", "d", "#3974d8"),
                        (True, "X", "f", "#d94841"),
                    ],
                )
                self.assertEqual(restored.dos_folder_var.get(), str(dos_folder))
                self.assertTrue(restored.dos_projection_enabled.get())
                self.assertEqual(restored.band_min_var.get(), "-3.25")
                self.assertEqual(restored.band_max_var.get(), "4.75")
                self.assertEqual(restored.band_complete_colour_var.get(), "#123456")
                self.assertEqual(restored.combined_complete_colour_var.get(), "#234567")
                self.assertEqual(restored.language_var.get(), "en_US")
                self.assertEqual(restored.nav_buttons["band"].cget("text"), "Band")
                self.assertEqual(restored.dos_density_max_var.get(), "35")
                self.assertEqual(restored.output_dir_var.get(), str(root / "论文图片"))
                self.assertEqual(restored.dpi_var.get(), "450")
                self.assertEqual(restored.width_var.get(), "165")
                self.assertFalse(restored.format_vars["pdf"].get())
            finally:
                restored.destroy()

    def test_invalid_saved_complete_band_colour_falls_back_to_black(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "settings.json")
            store.save({"band": {"complete_band_colour": "not-a-colour"}})

            app = LyraApp(settings_store=store)
            try:
                self.assertEqual(app.band_complete_colour_var.get(), "#000000")
                self.assertEqual(app.band_complete_colour_swatch.cget("bg"), "#000000")
            finally:
                app.destroy()

    def test_settings_restore_keeps_missing_folder_text_without_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "已经移动的计算")
            store = SettingsStore(Path(tmp) / "settings.json")
            store.save(
                {
                    "global": {"read_mode": "lite"},
                    "band": {
                        "folder": missing,
                        "data_type": "projected",
                    },
                    "dos": {
                        "folder": missing,
                        "projected": True,
                    },
                    "combined": {
                        "band_folder": missing,
                        "dos_folder": missing,
                        "band_projected": True,
                        "dos_projected": True,
                    },
                }
            )

            with patch("lyra.app.inspect_band_folder") as inspect:
                app = LyraApp(settings_store=store)
            try:
                self.assertEqual(app.band_folder_var.get(), missing)
                inspect.assert_not_called()
                self.assertEqual(app.band_data_type_var.get(), "complete")
                app._save_settings()
                saved = store.load()
                self.assertEqual(saved["band"]["data_type"], "projected")
                self.assertTrue(saved["dos"]["projected"])
                self.assertTrue(saved["combined"]["band_projected"])
                self.assertTrue(saved["combined"]["dos_projected"])
            finally:
                app.destroy()

    def test_malformed_string_booleans_do_not_enable_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "settings.json")
            store.save(
                {
                    "dos": {"projected": "false"},
                    "combined": {
                        "band_projected": "false",
                        "dos_projected": "true",
                    },
                }
            )

            app = LyraApp(settings_store=store)
            try:
                snapshot = app._settings_snapshot()
                self.assertFalse(snapshot["dos"]["projected"])
                self.assertFalse(snapshot["combined"]["band_projected"])
                self.assertFalse(snapshot["combined"]["dos_projected"])
            finally:
                app.destroy()

    def test_normal_close_saves_settings_once(self):
        store = Mock()
        store.load.return_value = {}
        store.load_namespace.return_value = {}
        app = LyraApp(settings_store=store)

        app._on_close()

        store.load_namespace.assert_called_once_with("classic")
        store.save_namespace.assert_called_once()
        self.assertEqual(store.save_namespace.call_args.args[0], "classic")
        self.assertIn("band", store.save_namespace.call_args.args[1])
        store.load.assert_not_called()
        store.save.assert_not_called()

    def test_close_warns_when_settings_cannot_be_saved(self):
        store = Mock()
        store.load.return_value = {}
        store.load_namespace.return_value = {}
        store.save_namespace.side_effect = OSError("disk full")
        app = LyraApp(settings_store=store)

        with patch("lyra.app.messagebox.showwarning") as warning:
            app._on_close()

        warning.assert_called_once()
        self.assertIn("disk full", warning.call_args.args[1])

    def test_missing_procar_keeps_complete_band_and_explains_projection_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._choose_band_folder(False)

                self.assertEqual(app.band_data_type_var.get(), "complete")
                self.assertEqual(str(app.band_complete_radio.cget("state")), tk.NORMAL)
                self.assertEqual(str(app.band_projected_radio.cget("state")), tk.DISABLED)
                self.assertIn("PROCAR", app.band_projection_reason_var.get())
                self.assertEqual(str(app.band_run_button.cget("state")), tk.NORMAL)
            finally:
                app.destroy()

    def test_worker_progress_updates_status_without_terminal_side_effects(self):
        app = self.app_with_worker_message(
            (
                "progress",
                {
                    "phase": "cache-hit",
                    "current": 0,
                    "total": 0,
                    "detail": "Loading normalized Band cache.",
                },
            )
        )
        try:
            with (
                patch.object(app, "_show_result") as show_result,
                patch.object(app, "_mode_changed") as mode_changed,
                patch.object(app, "_schedule_poll") as schedule_poll,
                patch("lyra.app.messagebox.showerror") as showerror,
            ):
                app._poll_worker()

            self.assertEqual(app.task_status_var.get(), "读取文件夹缓存 | Loading normalized Band cache.")
            show_result.assert_not_called()
            mode_changed.assert_not_called()
            showerror.assert_not_called()
            schedule_poll.assert_called_once_with()
        finally:
            app.destroy()

    def test_progress_status_shows_real_counter_and_elapsed_without_percentage(self):
        known_total = format_progress_status(
            {
                "phase": "parse-projections",
                "current": 43344,
                "total": 309600,
                "elapsed_seconds": 12.4,
            }
        )
        unknown_total = format_progress_status(
            {
                "phase": "parse-projections",
                "current": 126,
                "total": 0,
                "elapsed_seconds": 0.8,
            }
        )

        self.assertIn("解析投影记录", known_total)
        self.assertIn("43,344 / 309,600", known_total)
        self.assertIn("耗时 12.4 秒", known_total)
        self.assertNotIn("%", known_total)
        self.assertIn("126", unknown_total)
        self.assertNotIn("%", unknown_total)
        self.assertNotIn(" / ", unknown_total)

    def test_progress_status_uses_actual_procar_phase_and_detail(self):
        status = format_progress_status(
            {
                "phase": "parse",
                "current": 18,
                "total": 18,
                "detail": "Parsed 18/18 projection tables (k-point 9/9, band 2/2).",
                "elapsed_seconds": 1.2,
            }
        )

        self.assertIn("解析 PROCAR 投影表", status)
        self.assertIn("k-point 9/9", status)
        self.assertNotIn("%", status)

    def test_cache_write_failure_has_honest_chinese_status(self):
        self.assertEqual(
            cache_status_text({"phase": "cache-write-failed"}),
            "缓存：写入失败，仅本次内存可用",
        )

    def test_combined_rebuild_buttons_target_their_own_namespace(self):
        app = LyraApp()
        try:
            with patch.object(app, "_run_combined") as run_combined:
                app.combined_band_rebuild_cache_button.invoke()
                run_combined.assert_called_once_with(rebuild=True, rebuild_data_kind="band")
                run_combined.reset_mock()
                app.combined_dos_rebuild_cache_button.invoke()
                run_combined.assert_called_once_with(rebuild=True, rebuild_data_kind="dos")
        finally:
            app.destroy()

    def test_preview_does_not_require_formal_export_formats_or_dpi(self):
        app = LyraApp()
        try:
            for variable in app.format_vars.values():
                variable.set(False)
            app.dpi_var.set("not-a-dpi")
            with patch.object(app, "_start_job") as start_job, patch(
                "lyra.app.messagebox.showerror"
            ) as showerror:
                app._run_band()

            start_job.assert_called_once()
            showerror.assert_not_called()
        finally:
            app.destroy()

    def test_rendered_band_zoom_opens_independent_local_zoom_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            preview = folder / "preview.png"

            Image.new("RGB", (800, 500), "white").save(preview)
            identity = ProjectionDataService().band_identity(folder, "lite", ())
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._choose_band_folder(False)
                preview_settings = BandPlotSettings(
                    PlotRange(-2, 2),
                    mode="standard",
                )
                app._active_job = {
                    "intent": "preview",
                    "page": "band",
                    "settings": preview_settings,
                }
                app._show_result(
                    GeneratedResult(
                        (preview,),
                        preview,
                        "preview",
                        ((folder.resolve(), identity),),
                    )
                )

                self.assertEqual(str(app.preview_label.cget("cursor")), "hand2")
                self.assertTrue(app.preview_label.bind("<Double-Button-1>"))
                self.assertEqual(
                    str(app.zoom_preview_button.cget("state")),
                    tk.NORMAL,
                )
                self.assertEqual(app.zoom_preview_button.cget("text"), "局部放大图")
                self.assertTrue(app.zoom_preview_button.cget("command"))
                with patch.object(app, "_start_job") as start_job:
                    app._open_preview_viewer()

                self.assertEqual(
                    start_job.call_args.args[0],
                    "prepare-band-zoom",
                )
                self.assertEqual(
                    start_job.call_args.args[1]["identity"],
                    identity,
                )
                self.assertIs(
                    start_job.call_args.args[1]["settings"],
                    preview_settings,
                )
            finally:
                app.destroy()

    def test_rendered_combined_zoom_prepares_both_data_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            band_folder = root / "band"
            dos_folder = root / "dos"
            band_folder.mkdir()
            dos_folder.mkdir()
            preview = root / "preview.png"
            Image.new("RGB", (800, 500), "white").save(preview)
            band_identity = CacheIdentity(1, "test", "lite", "band", (CacheSource("EIGENVAL", 1, 1),))
            dos_identity = CacheIdentity(1, "test", "lite", "dos", (CacheSource("DOSCAR", 1, 1),))
            settings = CombinedPlotSettings(PlotRange(-2, 2))
            app = LyraApp()
            try:
                app.combined_band_folder_var.set(str(band_folder))
                app.combined_dos_folder_var.set(str(dos_folder))
                app.update_idletasks()
                app._active_job = {"intent": "preview", "page": "combined", "settings": settings}
                with patch.object(ProjectionDataService, "identity_matches", return_value=True):
                    app._show_result(
                        GeneratedResult(
                            (preview,), preview, "preview",
                            ((band_folder.resolve(), band_identity), (dos_folder.resolve(), dos_identity)),
                        )
                    )
                    with patch.object(app, "_start_job") as start_job:
                        app._open_preview_viewer()

                self.assertEqual(start_job.call_args.args[0], "prepare-combined-zoom")
                payload = start_job.call_args.args[1]
                self.assertEqual(payload["band_identity"], band_identity)
                self.assertEqual(payload["dos_identity"], dos_identity)
                self.assertIs(payload["settings"], settings)
            finally:
                app.destroy()

    def test_local_zoom_button_keeps_selected_language_after_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            preview = folder / "preview.png"
            Image.new("RGB", (120, 80), "white").save(preview)
            app = LyraApp()
            try:
                app.language_var.set("en_US")
                app._apply_language()
                app._active_job = {
                    "intent": "preview",
                    "page": "band",
                    "settings": BandPlotSettings(PlotRange(-2, 2)),
                }
                app._show_result(
                    GeneratedResult((preview,), preview, "preview")
                )

                self.assertEqual(
                    app.zoom_preview_button.cget("text"),
                    "Local Zoom Plot",
                )
            finally:
                app.destroy()

    def test_rapid_band_state_selection_queues_latest_without_busy_dialog(self):
        app = LyraApp()
        try:
            app._band_inspector_identity = Mock()
            app._band_detail_running = True
            selection = BandStateSelection("total", 4, 7)

            with (
                patch.object(app, "_launch_band_state_request") as launch,
                patch("lyra.app.messagebox.showinfo") as showinfo,
            ):
                app._request_band_state(selection, False, (7,))

            launch.assert_not_called()
            showinfo.assert_not_called()
            self.assertEqual(
                app._pending_band_state_request,
                (selection, False, (7,)),
            )
            self.assertIn("最后一次选择", app.task_status_var.get())
        finally:
            app.destroy()

    def test_completed_band_state_query_starts_latest_queued_selection(self):
        app = LyraApp()
        try:
            previous = (BandStateSelection("total", 1, 2), False, (2,))
            latest = (BandStateSelection("total", 5, 8), True, (8, 9))
            detail = Mock()
            inspector = Mock()
            inspector.winfo_exists.return_value = True
            app._band_inspector = inspector
            app._band_inspector_identity = Mock()
            app._band_detail_running = True
            app._pending_band_state_request = latest
            app._band_detail_queue.put(
                (app._band_detail_generation, previous, detail, None)
            )

            with (
                patch.object(app, "_launch_band_state_request") as launch,
            ):
                app._poll_band_detail_results()

            inspector.set_detail.assert_not_called()
            launch.assert_called_once_with(latest)
            self.assertIsNone(app._pending_band_state_request)
        finally:
            app.destroy()

    def test_discarded_preview_is_no_longer_clickable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            preview = folder / "preview.png"

            Image.new("RGB", (800, 500), "white").save(preview)
            write_procar(folder)
            identity = ProjectionDataService().band_identity(folder, "lite", ())
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._choose_band_folder(False)
                app._active_job = {"intent": "preview", "page": "band"}
                app._show_result(
                    GeneratedResult(
                        (preview,),
                        preview,
                        "preview",
                        ((folder.resolve(), identity),),
                    )
                )

                self.assertTrue(app.preview_label.bind("<Double-Button-1>"))
                app._discard_preview_state()

                self.assertEqual(str(app.preview_label.cget("cursor")), "")
                self.assertFalse(app.preview_label.bind("<Double-Button-1>"))
            finally:
                app.destroy()

    def test_external_source_change_expires_preview_before_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            identity = ProjectionDataService().band_identity(folder, "lite", ())
            preview = folder / ".lyra-cache" / "previews" / "lyra_preview_band.png"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"not-an-image")
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._active_job = {"intent": "preview", "page": "band"}
                app._show_result(
                    GeneratedResult(
                        (preview,),
                        preview,
                        "preview",
                        ((folder.resolve(), identity),),
                    )
                )
                self.assertEqual(str(app.save_image_button.cget("state")), tk.NORMAL)

                with (folder / "EIGENVAL").open("a", encoding="utf-8") as handle:
                    handle.write("changed")
                app._refresh_save_image_state()

                self.assertEqual(str(app.save_image_button.cget("state")), tk.DISABLED)
                self.assertIn("已过期", app.task_status_var.get())
                with patch.object(app, "_run_band") as run_band:
                    app._save_image()
                run_band.assert_not_called()
            finally:
                app.destroy()

    def test_preview_controls_follow_cache_evidence_and_stale_inputs(self):
        app = LyraApp()
        try:
            self.assertEqual(app.band_run_button.cget("text"), "预览")
            self.assertEqual(str(app.save_image_button.cget("state")), tk.DISABLED)
            self.assertIn("待验证", app.band_cache_status_var.get())

            app._apply_progress(
                {
                    "phase": "cache-hit",
                    "data_kind": "band",
                    "elapsed_seconds": 0.4,
                }
            )
            self.assertIn("已命中", app.band_cache_status_var.get())

            app._apply_progress({"phase": "cache-not-used", "data_kind": "band"})
            self.assertIn("本次不使用", app.band_cache_status_var.get())

            app._active_preview = ("band", app._plot_signature("band"))
            app.save_image_button.configure(state="normal")
            app.band_min_var.set("-4")

            self.assertEqual(str(app.save_image_button.cget("state")), tk.DISABLED)
            self.assertIn("已过期", app.task_status_var.get())

            app._set_job_controls("disabled")
            self.assertEqual(str(app.band_clear_cache_button.cget("state")), tk.DISABLED)
            self.assertEqual(str(app.band_projected_radio.cget("state")), tk.DISABLED)
        finally:
            app.destroy()

    def test_job_controls_restore_capability_state_after_worker_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            write_procar(folder)
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._choose_band_folder(False)
                app._set_job_controls("disabled")
                app._set_job_controls("normal")

                self.assertEqual(str(app.band_complete_radio.cget("state")), tk.NORMAL)
                self.assertEqual(str(app.band_projected_radio.cget("state")), tk.NORMAL)
                self.assertEqual(str(app.band_clear_cache_button.cget("state")), tk.NORMAL)
                self.assertEqual(str(app._job_controls[0].cget("state")), tk.NORMAL)
                self.assertEqual(str(app.nav_buttons["band"].cget("state")), tk.NORMAL)
            finally:
                app.destroy()

    def test_preview_result_enables_save_without_listing_cache_artifact_as_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_band_fixture(folder)
            preview = folder / ".lyra-cache" / "previews" / "lyra_band_preview.png"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"not-an-image")
            app = LyraApp()
            try:
                app.band_folder_var.set(str(folder))
                app._active_job = {"intent": "preview", "page": "band"}
                app._show_result(
                    GeneratedResult(
                        (preview,),
                        preview,
                        "preview",
                        ((folder.resolve(), ProjectionDataService().band_identity(folder, "lite", ())),),
                    )
                )

                self.assertEqual(str(app.save_image_button.cget("state")), tk.NORMAL)
                self.assertEqual(app.output_list.size(), 0)
                self.assertEqual(app.last_result.preview_file, preview)
            finally:
                app.destroy()

    def test_cache_cleared_terminal_message_does_not_show_plot_result(self):
        payload = {
            "folder": "C:/calculation",
            "message": "Projection cache cleared.",
        }
        app = self.app_with_worker_message(("cache-cleared", payload))
        try:
            with (
                patch.object(app, "_show_result") as show_result,
                patch.object(app, "_mode_changed") as mode_changed,
                patch.object(app, "_schedule_poll") as schedule_poll,
                patch("lyra.app.messagebox.showerror") as showerror,
            ):
                app._poll_worker()

            self.assertEqual(app.task_status_var.get(), payload["message"])
            show_result.assert_not_called()
            mode_changed.assert_not_called()
            showerror.assert_not_called()
            schedule_poll.assert_called_once_with()
        finally:
            app.destroy()

    def test_cache_clear_discards_deleted_preview_state_and_open_target(self):
        payload = {
            "folder": "C:/calculation",
            "message": "Projection cache cleared.",
        }
        app = self.app_with_worker_message(("cache-cleared", payload))
        try:
            preview = Path("C:/calculation/.lyra-cache/previews/lyra_preview_band.png")
            app.last_result = GeneratedResult((preview,), preview, "preview")
            app._active_preview = ("band", app._plot_signature("band"))
            app._preview_sources = (
                (
                    Path("C:/calculation"),
                    CacheIdentity(
                        1,
                        "test",
                        "lite",
                        "band",
                        (CacheSource("EIGENVAL", 1, 1),),
                    ),
                ),
            )
            app._active_job = {"data_kind": "band", "folder": payload["folder"]}
            app.output_list.insert(tk.END, str(preview))
            app.open_plot_button.configure(state="normal")
            app.preview_label.configure(text=str(preview))

            with patch.object(app, "_schedule_poll") as schedule_poll:
                app._poll_worker()

            self.assertIsNone(app.last_result)
            self.assertIsNone(app._active_preview)
            self.assertEqual(app._preview_sources, ())
            self.assertEqual(app.output_list.size(), 0)
            self.assertEqual(str(app.open_plot_button.cget("state")), tk.DISABLED)
            self.assertEqual(app.preview_label.cget("text"), "生成图片后在此预览")
            schedule_poll.assert_called_once_with()
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()

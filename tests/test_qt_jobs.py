from __future__ import annotations

from pathlib import Path

from lyra.cache import CacheIdentity, CacheSource
from lyra.models import GeneratedResult
from lyra.qt_controller import LyraController
from lyra.settings_store import SettingsStore


class RecordingWorker:
    def __init__(self) -> None:
        self.running = False
        self.started = []
        self.messages = []

    def start(self, operation, payload):
        self.running = True
        self.started.append((operation, payload))

    def poll(self):
        if not self.messages:
            return None
        message = self.messages.pop(0)
        if message[0] != "progress":
            self.running = False
        return message

    def close(self):
        self.running = False


def controller(tmp_path):
    worker = RecordingWorker()
    return LyraController(SettingsStore(tmp_path / "settings.json"), worker), worker


def test_band_preview_builds_existing_worker_contract(tmp_path):
    gui, worker = controller(tmp_path)
    gui.setValue("bandFolder", str(tmp_path / "band"))
    gui.setValue("bandEnergyMin", "-3")
    gui.setValue("bandEnergyMax", "4")

    gui.previewCurrent()

    operation, payload = worker.started[-1]
    assert operation == "preview-band"
    assert payload["mode"] == "lite"
    assert payload["folder"] == str(tmp_path / "band")
    assert payload["settings"].energy_range.minimum == -3.0
    assert payload["settings"].energy_range.maximum == 4.0
    assert payload["settings"].mode == "standard"


def test_combined_payload_keeps_separate_modes_and_dos_reference(tmp_path):
    gui, worker = controller(tmp_path)
    gui.setValue("currentPage", "combined")
    gui.setValue("combinedBandFolder", str(tmp_path / "band"))
    gui.setValue("combinedDosFolder", str(tmp_path / "dos"))
    gui.setValue("combinedBandReadMode", "full")
    gui.setValue("combinedDosReadMode", "lite")

    gui.previewCurrent()

    operation, payload = worker.started[-1]
    assert operation == "preview-combined"
    assert payload["band_mode"] == "full"
    assert payload["dos_mode"] == "lite"
    assert payload["settings"].reference_energy is None


def test_preview_completion_sets_preview_without_claiming_export(tmp_path):
    gui, worker = controller(tmp_path)
    source = tmp_path / "band"
    preview = source / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"png")
    identity = CacheIdentity(
        1,
        "test",
        "lite",
        "band",
        (CacheSource("EIGENVAL", 1, 1),),
    )
    gui.setValue("bandFolder", str(source))
    gui.previewCurrent()
    worker.messages.append(
        (
            "success",
            GeneratedResult(
                (preview,),
                preview,
                "done",
                source_identities=((source, identity),),
            ),
        )
    )

    gui.pollWorker()

    assert gui.previewUrl.startswith("file:")
    assert gui.canSave is True
    assert gui.outputFiles == []
    assert gui.statusText == "预览完成"


def test_export_reuses_preview_identity_and_lists_formal_outputs(tmp_path):
    gui, worker = controller(tmp_path)
    source = tmp_path / "band"
    output = tmp_path / "output"
    preview = source / ".lyra-cache" / "previews" / "lyra_preview_band.png"
    formal = output / "lyra_band.png"
    identity = CacheIdentity(1, "test", "lite", "band", (CacheSource("EIGENVAL", 1, 1),))
    gui.setValue("bandFolder", str(source))
    gui.setValue("outputFolder", str(output))
    gui.previewCurrent()
    worker.messages.append(("success", GeneratedResult((preview,), preview, "done", ((source, identity),))))
    gui.pollWorker()

    gui.saveCurrent()
    operation, payload = worker.started[-1]
    assert operation == "export-band"
    assert payload["preview_sources"] == ((source, identity),)

    worker.messages.append(("success", GeneratedResult((formal,), formal, "saved")))
    gui.pollWorker()
    assert gui.outputFiles == [str(formal)]
    assert gui.statusText == "图片已保存"

from __future__ import annotations

from lyra.qt_controller import LyraController
from lyra.qt_models import ProjectionChannelModel
from lyra.settings_store import SettingsStore


class RecordingWorker:
    running = False

    def __init__(self):
        self.started = []

    def start(self, operation, payload):
        self.started.append((operation, payload))

    def poll(self):
        return None

    def close(self):
        pass


def test_rgb_compatibility_turns_false_above_three_enabled_channels():
    model = ProjectionChannelModel()
    for element in ("Ta", "Ga", "Pt", "Sn"):
        model.add(element=element, orbital="全部", colour="#3974d8")

    assert model.rgbCompatible is False


def test_all_elements_expands_to_independent_scientific_channels():
    model = ProjectionChannelModel()
    model.add(element="全部元素", orbital="d", colour="#e34a3f")

    channels = model.to_channels(("Ta", "Ga"))

    assert [(item.element, item.orbital) for item in channels] == [
        ("Ta", "d"),
        ("Ga", "d"),
    ]


def test_controller_builds_fat_band_when_more_than_three_channels_are_enabled(tmp_path):
    worker = RecordingWorker()
    gui = LyraController(SettingsStore(tmp_path / "settings.json"), worker)
    gui.setValue("bandFolder", str(tmp_path / "band"))
    gui.setBoolValue("bandProjected", True)
    for element in ("Ta", "Ga", "Pt", "Sn"):
        gui.bandProjectionModel.add(element=element, orbital="d", colour="#3974d8")

    gui.previewCurrent()

    _operation, payload = worker.started[-1]
    assert payload["settings"].mode == "fat"
    assert len(payload["settings"].projection_channels) == 4
    assert "Fat band" in gui.statusText


def test_projection_rows_round_trip_through_settings(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    first = LyraController(store, RecordingWorker())
    first.bandProjectionModel.add(element="Ta", orbital="d", colour="#123456")
    first.setBoolValue("bandProjected", True)
    first.save()

    second = LyraController(store, RecordingWorker())

    assert second.bandProjected is True
    assert second.bandProjectionModel.snapshot() == [
        {"enabled": True, "element": "Ta", "orbital": "d", "colour": "#123456"}
    ]

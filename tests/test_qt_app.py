from __future__ import annotations

import os
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication

from lyra.qt_app import create_engine, qml_path
from lyra.qt_controller import LyraController
from lyra.settings_store import SettingsStore


class IdleWorker:
    running = False

    def poll(self):
        return None

    def close(self):
        pass


def test_packaged_qml_exists():
    path = qml_path()

    assert path.is_file()
    assert path.name == "Main.qml"


def test_engine_loads_main_window_offscreen(tmp_path):
    application = QGuiApplication.instance() or QGuiApplication([])
    controller = LyraController(SettingsStore(tmp_path / "settings.json"), IdleWorker())
    controller.setBoolValue("bandProjected", True)
    controller.bandProjectionModel.add("Ta", "d", "#3974d8")
    engine = create_engine(controller)

    assert engine.rootObjects()
    window = engine.rootObjects()[0]
    assert window.title() == "Lyra"
    window.close()
    engine.deleteLater()
    application.processEvents()

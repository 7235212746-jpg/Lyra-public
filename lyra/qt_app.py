from __future__ import annotations

from multiprocessing import freeze_support
import os
from pathlib import Path
import sys

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from lyra.qt_controller import LyraController
from lyra.lifecycle import SingleInstanceGuard
from lyra.qt_lifecycle import ApplicationLifecycle
from lyra.qt_window_effects import apply_acrylic
from lyra.settings_store import SettingsStore
from lyra.workers import SpawnWorker


def qml_path() -> Path:
    return Path(__file__).resolve().parent / "qml" / "Main.qml"


def create_engine(controller: LyraController) -> QQmlApplicationEngine:
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("lyra", controller)
    engine._lyra_controller = controller
    engine.load(QUrl.fromLocalFile(str(qml_path())))
    return engine


def main() -> int:
    freeze_support()
    application = QGuiApplication.instance() or QGuiApplication(sys.argv)
    application.setApplicationName("Lyra")
    application.setOrganizationName("Lyra")

    guard = SingleInstanceGuard("Lyra-Qt-v0.1")
    if not guard.acquire():
        return 0

    worker = SpawnWorker()
    controller = LyraController(SettingsStore.default(), worker)
    lifecycle = ApplicationLifecycle(worker, controller, guard)
    application.aboutToQuit.connect(lifecycle.shutdown)
    engine = create_engine(controller)
    if not engine.rootObjects():
        lifecycle.shutdown()
        return 1

    window = engine.rootObjects()[0]
    if os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
        QTimer.singleShot(80, lambda: _apply_material_safely(window))
    return application.exec()


def _apply_material_safely(window) -> None:
    try:
        apply_acrylic(window)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

from __future__ import annotations

from queue import Queue

from lyra.data_service import ProjectionDataService
from lyra.qt_controller import LyraController
from lyra.settings_store import SettingsStore
from lyra.workers import _execute_job
from tests.fixtures import write_magnetic_outcar, write_magnetic_poscar


class RecordingWorker:
    running = False

    def __init__(self):
        self.started = []
        self.messages = []

    def start(self, operation, payload):
        self.started.append((operation, payload))
        self.running = True

    def poll(self):
        if not self.messages:
            return None
        self.running = False
        return self.messages.pop(0)

    def close(self):
        pass


def test_magnetic_source_requires_poscar_and_outcar(tmp_path):
    gui = LyraController(SettingsStore(tmp_path / "settings.json"), RecordingWorker())

    gui.setFolder("magnetic", str(tmp_path))
    assert gui.canExportMagnetic is False
    assert "POSCAR" in gui.magneticStatus

    (tmp_path / "POSCAR").write_text("placeholder", encoding="utf-8")
    (tmp_path / "OUTCAR").write_text("placeholder", encoding="utf-8")
    gui.setFolder("magnetic", str(tmp_path))
    assert gui.canExportMagnetic is True


def test_magnetic_export_uses_background_worker(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    write_magnetic_poscar(source)
    write_magnetic_outcar(source, scalar_blocks=((1.0, -1.0),))
    worker = RecordingWorker()
    gui = LyraController(SettingsStore(tmp_path / "settings.json"), worker)
    gui.setFolder("magnetic", str(source))
    gui.setFolder("output", str(output))

    gui.exportMagnetic()

    assert worker.started == [
        ("export-magnetic", {"folder": str(source.resolve()), "output": str(output.resolve())})
    ]
    assert gui.busy is True


def test_worker_writes_vesta_magnetic_pair_transactionally(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    write_magnetic_poscar(source)
    write_magnetic_outcar(source, scalar_blocks=((1.0, -1.0),))

    result = _execute_job(
        ProjectionDataService(),
        Queue(),
        "export-magnetic",
        {"folder": str(source), "output": str(output)},
    )

    assert result["output_files"] == (
        output / "vesta_magnetic_moments.mcif",
        output / "magnetic_moment_evidence.txt",
    )
    assert all(path.is_file() for path in result["output_files"])

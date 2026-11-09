from __future__ import annotations

from unittest.mock import Mock

from lyra.qt_controller import LyraController
from lyra.settings_store import SettingsStore
from tests.fixtures import write_band_fixture, write_dos_fixture


class IdleWorker:
    running = False

    def poll(self):
        return None

    def close(self) -> None:
        pass


def test_lite_band_inspection_exposes_detected_files_and_disables_projection(tmp_path):
    write_band_fixture(tmp_path)
    controller = LyraController(SettingsStore(tmp_path / "settings.json"), IdleWorker())

    controller.setFolder("band", str(tmp_path))

    assert controller.bandAvailable is True
    assert controller.bandProjectionAvailable is False
    assert "EIGENVAL" in controller.bandStatus
    assert "PROCAR" in controller.bandProjectionReason
    assert controller.bandElements == ["X"]


def test_lite_dos_inspection_enables_projected_dos_when_blocks_exist(tmp_path):
    write_dos_fixture(tmp_path, projected=True)
    controller = LyraController(SettingsStore(tmp_path / "settings.json"), IdleWorker())

    controller.setFolder("dos", str(tmp_path))

    assert controller.dosAvailable is True
    assert controller.dosProjectionAvailable is True
    assert controller.dosElements == ["X"]


def test_settings_round_trip_preserves_folders_ranges_and_page(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    first = LyraController(store, IdleWorker())
    first.setValue("currentPage", "dos")
    first.setValue("dosFolder", "C:/calc/dos")
    first.setValue("dosEnergyMin", "-7")
    first.setValue("dosDensityMax", "42")
    first.save()

    second = LyraController(store, IdleWorker())

    assert second.currentPage == "dos"
    assert second.dosFolder == "C:/calc/dos"
    assert second.dosEnergyMin == "-7"
    assert second.dosDensityMax == "42"


def test_qt_controller_uses_only_qt_namespace():
    store = Mock()
    store.load.return_value = {}
    store.load_namespace.return_value = {}
    controller = LyraController(store, IdleWorker())

    controller.save()

    store.load_namespace.assert_called_once_with("qt")
    store.save_namespace.assert_called_once()
    assert store.save_namespace.call_args.args[0] == "qt"
    assert "band" in store.save_namespace.call_args.args[1]
    store.load.assert_not_called()
    store.save.assert_not_called()


def test_combined_availability_requires_both_explicit_folders(tmp_path):
    band = tmp_path / "band"
    dos = tmp_path / "dos"
    band.mkdir()
    dos.mkdir()
    write_band_fixture(band)
    write_dos_fixture(dos)
    controller = LyraController(SettingsStore(tmp_path / "settings.json"), IdleWorker())

    controller.setFolder("combinedBand", str(band))
    assert controller.combinedAvailable is False
    controller.setFolder("combinedDos", str(dos))

    assert controller.combinedAvailable is True
    assert "DOS/SCF E-fermi" in controller.combinedReference

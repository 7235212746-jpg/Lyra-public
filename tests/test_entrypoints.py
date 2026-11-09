from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch
import tomllib

import pytest

import lyra.__main__ as default_entry
import lyra.legacy_main as legacy_entry
from lyra.lifecycle import SingleInstanceGuard, run_classic
from lyra.qt_lifecycle import SingleInstanceGuard as QtCompatibleGuard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_module_uses_classic_entrypoint():
    with patch.object(default_entry, "run_classic", return_value=0) as classic:
        assert default_entry.main() == 0

    classic.assert_called_once_with()


def test_legacy_module_is_an_alias_for_the_classic_entrypoint():
    with patch.object(legacy_entry, "run_classic", return_value=0) as classic:
        assert legacy_entry.main() == 0

    classic.assert_called_once_with()


def test_classic_runner_owns_and_releases_the_single_instance_guard():
    guard = Mock()
    guard.acquire.return_value = True
    app = Mock()
    settings = Mock()

    with (
        patch("lyra.lifecycle.freeze_support") as freeze,
        patch("lyra.lifecycle.SingleInstanceGuard", return_value=guard),
        patch("lyra.lifecycle._create_classic_app", return_value=app) as app_type,
    ):
        assert run_classic(settings_store=settings) == 0

    freeze.assert_called_once_with()
    app_type.assert_called_once_with(settings_store=settings)
    app.mainloop.assert_called_once_with()
    guard.release.assert_called_once_with()


def test_classic_runner_does_not_construct_a_second_window():
    guard = Mock()
    guard.acquire.return_value = False

    with (
        patch("lyra.lifecycle.freeze_support"),
        patch("lyra.lifecycle.SingleInstanceGuard", return_value=guard),
        patch("lyra.lifecycle._create_classic_app") as app_type,
    ):
        assert run_classic() == 0

    app_type.assert_not_called()
    guard.release.assert_not_called()


def test_classic_runner_releases_guard_when_app_construction_fails():
    guard = Mock()
    guard.acquire.return_value = True

    with (
        patch("lyra.lifecycle.freeze_support"),
        patch("lyra.lifecycle.SingleInstanceGuard", return_value=guard),
        patch(
            "lyra.lifecycle._create_classic_app",
            side_effect=RuntimeError("construction failed"),
        ),
        pytest.raises(RuntimeError, match="construction failed"),
    ):
        run_classic(settings_store=Mock())

    guard.release.assert_called_once_with()


def test_classic_runner_releases_guard_when_mainloop_fails():
    guard = Mock()
    guard.acquire.return_value = True
    app = Mock()
    app.mainloop.side_effect = RuntimeError("mainloop failed")

    with (
        patch("lyra.lifecycle.freeze_support"),
        patch("lyra.lifecycle.SingleInstanceGuard", return_value=guard),
        patch("lyra.lifecycle._create_classic_app", return_value=app),
        pytest.raises(RuntimeError, match="mainloop failed"),
    ):
        run_classic(settings_store=Mock())

    guard.release.assert_called_once_with()


def test_qt_lifecycle_reexports_the_neutral_guard():
    assert QtCompatibleGuard is SingleInstanceGuard


def test_qt_lifecycle_import_does_not_eagerly_load_the_classic_gui_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import lyra.qt_lifecycle; "
                "blocked = {'lyra.app', 'tkinter', 'PIL.ImageTk', 'matplotlib'}; "
                "loaded = sorted(blocked.intersection(sys.modules)); "
                "assert not loaded, 'eager imports: ' + repr(loaded)"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_console_scripts_keep_qt_explicit_and_classic_default():
    document = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = document["project"]["scripts"]

    assert scripts["lyra"] == "lyra.__main__:main"
    assert scripts["lyra-mobile"] == "lyra.mobile.__main__:main"
    assert scripts["lyra-legacy"] == "lyra.legacy_main:main"
    assert scripts["lyra-qt"] == "lyra.qt_app:main"
    assert document["project"]["version"] == "0.2.0"


def test_desktop_shortcut_prefers_the_live_source_runtime():
    script = (PROJECT_ROOT / "install_shortcut.ps1").read_text(encoding="utf-8")

    python_branch = script.index(
        "if (Test-Path -LiteralPath $Pythonw -PathType Leaf)"
    )
    executable_branch = script.index(
        "elseif (Test-Path -LiteralPath $Executable -PathType Leaf)"
    )

    assert python_branch < executable_branch
    assert '$Arguments = "-m lyra"' in script
    assert "lyra.legacy_main" not in script


def test_desktop_shortcut_uses_the_lyra_icon_asset():
    script = (PROJECT_ROOT / "install_shortcut.ps1").read_text(encoding="utf-8")

    assert 'Join-Path $Root "assets\\lyra_icon.ico"' in script
    assert '$Shortcut.IconLocation = "$IconPath,0"' in script


def test_package_and_exe_include_mobile_incar_and_winscp_resources():
    document = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = document["tool"]["setuptools"]["package-data"]["lyra"]
    build_script = (PROJECT_ROOT / "build_lyra.ps1").read_text(encoding="utf-8")

    assert "resources/incar/*" in package_data
    assert "resources/mobile/*" in package_data
    assert "remote/*.ps1" in package_data
    assert "lyra\\resources;lyra\\resources" in build_script
    assert "winscp_bridge.ps1;lyra\\remote" in build_script

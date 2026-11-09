from pathlib import Path
import threading
from types import SimpleNamespace

from lyra.remote.manifest import build_calculation_manifest
from lyra.remote.credentials import RemoteCredentials
from lyra.remote.profiles import RuntimeAuthentication
from lyra.remote.transport import RemoteStat
from lyra.remote_window import (
    available_workflows,
    breadcrumb_paths,
    cache_directory,
    credential_store_path,
    format_modified_time,
    format_task_progress,
    resolve_runtime_authentication,
    selected_sync_files,
    workflow_supports_preview,
    workflow_supports_remote_analysis,
    RemoteWorkspaceWindow,
)


def manifest():
    names = ("EIGENVAL", "KPOINTS", "POSCAR", "OUTCAR", "PROCAR", "WAVECAR")
    return build_calculation_manifest(
        "/work/calc",
        tuple(
            RemoteStat(name, f"/work/calc/{name}", False, 10, 1)
            for name in names
        ),
    )


def test_selected_sync_files_never_pulls_wavecar_for_standard_plotting() -> None:
    result = selected_sync_files(manifest(), "projected_band")

    assert {item.name for item in result} == {
        "EIGENVAL",
        "KPOINTS",
        "POSCAR",
        "OUTCAR",
        "PROCAR",
    }
    assert "WAVECAR" not in {item.name for item in result}


def test_cache_directory_is_profile_and_manifest_scoped(tmp_path: Path) -> None:
    target = cache_directory(tmp_path, "cluster-a", manifest())

    assert target.parent.name == "cluster-a"
    assert target.name == manifest().fingerprint
    assert target.is_relative_to(tmp_path.resolve())


def test_credential_store_path_is_profile_scoped_and_confined(tmp_path: Path) -> None:
    target = credential_store_path(tmp_path, "cluster-a")

    assert target == tmp_path.resolve() / "remote_credentials" / "cluster-a.json"
    assert target.is_relative_to(tmp_path.resolve())


def test_credential_store_path_rejects_unsafe_profile_ids(tmp_path: Path) -> None:
    for profile_id in ("../escape", "cluster/a", "cluster\\a", ""):
        try:
            credential_store_path(tmp_path, profile_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe profile ID accepted: {profile_id!r}")


def test_entered_credentials_override_saved_credentials() -> None:
    saved = RemoteCredentials(password="saved", private_key_passphrase="saved-key")

    result = resolve_runtime_authentication("entered", "", saved)

    assert result == RuntimeAuthentication("entered", None)


def test_saved_credentials_are_used_when_fields_are_empty() -> None:
    saved = RemoteCredentials(password="saved", private_key_passphrase="saved-key")

    result = resolve_runtime_authentication("", "", saved)

    assert result == RuntimeAuthentication("saved", "saved-key")


def test_available_workflows_follow_workspace_action_order() -> None:
    assert available_workflows(manifest()) == (
        "band",
        "projected_band",
        "magnetic_export",
    )


def test_format_modified_time_handles_unknown_and_real_timestamps() -> None:
    assert format_modified_time(0) == ""
    assert format_modified_time(1_700_000_000_000_000_000).startswith("2023-11-")


def test_format_task_progress_names_real_transfer_step() -> None:
    assert format_task_progress("PROCAR", 3, 5) == "正在同步 PROCAR（3/5）"


def test_only_plot_workflows_offer_immediate_preview() -> None:
    assert workflow_supports_preview("band") is True
    assert workflow_supports_preview("dos") is True
    assert workflow_supports_preview("fermi_surface") is True
    assert workflow_supports_preview("magnetic_export") is False


def test_first_remote_native_delivery_is_limited_to_supported_lite_workflows() -> None:
    assert workflow_supports_remote_analysis("band") is True
    assert workflow_supports_remote_analysis("projected_band") is True
    assert workflow_supports_remote_analysis("dos") is True
    assert workflow_supports_remote_analysis("fermi_surface") is False
    assert workflow_supports_remote_analysis("magnetic_export") is False


def test_breadcrumb_paths_collapse_long_locations_without_losing_targets() -> None:
    assert breadcrumb_paths("/work/group/project/soc/band/output", max_parts=4) == (
        ("/", "/"),
        ("…", "/work/group"),
        ("project", "/work/group/project"),
        ("soc", "/work/group/project/soc"),
        ("band", "/work/group/project/soc/band"),
        ("output", "/work/group/project/soc/band/output"),
    )


def test_close_only_waits_briefly_for_cancelled_background_work() -> None:
    class ThreadSpy:
        def __init__(self):
            self.timeouts = []

        def join(self, timeout=None):
            self.timeouts.append(timeout)

    worker = ThreadSpy()
    window = SimpleNamespace(
        _closing=False,
        _request_generation=0,
        _poll_id=None,
        session=None,
        transport=None,
        _threads_lock=threading.Lock(),
        _threads={worker},
        _save_workspace_preferences=lambda: None,
        destroy=lambda: None,
    )

    RemoteWorkspaceWindow.close(window)

    assert worker.timeouts == [0.2]
    assert window._closing is True

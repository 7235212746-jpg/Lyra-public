from __future__ import annotations

import subprocess
import sys

from lyra.qt_lifecycle import ApplicationLifecycle, SingleInstanceGuard


class RecordingWorker:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RecordingSettings:
    def __init__(self) -> None:
        self.save_calls = 0

    def save(self) -> None:
        self.save_calls += 1


class FailingSettings(RecordingSettings):
    def save(self) -> None:
        super().save()
        raise OSError("settings disk unavailable")


def test_second_single_instance_guard_is_rejected(tmp_path):
    first = SingleInstanceGuard("lyra-test", lock_root=tmp_path)
    second = SingleInstanceGuard("lyra-test", lock_root=tmp_path)

    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    assert second.acquire() is True
    second.release()


def test_shutdown_saves_settings_closes_worker_and_releases_guard_once(tmp_path):
    guard = SingleInstanceGuard("lyra-lifecycle", lock_root=tmp_path)
    assert guard.acquire() is True
    worker = RecordingWorker()
    settings = RecordingSettings()
    lifecycle = ApplicationLifecycle(worker, settings, guard)

    lifecycle.shutdown()
    lifecycle.shutdown()

    assert worker.close_calls == 1
    assert settings.save_calls == 1
    replacement = SingleInstanceGuard("lyra-lifecycle", lock_root=tmp_path)
    assert replacement.acquire() is True
    replacement.release()


def test_shutdown_records_save_failure_and_still_releases_resources(tmp_path):
    guard = SingleInstanceGuard("lyra-save-failure", lock_root=tmp_path)
    assert guard.acquire() is True
    worker = RecordingWorker()
    settings = FailingSettings()
    lifecycle = ApplicationLifecycle(worker, settings, guard)

    lifecycle.shutdown()
    lifecycle.shutdown()

    assert settings.save_calls == 1
    assert worker.close_calls == 1
    assert isinstance(lifecycle.settings_error, OSError)
    assert str(lifecycle.settings_error) == "settings disk unavailable"
    replacement = SingleInstanceGuard("lyra-save-failure", lock_root=tmp_path)
    assert replacement.acquire() is True
    replacement.release()


def test_live_guard_is_not_treated_as_stale_across_processes(tmp_path):
    script = (
        "import sys; "
        "from pathlib import Path; "
        "from lyra.qt_lifecycle import SingleInstanceGuard; "
        f"guard=SingleInstanceGuard('cross-process', lock_root=Path({str(tmp_path)!r})); "
        "assert guard.acquire(); print('ready', flush=True); sys.stdin.readline(); guard.release()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        contender = SingleInstanceGuard("cross-process", lock_root=tmp_path)
        assert contender.acquire() is False
    finally:
        child.communicate("\n", timeout=5)


def test_exited_process_lock_is_reclaimed_even_while_process_handle_is_retained(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    lock_path = tmp_path / "exited-process.lock"
    lock_path.write_text(str(child.pid), encoding="ascii")

    guard = SingleInstanceGuard("exited-process", lock_root=tmp_path)

    assert guard.acquire() is True
    guard.release()

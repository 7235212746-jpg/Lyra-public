from __future__ import annotations

import pytest

from lyra.mobile.tasks import InMemoryTaskBackend, MobileTask, TaskError, validate_task_request


def test_task_lifecycle_create_list_detail_progress_and_cancel() -> None:
    backend = InMemoryTaskBackend()
    task = backend.create("band", {"remote_profile_id": "lab", "directory_id": "calc-42"})
    backend.update(task.task_id, state="running", progress=0.4, stage="Parsing PROCAR")

    listed = backend.list_tasks()
    detail = backend.get(task.task_id)
    cancelled = backend.cancel(task.task_id)

    assert [item.task_id for item in listed] == [task.task_id]
    assert detail.progress == 0.4
    assert detail.stage == "Parsing PROCAR"
    assert cancelled.state == "cancelled"


@pytest.mark.parametrize("field", ["path", "folder", "password", "private_key", "linux_key", "secret"])
def test_task_request_rejects_paths_and_credentials(field: str) -> None:
    with pytest.raises(TaskError, match="not permitted"):
        validate_task_request({"kind": "band", "parameters": {field: "sensitive"}})


def test_public_task_payload_contains_no_backend_private_data() -> None:
    backend = InMemoryTaskBackend()
    task = backend.create("dos", {"directory_id": "calc-1"})

    payload = task.to_public_dict()

    assert payload["parameters"] == {"directory_id": "calc-1"}
    assert "path" not in repr(payload).lower()


def test_public_task_payload_filters_private_fields_from_plugin_backend() -> None:
    task = MobileTask(
        task_id="a" * 32,
        kind="band",
        parameters={
            "directory_id": "calc-1",
            "password": "do-not-leak",
            "options": {"private_key": "nested-secret"},
            "debug": "C:/private/debug.log",
        },
        resources=(
            {
                "id": "opaque-resource-id-123456",
                "label": "Plot",
                "previewable": True,
                "preview_url": "/api/resources/opaque-resource-id-123456/preview",
                "download_url": "/api/resources/opaque-resource-id-123456/download",
                "path": "C:/private/plot.png",
            },
        ),
    )

    serialized = task.to_public_dict()

    assert serialized["parameters"] == {"directory_id": "calc-1"}
    assert "do-not-leak" not in repr(serialized)
    assert "nested-secret" not in repr(serialized)
    assert "debug.log" not in repr(serialized)
    assert "C:/private" not in repr(serialized)
    assert serialized["resources"][0]["preview_url"] == "/api/resources/opaque-resource-id-123456/preview"
    assert set(serialized["resources"][0]) <= {
        "id", "label", "media_type", "previewable", "preview_url", "download_url"
    }

from __future__ import annotations

from pathlib import Path

import pytest

from lyra.mobile.resources import ResourceError, ResourceRegistry


def test_registry_uses_opaque_ids_and_never_serializes_real_paths(tmp_path: Path) -> None:
    image = tmp_path / "private" / "preview.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    registry = ResourceRegistry()

    public = registry.register(image, label="Band preview")

    assert str(tmp_path) not in repr(public)
    assert "/" not in public.resource_id
    assert "\\" not in public.resource_id
    assert registry.resolve(public.resource_id, purpose="preview").path == image.resolve()


def test_registry_rejects_traversal_unknown_ids_and_disallowed_files(tmp_path: Path) -> None:
    registry = ResourceRegistry()
    executable = tmp_path / "secret.exe"
    executable.write_bytes(b"MZ")

    with pytest.raises(ResourceError, match="not allowed"):
        registry.register(executable)
    with pytest.raises(ResourceError):
        registry.resolve("../../secret", purpose="download")
    with pytest.raises(ResourceError):
        registry.resolve("unknown", purpose="preview")


def test_svg_is_downloadable_but_not_inline_previewable(tmp_path: Path) -> None:
    svg = tmp_path / "plot.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    registry = ResourceRegistry()
    public = registry.register(svg)

    assert registry.resolve(public.resource_id, purpose="download").path == svg.resolve()
    with pytest.raises(ResourceError, match="preview"):
        registry.resolve(public.resource_id, purpose="preview")


def test_registered_resources_survive_service_restart(tmp_path: Path) -> None:
    state = tmp_path / "resources.json"
    image = tmp_path / "preview.png"
    image.write_bytes(b"png")
    first = ResourceRegistry(state_file=state)
    public = first.register(image, label="Band")

    second = ResourceRegistry(state_file=state)

    assert second.resolve(public.resource_id, purpose="preview").path == image.resolve()
    assert str(image) not in repr(second.resolve(public.resource_id, purpose="preview"))

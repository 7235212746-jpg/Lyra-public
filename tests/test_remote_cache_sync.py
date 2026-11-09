from __future__ import annotations

import hashlib

import pytest

from lyra.remote.cache_sync import SelectiveCacheSync
from lyra.remote.protocol import Artifact, RemoteResult
from tests.test_remote_support import FakeSession


def artifact(role, remote_path, relative_path, data):
    return Artifact(
        role,
        remote_path,
        relative_path,
        len(data),
        hashlib.sha256(data).hexdigest(),
        "application/octet-stream",
    )


def test_sync_downloads_only_selected_roles_and_preserves_layout(tmp_path):
    session = FakeSession()
    preview = b"png-data"
    cache = b"parsed-cache"
    session.files["/remote/preview.png"] = preview
    session.files["/remote/cache.bin"] = cache
    result = RemoteResult(
        __import__("uuid").uuid4(),
        True,
        {},
        (
            artifact("preview", "/remote/preview.png", "previews/plot.png", preview),
            artifact("parsed-cache", "/remote/cache.bin", "cache/data.bin", cache),
        ),
    )

    synced = SelectiveCacheSync(session).sync(result, tmp_path, roles={"preview"})

    assert synced == (tmp_path / "previews" / "plot.png",)
    assert (tmp_path / "previews" / "plot.png").read_bytes() == preview
    assert not (tmp_path / "cache" / "data.bin").exists()


def test_hash_mismatch_does_not_replace_existing_cache(tmp_path):
    session = FakeSession()
    destination = tmp_path / "cache.bin"
    destination.write_bytes(b"known-good")
    expected = b"expected"
    session.files["/remote/cache.bin"] = b"corrupt"
    result = RemoteResult(
        __import__("uuid").uuid4(),
        True,
        {},
        (artifact("parsed-cache", "/remote/cache.bin", "cache.bin", expected),),
    )

    with pytest.raises(ValueError, match="hash|size"):
        SelectiveCacheSync(session).sync(result, tmp_path, roles={"parsed-cache"})

    assert destination.read_bytes() == b"known-good"


def test_sync_rejects_traversal_and_total_size_over_limit(tmp_path):
    session = FakeSession()
    data = b"1234"
    with pytest.raises(ValueError, match="relative"):
        artifact("preview", "/remote/a", "../escape", data)
    good = artifact("preview", "/remote/a", "a", data)
    with pytest.raises(ValueError, match="limit"):
        SelectiveCacheSync(session).sync(
            RemoteResult(__import__("uuid").uuid4(), True, {}, (good,)),
            tmp_path,
            roles={"preview"},
            max_total_bytes=3,
        )

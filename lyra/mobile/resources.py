from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading


class ResourceError(ValueError):
    pass


DOWNLOAD_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".pdf": "application/pdf",
    ".svg": "image/svg+xml", ".csv": "text/csv", ".json": "application/json",
    ".txt": "text/plain",
}
PREVIEW_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
RESOURCE_ID = re.compile(r"^[A-Za-z0-9_-]{20,80}$")


@dataclass(frozen=True)
class PublicResource:
    resource_id: str
    label: str
    media_type: str
    previewable: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "id": self.resource_id,
            "label": self.label,
            "media_type": self.media_type,
            "previewable": self.previewable,
            "preview_url": f"/api/resources/{self.resource_id}/preview" if self.previewable else None,
            "download_url": f"/api/resources/{self.resource_id}/download",
        }


@dataclass(frozen=True, repr=False)
class RegisteredResource:
    public: PublicResource
    path: Path

    def __repr__(self) -> str:
        return f"RegisteredResource(id={self.public.resource_id!r})"


class ResourceRegistry:
    def __init__(self, *, state_file: Path | None = None) -> None:
        self._resources: dict[str, RegisteredResource] = {}
        self._lock = threading.RLock()
        self.state_file = Path(state_file).resolve() if state_file else None
        self._load()

    def register(self, path: Path, *, label: str | None = None) -> PublicResource:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ResourceError("resource must be a file")
        suffix = resolved.suffix.lower()
        media_type = DOWNLOAD_TYPES.get(suffix)
        if media_type is None:
            raise ResourceError("file type is not allowed for mobile download")
        resource_id = secrets.token_urlsafe(24)
        raw_label = " ".join((label or resolved.name).split())
        safe_label = raw_label.replace("\\", "/").rsplit("/", 1)[-1][:120] or "Lyra result"
        public = PublicResource(resource_id, safe_label, media_type, suffix in PREVIEW_TYPES)
        with self._lock:
            self._resources[resource_id] = RegisteredResource(public, resolved)
            self._save()
        return public

    def resolve(self, resource_id: str, *, purpose: str) -> RegisteredResource:
        if not isinstance(resource_id, str) or not RESOURCE_ID.fullmatch(resource_id):
            raise ResourceError("invalid resource identifier")
        with self._lock:
            resource = self._resources.get(resource_id)
        if resource is None or not resource.path.is_file():
            raise ResourceError("resource not found")
        if purpose == "preview" and not resource.public.previewable:
            raise ResourceError("resource is not allowed for inline preview")
        if purpose not in {"preview", "download"}:
            raise ResourceError("invalid resource purpose")
        return resource

    def discard(self, resource_id: str) -> None:
        with self._lock:
            if self._resources.pop(resource_id, None) is not None:
                self._save()

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            if self.state_file.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("resource state is too large")
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(payload.get("resources"), list):
                raise ValueError("unsupported resource state")
            for item in payload["resources"][:500]:
                resource_id = str(item["id"])
                if not RESOURCE_ID.fullmatch(resource_id):
                    raise ValueError("invalid resource identifier")
                path = Path(str(item["path"])).resolve(strict=True)
                suffix = path.suffix.lower()
                media_type = DOWNLOAD_TYPES.get(suffix)
                if not path.is_file() or media_type is None:
                    continue
                label = re.split(r"[\\/]", str(item["label"]))[-1][:120]
                public = PublicResource(
                    resource_id,
                    label or path.name,
                    media_type,
                    suffix in PREVIEW_TYPES,
                )
                self._resources[resource_id] = RegisteredResource(public, path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._resources.clear()

    def _save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        entries = list(self._resources.values())[-500:]
        payload = {
            "version": 1,
            "resources": [
                {
                    "id": entry.public.resource_id,
                    "label": entry.public.label,
                    "path": str(entry.path),
                }
                for entry in entries
            ],
        }
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_file)

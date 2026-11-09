from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .transport import validate_remote_path


_DEFAULT_PATH = "/home"
_DEFAULT_WORKFLOW = "band"
_WORKFLOWS = frozenset({"band", "projected_band", "dos", "fermi_surface"})
_MAX_BOOKMARKS = 32
_MAX_HISTORY = 128
_MAX_PROFILE_NAME = 128


def _safe_path(value: object, default: str = _DEFAULT_PATH) -> str:
    if not isinstance(value, str):
        return default
    try:
        return validate_remote_path(value.strip())
    except ValueError:
        return default


def _safe_profile_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if (
        not name
        or len(name) > _MAX_PROFILE_NAME
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        return ""
    return name


class RemoteNavigation:
    def __init__(self, initial_path: str = _DEFAULT_PATH) -> None:
        self._history = [_safe_path(initial_path)]
        self._index = 0

    @property
    def current(self) -> str:
        return self._history[self._index]

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    @property
    def can_back(self) -> bool:
        return self._index > 0

    @property
    def can_forward(self) -> bool:
        return self._index + 1 < len(self._history)

    def visit(self, path: str) -> str:
        normalized = validate_remote_path(path.strip())
        if normalized == self.current:
            return normalized
        del self._history[self._index + 1 :]
        self._history.append(normalized)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]
        self._index = len(self._history) - 1
        return normalized

    def back(self) -> str:
        if self.can_back:
            self._index -= 1
        return self.current

    def forward(self) -> str:
        if self.can_forward:
            self._index += 1
        return self.current


@dataclass(frozen=True)
class RemoteWorkspacePreferences:
    last_profile: str = ""
    last_path: str = _DEFAULT_PATH
    workflow: str = _DEFAULT_WORKFLOW
    bookmarks: tuple[str, ...] = ()
    editor_expanded: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> "RemoteWorkspacePreferences":
        data = value if isinstance(value, Mapping) else {}
        bookmarks: list[str] = []
        raw_bookmarks = data.get("bookmarks", ())
        if isinstance(raw_bookmarks, (list, tuple)):
            for item in raw_bookmarks:
                if len(bookmarks) >= _MAX_BOOKMARKS:
                    break
                if not isinstance(item, str):
                    continue
                try:
                    path = validate_remote_path(item.strip())
                except ValueError:
                    continue
                if path not in bookmarks:
                    bookmarks.append(path)
        workflow = data.get("workflow")
        return cls(
            last_profile=_safe_profile_name(data.get("last_profile")),
            last_path=_safe_path(data.get("last_path")),
            workflow=(
                workflow
                if isinstance(workflow, str) and workflow in _WORKFLOWS
                else _DEFAULT_WORKFLOW
            ),
            bookmarks=tuple(bookmarks),
            editor_expanded=data.get("editor_expanded") is True,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "last_profile": self.last_profile,
            "last_path": self.last_path,
            "workflow": self.workflow,
            "bookmarks": list(self.bookmarks),
            "editor_expanded": self.editor_expanded,
        }


def load_remote_workspace_preferences(settings_store) -> RemoteWorkspacePreferences:
    classic = settings_store.load_namespace("classic")
    data = classic.get("remote_workspace", {}) if isinstance(classic, Mapping) else {}
    return RemoteWorkspacePreferences.from_mapping(data)


def save_remote_workspace_preferences(
    settings_store,
    preferences: RemoteWorkspacePreferences,
) -> None:
    settings_store.save_namespace(
        "classic",
        {"remote_workspace": preferences.to_mapping()},
    )

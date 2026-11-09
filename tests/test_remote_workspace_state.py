from lyra.remote.workspace_state import (
    RemoteNavigation,
    RemoteWorkspacePreferences,
    load_remote_workspace_preferences,
    save_remote_workspace_preferences,
)


class MemorySettings:
    def __init__(self, classic=None):
        self.classic = classic or {}

    def load_namespace(self, name):
        assert name == "classic"
        return self.classic

    def save_namespace(self, name, payload):
        assert name == "classic"
        self.classic.update(payload)


def test_navigation_tracks_browser_style_history() -> None:
    navigation = RemoteNavigation("/home")

    navigation.visit("/work")
    navigation.visit("/work/calc")

    assert navigation.back() == "/work"
    assert navigation.back() == "/home"
    assert navigation.can_back is False
    assert navigation.forward() == "/work"
    assert navigation.can_forward is True


def test_navigation_deduplicates_and_discards_forward_branch() -> None:
    navigation = RemoteNavigation("/home")
    navigation.visit("/work")
    navigation.visit("/work")
    navigation.back()

    navigation.visit("/scratch")

    assert navigation.history == ("/home", "/scratch")
    assert navigation.can_forward is False


def test_preferences_ignore_invalid_and_duplicate_bookmarks() -> None:
    preferences = RemoteWorkspacePreferences.from_mapping(
        {
            "last_profile": "  cluster-a  ",
            "last_path": "relative/path",
            "workflow": "unknown",
            "bookmarks": ["/work", "/work/", "relative", "/scratch"],
            "editor_expanded": "yes",
        }
    )

    assert preferences.last_profile == "cluster-a"
    assert preferences.last_path == "/home"
    assert preferences.workflow == "band"
    assert preferences.bookmarks == ("/work", "/scratch")
    assert preferences.editor_expanded is False


def test_preferences_round_trip_through_classic_settings_namespace() -> None:
    store = MemorySettings()
    expected = RemoteWorkspacePreferences(
        last_profile="cluster-a",
        last_path="/work/calc",
        workflow="dos",
        bookmarks=("/work",),
        editor_expanded=True,
    )

    save_remote_workspace_preferences(store, expected)

    assert load_remote_workspace_preferences(store) == expected


def test_preferences_bound_untrusted_collections_and_strings() -> None:
    preferences = RemoteWorkspacePreferences.from_mapping(
        {
            "last_profile": "x" * 1000,
            "bookmarks": [f"/work/{index}" for index in range(100)],
        }
    )

    assert preferences.last_profile == ""
    assert len(preferences.bookmarks) == 32


def test_preferences_reject_non_string_workflow_values() -> None:
    preferences = RemoteWorkspacePreferences.from_mapping({"workflow": []})

    assert preferences.workflow == "band"

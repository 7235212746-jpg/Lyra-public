"""Independent, transport-neutral remote Linux support for Lyra."""

from .browser import RemoteBrowser
from .cache_sync import SelectiveCacheSync
from .credentials import (
    CredentialCorruptError,
    CredentialNotFoundError,
    CredentialPlatformError,
    CredentialStore,
    CredentialStoreError,
    RemoteCredentials,
    delete as delete_credentials,
    load as load_credentials,
    save as save_credentials,
)
from .jobs import RemoteJobClient
from .manifest import CalculationManifest, build_calculation_manifest
from .profiles import ProfileStore, RemoteProfile, RuntimeAuthentication
from .runtime import (
    RemotePreflightReport,
    RemoteRuntimeBundle,
    RemoteRuntimePaths,
    RemoteRuntimeRequirements,
    build_runtime_bundle,
    decode_preflight_report,
    preflight_script_bytes,
    runtime_remote_paths,
)
from .runner import RemoteRunner
from .transport import connect_pinned
from .winscp import WinScpInstallation, WinScpTransport
from .workspace_state import (
    RemoteNavigation,
    RemoteWorkspacePreferences,
    load_remote_workspace_preferences,
    save_remote_workspace_preferences,
)

__all__ = [
    "CalculationManifest",
    "CredentialCorruptError",
    "CredentialNotFoundError",
    "CredentialPlatformError",
    "CredentialStore",
    "CredentialStoreError",
    "ProfileStore",
    "RemoteBrowser",
    "RemoteCredentials",
    "RemoteJobClient",
    "RemoteNavigation",
    "RemoteProfile",
    "RemotePreflightReport",
    "RemoteRuntimeBundle",
    "RemoteRuntimePaths",
    "RemoteRuntimeRequirements",
    "RemoteRunner",
    "RemoteWorkspacePreferences",
    "RuntimeAuthentication",
    "SelectiveCacheSync",
    "build_calculation_manifest",
    "build_runtime_bundle",
    "connect_pinned",
    "delete_credentials",
    "load_credentials",
    "load_remote_workspace_preferences",
    "decode_preflight_report",
    "preflight_script_bytes",
    "runtime_remote_paths",
    "save_credentials",
    "save_remote_workspace_preferences",
    "WinScpInstallation",
    "WinScpTransport",
]

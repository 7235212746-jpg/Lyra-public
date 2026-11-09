from __future__ import annotations

import hashlib
from pathlib import Path

from .jobs import RemoteJobClient
from .protocol import RemoteEvent, RemoteResult
from .runtime import (
    RemotePreflightReport,
    RemoteRuntimeBundle,
    RemoteRuntimePaths,
    build_runtime_bundle,
    decode_preflight_report,
    preflight_script_bytes,
    runtime_remote_paths,
)
from .transport import RemoteSession, validate_remote_path


class RemoteRunner:
    def __init__(self, session: RemoteSession, package_root: str | Path) -> None:
        self.session = session
        self.package_root = Path(package_root).resolve()
        self.bundle: RemoteRuntimeBundle | None = None
        self.paths: RemoteRuntimePaths | None = None
        self.client: RemoteJobClient | None = None
        self.preflight: RemotePreflightReport | None = None

    def prepare(self) -> RemotePreflightReport:
        remote_home = self.session.remote_home()
        bundle = build_runtime_bundle(self.package_root)
        paths = runtime_remote_paths(remote_home, bundle.digest)
        agent = Path(__file__).with_name("agent.py").read_bytes()
        self.session.write_bytes_atomic(paths.preflight_path, preflight_script_bytes())
        self.session.write_bytes_atomic(paths.runtime_zip, bundle.payload)
        self.session.write_bytes_atomic(paths.agent_path, agent)
        report = decode_preflight_report(
            self.session.runtime_preflight(paths.preflight_path, paths.cache_root)
        )
        self.bundle = bundle
        self.paths = paths
        self.preflight = report
        if hasattr(self.session, "remote_agent_path"):
            self.session.remote_agent_path = paths.agent_path
        self.client = RemoteJobClient(self.session, paths.cache_root)
        return report

    def start(
        self,
        workflow: str,
        source_directory: str,
        manifest_fingerprint: str,
        plot_options: dict | None = None,
    ):
        if self.client is None or self.paths is None or self.preflight is None:
            raise RuntimeError("Remote runtime has not been prepared.")
        if not self.preflight.compatible:
            raise RuntimeError("远端科学环境不可用：" + " ".join(self.preflight.reasons))
        if workflow not in {"band", "projected_band", "dos"}:
            raise ValueError("Remote workflow is unsupported.")
        options = {
            "workflow": workflow,
            "manifest_fingerprint": manifest_fingerprint,
            "runtime_zip": self.paths.runtime_zip,
            "plot": dict(plot_options or {}),
        }
        task = "dos" if workflow == "dos" else "band"
        return self.client.submit(
            task,
            (validate_remote_path(source_directory, allow_root=False),),
            options,
        )

    def poll(self, job_id) -> RemoteEvent | None:
        if self.client is None:
            raise RuntimeError("Remote runtime has not been prepared.")
        return self.client.poll(job_id)

    def cancel(self, job_id) -> bool:
        if self.client is None:
            return False
        return self.client.cancel(job_id)

    def fetch_preview_bytes(self, result: RemoteResult) -> bytes:
        matches = [artifact for artifact in result.artifacts if artifact.role == "preview"]
        if len(matches) != 1:
            raise ValueError("Remote result must contain exactly one preview.")
        artifact = matches[0]
        if artifact.media_type != "image/png" or artifact.size > 32 * 1024 * 1024:
            raise ValueError("Remote preview type or size is invalid.")
        payload = self.session.read_bytes(
            artifact.remote_path,
            max_bytes=max(artifact.size, 1),
        )
        if len(payload) != artifact.size:
            raise ValueError("Remote preview size verification failed.")
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("Remote preview hash verification failed.")
        return payload

    def read_error(self, job_id) -> str:
        if self.paths is None:
            return "远端任务失败。"
        path = f"{self.paths.cache_root}/jobs/{job_id}/error.json"
        try:
            import json

            value = json.loads(self.session.read_bytes(path, max_bytes=64 * 1024))
        except (OSError, KeyError, ValueError, UnicodeError):
            return "远端任务失败，服务器未返回可读取的错误详情。"
        error = value.get("error") if isinstance(value, dict) else None
        return error if isinstance(error, str) and error else "远端任务失败。"

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        else:
            self.session.close()

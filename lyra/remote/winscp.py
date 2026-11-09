from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Callable
from uuid import UUID

from .profiles import RemoteProfile, RuntimeAuthentication
from .transport import (
    RemoteDisconnected,
    RemoteError,
    RemoteSession,
    RemoteStat,
    validate_remote_path,
)


@dataclass(frozen=True)
class WinScpInstallation:
    executable: Path
    assembly: Path
    bridge_script: Path

    @classmethod
    def find(cls) -> "WinScpInstallation | None":
        roots = tuple(
            Path(value)
            for value in (
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
                os.environ.get("LOCALAPPDATA"),
            )
            if value
        )
        executable = shutil.which("WinSCP.exe")
        candidates = ([Path(executable)] if executable else []) + [
            candidate
            for root in roots
            for candidate in (
                root / "WinSCP" / "WinSCP.exe",
                root / "Programs" / "WinSCP" / "WinSCP.exe",
            )
        ]
        bridge = Path(__file__).with_name("winscp_bridge.ps1")
        for candidate in candidates:
            assembly = candidate.with_name("WinSCPnet.dll")
            if candidate.is_file() and assembly.is_file() and bridge.is_file():
                return cls(candidate.resolve(), assembly.resolve(), bridge.resolve())
        return None


class WinScpTransport:
    def __init__(
        self,
        installation: WinScpInstallation | None = None,
        *,
        powershell: str | Path = "powershell.exe",
        runner: Callable | None = None,
        timeout: float = 60.0,
        remote_agent_path: str = "/home/lyra/.cache/lyra/agent.py",
    ) -> None:
        located = installation or WinScpInstallation.find()
        if located is None:
            raise FileNotFoundError("未找到 WinSCP.exe 与同版本 WinSCPnet.dll。")
        self.installation = located
        self.powershell = str(powershell)
        self.runner = runner
        self.timeout = float(timeout)
        self.remote_agent_path = validate_remote_path(
            remote_agent_path, allow_root=False
        )
        self._processes: set[subprocess.Popen] = set()
        self._process_lock = threading.Lock()
        self._cancelled = False

    def observe_host_fingerprint(self, profile: RemoteProfile) -> str:
        result = self._invoke("scan", profile, {})
        fingerprint = result.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise RemoteError("WinSCP 未返回服务器 SHA-256 主机指纹。")
        return fingerprint.strip()

    def connect(
        self,
        profile: RemoteProfile,
        authentication: RuntimeAuthentication,
    ) -> RemoteSession:
        if not isinstance(authentication, RuntimeAuthentication):
            raise TypeError("authentication must be RuntimeAuthentication")
        if profile.host_fingerprint is None:
            raise ValueError("连接前必须确认并保存服务器 SHA-256 主机指纹。")
        if authentication.password is None and profile.private_key_path is None:
            raise ValueError(
                "尚未提供登录凭据。请在连接配置中输入 SSH 密码，"
                "或选择本机私钥后再连接。"
            )
        if profile.private_key_path and not Path(profile.private_key_path).expanduser().is_file():
            raise FileNotFoundError(f"未找到私钥文件：{profile.private_key_path}")
        return WinScpSession(
            self,
            profile,
            self.remote_agent_path,
            authentication,
        )

    def _invoke(
        self,
        operation: str,
        profile: RemoteProfile,
        parameters: dict,
        authentication: RuntimeAuthentication | None = None,
    ) -> dict:
        authentication = authentication or RuntimeAuthentication()
        request = {
            "operation": operation,
            "exe_path": str(self.installation.executable),
            "dll_path": str(self.installation.assembly),
            "profile": profile.to_persistent_dict(),
            "authentication": {
                "password": authentication.password,
                "private_key_passphrase": authentication.private_key_passphrase,
            },
            **parameters,
        }
        command = [
            self.powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.installation.bridge_script),
        ]
        try:
            if self.runner is not None:
                completed = self.runner(
                    command,
                    input=json.dumps(request, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                completed = self._run_interruptible(
                    command,
                    json.dumps(request, ensure_ascii=False),
                )
        except subprocess.TimeoutExpired as exc:
            raise RemoteDisconnected("WinSCP 操作超时。") from exc
        except OSError as exc:
            raise RemoteDisconnected(f"无法启动 WinSCP：{exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()
            if not detail and completed.stdout:
                try:
                    failure = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    failure = None
                if isinstance(failure, dict):
                    detail = str(failure.get("error") or "")
            detail = self._redact_runtime_secrets(
                (detail or "WinSCP bridge failed")[-1000:],
                authentication,
            )
            raise RemoteError(detail)
        try:
            result = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RemoteError("WinSCP 返回了无效响应。") from exc
        if not isinstance(result, dict) or result.get("ok") is not True:
            detail = result.get("error") if isinstance(result, dict) else None
            detail = self._redact_runtime_secrets(
                str(detail or "WinSCP 操作失败。")[:1000],
                authentication,
            )
            raise RemoteError(detail)
        return result

    @staticmethod
    def _redact_runtime_secrets(
        message: str,
        authentication: RuntimeAuthentication,
    ) -> str:
        redacted = message
        for secret in (
            authentication.password,
            authentication.private_key_passphrase,
        ):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        return redacted

    def _run_interruptible(self, command: list[str], request: str):
        with self._process_lock:
            if self._cancelled:
                raise RemoteDisconnected("WinSCP 操作已取消。")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._processes.add(process)
        try:
            try:
                stdout, stderr = process.communicate(input=request, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                raise
        finally:
            with self._process_lock:
                self._processes.discard(process)

        class Completed:
            pass

        completed = Completed()
        completed.returncode = process.returncode
        completed.stdout = stdout
        completed.stderr = stderr
        return completed

    def cancel_all(self) -> None:
        with self._process_lock:
            self._cancelled = True
            processes = tuple(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass


class WinScpSession(RemoteSession):
    def __init__(
        self,
        transport: WinScpTransport,
        profile: RemoteProfile,
        remote_agent_path: str,
        authentication: RuntimeAuthentication,
    ) -> None:
        self.transport = transport
        self.profile = profile
        self.remote_agent_path = remote_agent_path
        self.authentication = authentication
        self._closed = False

    def _invoke(self, operation: str, parameters: dict) -> dict:
        if self._closed:
            raise RemoteDisconnected("WinSCP 会话已关闭。")
        return self.transport._invoke(
            operation,
            self.profile,
            parameters,
            self.authentication,
        )

    def list_directory(self, path: str) -> tuple[RemoteStat, ...]:
        remote_path = validate_remote_path(path)
        result = self._invoke("list", {"remote_path": remote_path})
        entries = result.get("entries")
        if not isinstance(entries, list):
            raise RemoteError("WinSCP 目录响应缺少 entries。")
        return tuple(
            RemoteStat(
                name=item["name"],
                path=item["path"],
                is_directory=item["is_directory"],
                size=item["size"],
                modified_ns=item["modified_ns"],
            )
            for item in entries
            if isinstance(item, dict)
        )

    def remote_home(self) -> str:
        result = self._invoke("home", {})
        home = result.get("home")
        if not isinstance(home, str):
            raise RemoteError("WinSCP 未返回远端 Home 路径。")
        return validate_remote_path(home, allow_root=False)

    def runtime_preflight(self, script_path: str, cache_root: str) -> bytes:
        result = self._invoke(
            "runtime-preflight",
            {
                "script_path": validate_remote_path(script_path, allow_root=False),
                "cache_root": validate_remote_path(cache_root, allow_root=False),
            },
        )
        report = result.get("report")
        if not isinstance(report, str):
            raise RemoteError("WinSCP 未返回远端运行环境报告。")
        data = report.encode("utf-8")
        if len(data) > 64 * 1024:
            raise RemoteError("远端运行环境报告超过大小限制。")
        return data

    def read_bytes(self, path: str, *, max_bytes: int) -> bytes:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        with tempfile.TemporaryDirectory(prefix="lyra-winscp-") as temporary:
            local = Path(temporary) / "payload.bin"
            self._invoke(
                "get",
                {
                    "remote_path": validate_remote_path(path, allow_root=False),
                    "local_path": str(local),
                    "max_bytes": int(max_bytes),
                },
            )
            data = local.read_bytes()
        if len(data) > max_bytes:
            raise ValueError("remote payload exceeds configured bound")
        return data

    def write_bytes_atomic(self, path: str, data: bytes, *, mode: int = 0o600) -> None:
        if not isinstance(data, bytes):
            raise TypeError("remote payload must be bytes")
        if mode not in {0o600, 0o700}:
            raise ValueError("remote file mode must be 0600 or 0700")
        with tempfile.TemporaryDirectory(prefix="lyra-winscp-") as temporary:
            local = Path(temporary) / "upload.bin"
            local.write_bytes(data)
            self._invoke(
                "put-atomic",
                {
                    "remote_path": validate_remote_path(path, allow_root=False),
                    "local_path": str(local),
                    "mode": format(mode, "o"),
                },
            )

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        *,
        max_bytes: int,
    ) -> int:
        destination = Path(local_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.lyra-part")
        try:
            result = self._invoke(
                "get",
                {
                    "remote_path": validate_remote_path(
                        remote_path, allow_root=False
                    ),
                    "local_path": str(temporary),
                    "max_bytes": int(max_bytes),
                },
            )
            size = temporary.stat().st_size
            if size > max_bytes:
                raise ValueError("remote payload exceeds configured bound")
            os.replace(temporary, destination)
            return int(result.get("size", size))
        finally:
            temporary.unlink(missing_ok=True)

    def start_job(self, job_id: UUID) -> None:
        if not isinstance(job_id, UUID):
            raise ValueError("Remote job ID must be a UUID.")
        self._invoke(
            "start-job",
            {"agent_path": self.remote_agent_path, "job_id": str(job_id)},
        )

    def close(self) -> None:
        self._closed = True

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from .auth import AuthStore, Device, PairingError
from .resources import ResourceError, ResourceRegistry
from .tasks import InMemoryTaskBackend, TaskBackend, TaskError, validate_task_request


MAX_BODY_BYTES = 64 * 1024
TASK_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
DEVICE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
RESOURCE_ROUTE = re.compile(r"^/api/resources/([A-Za-z0-9_-]{20,80})/(preview|download)$")
TASK_ROUTE = re.compile(r"^/api/tasks/([a-f0-9]{32})$")
CANCEL_ROUTE = re.compile(r"^/api/tasks/([a-f0-9]{32})/cancel$")
REVOKE_ROUTE = re.compile(r"^/api/devices/([a-f0-9]{32})/revoke$")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    trusted_origins: tuple[str, ...] = ()
    rate_limit: int = 120
    rate_window_seconds: float = 60.0

    def __post_init__(self) -> None:
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise ValueError("mobile service host must be a numeric loopback address") from exc
        if not address.is_loopback:
            raise ValueError("mobile service may bind only to a loopback address")
        if not 0 <= self.port <= 65535:
            raise ValueError("invalid mobile service port")
        if self.rate_limit <= 0 or self.rate_window_seconds <= 0:
            raise ValueError("rate limit settings must be positive")
        for origin in self.trusted_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
                raise ValueError("trusted origins must be absolute HTTP(S) origins")


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float, *, clock=time.monotonic) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.clock = clock
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = self.clock()
        with self._lock:
            started, count = self._entries.get(key, (now, 0))
            if now - started >= self.window_seconds:
                started, count = now, 0
            count += 1
            self._entries[key] = (started, count)
            retry_after = max(1, int(self.window_seconds - (now - started)) + 1)
            return count <= self.limit, retry_after


class MobileService:
    def __init__(
        self,
        *,
        backend: TaskBackend | None = None,
        auth: AuthStore | None = None,
        resources: ResourceRegistry | None = None,
        config: ServerConfig | None = None,
        static_root: Path | None = None,
    ) -> None:
        self.backend = backend or InMemoryTaskBackend()
        self.auth = auth or AuthStore()
        self.resources = resources or ResourceRegistry()
        self.config = config or ServerConfig()
        self.static_root = (
            Path(static_root).resolve()
            if static_root
            else Path(__file__).resolve().parents[1] / "resources" / "mobile"
        )
        self.rate_limiter = RateLimiter(
            self.config.rate_limit, self.config.rate_window_seconds
        )
        self.pair_rate_limiter = RateLimiter(8, 300.0)
        self._runtime_trusted_origins: set[str] = set()
        self.started_at = time.time()

    def trust_origin(self, origin: str) -> None:
        normalized = str(origin).strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("runtime trusted origin must be an HTTPS origin")
        self._runtime_trusted_origins.add(normalized)

    def create_server(self) -> ThreadingHTTPServer:
        service = self

        class Handler(MobileRequestHandler):
            mobile_service = service

        server = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        server.daemon_threads = True
        return server

    def serve_forever(self) -> None:
        server = self.create_server()
        try:
            server.serve_forever()
        finally:
            server.server_close()

    def origin_is_trusted(self, origin: str | None, server_port: int) -> bool:
        if not origin:
            return False
        normalized = origin.rstrip("/")
        if normalized in self._runtime_trusted_origins or normalized in {
            item.rstrip("/") for item in self.config.trusted_origins
        }:
            return True
        try:
            parsed = urlsplit(normalized)
            host = ipaddress.ip_address(parsed.hostname or "")
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except (ValueError, TypeError):
            return False
        return parsed.scheme == "http" and host.is_loopback and port == server_port


class MobileRequestHandler(BaseHTTPRequestHandler):
    mobile_service: MobileService
    server_version = "LyraMobile/0.1"
    sys_version = ""

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self._json_error(HTTPStatus.METHOD_NOT_ALLOWED, "method_not_allowed", "CORS preflight is not enabled")

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _dispatch(self, method: str) -> None:
        authorization = self.headers.get("Authorization", "")
        presented_token = authorization[7:] if authorization.startswith("Bearer ") else ""
        rate_device = self.mobile_service.auth.authenticate(presented_token)
        rate_identity = rate_device.device_id if rate_device is not None else "anonymous"
        allowed, retry_after = self.mobile_service.rate_limiter.allow(
            f"{self.client_address[0]}:{rate_identity}"
        )
        if not allowed:
            self._json_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "rate_limited",
                "Too many requests",
                headers={"Retry-After": str(retry_after)},
            )
            return
        try:
            path = unquote(urlsplit(self.path).path)
            if method == "GET" and path in {"/", "/index.html", "/app.css", "/app.js", "/sw.js", "/manifest.webmanifest", "/icon.svg"}:
                self._serve_static(path)
                return
            if method == "GET" and path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "lyra-mobile",
                        "version": 1,
                        "uptime_seconds": max(0, int(time.time() - self.mobile_service.started_at)),
                    },
                )
                return
            if method == "POST" and path == "/api/pair":
                self._pair()
                return

            device = self._require_auth()
            if device is None:
                return
            if method == "GET" and path == "/api/tasks":
                tasks = [task.to_public_dict() for task in self.mobile_service.backend.list_tasks()]
                self._json(HTTPStatus.OK, {"tasks": tasks})
                return
            if method == "GET" and path == "/api/devices":
                devices = [item.to_public_dict() for item in self.mobile_service.auth.list_devices()]
                self._json(HTTPStatus.OK, {"devices": devices, "current_device_id": device.device_id})
                return
            if method == "GET" and path == "/api/capabilities":
                capability_provider = getattr(self.mobile_service.backend, "capabilities", None)
                capabilities = capability_provider() if callable(capability_provider) else {
                    "task_kinds": [],
                    "projection": {"supported": False, "reason": "The active backend does not publish plotting capabilities."},
                    "directories": [],
                }
                self._json(HTTPStatus.OK, {"capabilities": capabilities})
                return
            task_match = TASK_ROUTE.fullmatch(path)
            if method == "GET" and task_match:
                task = self.mobile_service.backend.get(task_match.group(1))
                self._json(HTTPStatus.OK, {"task": task.to_public_dict()})
                return
            resource_match = RESOURCE_ROUTE.fullmatch(path)
            if method == "GET" and resource_match:
                self._serve_resource(resource_match.group(1), resource_match.group(2))
                return

            if method == "POST":
                if not self._require_mutation_protection(device):
                    return
                if path == "/api/tasks":
                    payload = self._read_json()
                    kind, parameters = validate_task_request(payload)
                    task = self.mobile_service.backend.create(kind, parameters)
                    self._json(HTTPStatus.CREATED, {"task": task.to_public_dict()})
                    return
                cancel_match = CANCEL_ROUTE.fullmatch(path)
                if cancel_match:
                    task = self.mobile_service.backend.cancel(cancel_match.group(1))
                    self._json(HTTPStatus.OK, {"task": task.to_public_dict()})
                    return
                revoke_match = REVOKE_ROUTE.fullmatch(path)
                if revoke_match:
                    if revoke_match.group(1) != device.device_id:
                        self._json_error(HTTPStatus.FORBIDDEN, "forbidden", "A mobile device may revoke only itself")
                        return
                    revoked = self.mobile_service.auth.revoke(revoke_match.group(1))
                    if not revoked:
                        self._json_error(HTTPStatus.NOT_FOUND, "not_found", "Device not found")
                        return
                    self._json(HTTPStatus.OK, {"revoked": True})
                    return
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "Endpoint not found")
        except (TaskError, ResourceError) as exc:
            status = HTTPStatus.NOT_FOUND if "not found" in str(exc).lower() else HTTPStatus.BAD_REQUEST
            self._json_error(status, "invalid_request", str(exc))
        except (ValueError, json.JSONDecodeError):
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_json", "Invalid JSON request")
        except Exception:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "The request could not be completed")

    def _pair(self) -> None:
        allowed, retry_after = self.mobile_service.pair_rate_limiter.allow(self.client_address[0])
        if not allowed:
            self._json_error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "rate_limited",
                "Too many pairing attempts",
                headers={"Retry-After": str(retry_after)},
            )
            return
        if not self._origin_is_trusted():
            self._json_error(HTTPStatus.FORBIDDEN, "untrusted_origin", "Request origin is not trusted")
            return
        payload = self._read_json()
        if not isinstance(payload, dict):
            raise ValueError("invalid payload")
        try:
            credentials = self.mobile_service.auth.pair(
                str(payload.get("code", "")), str(payload.get("device_name", ""))
            )
        except PairingError:
            self._json_error(HTTPStatus.UNAUTHORIZED, "pairing_failed", "Invalid or expired pairing code")
            return
        self._json(
            HTTPStatus.CREATED,
            {
                "device_id": credentials.device_id,
                "token": credentials.token,
                "csrf_token": credentials.csrf_token,
            },
        )

    def _require_auth(self) -> Device | None:
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        device = self.mobile_service.auth.authenticate(token)
        if device is None:
            self._json_error(HTTPStatus.UNAUTHORIZED, "unauthorized", "Valid device authorization is required")
        return device

    def _require_mutation_protection(self, device: Device) -> bool:
        if not self._origin_is_trusted():
            self._json_error(HTTPStatus.FORBIDDEN, "untrusted_origin", "Request origin is not trusted")
            return False
        if not self.mobile_service.auth.verify_csrf(device, self.headers.get("X-Lyra-CSRF", "")):
            self._json_error(HTTPStatus.FORBIDDEN, "csrf_failed", "CSRF validation failed")
            return False
        return True

    def _origin_is_trusted(self) -> bool:
        return self.mobile_service.origin_is_trusted(
            self.headers.get("Origin"), self.server.server_address[1]
        )

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("invalid body size")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("JSON content type required")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _serve_resource(self, resource_id: str, purpose: str) -> None:
        resource = self.mobile_service.resources.resolve(resource_id, purpose=purpose)
        disposition = "inline" if purpose == "preview" else "attachment"
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", resource.public.label) or "lyra-result"
        size = resource.path.stat().st_size
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", resource.public.media_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'{disposition}; filename="{filename}"')
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        with resource.path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                self.wfile.write(chunk)

    def _serve_static(self, request_path: str) -> None:
        names = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.css": "app.css",
            "/app.js": "app.js",
            "/sw.js": "sw.js",
            "/manifest.webmanifest": "manifest.webmanifest",
            "/icon.svg": "icon.svg",
        }
        name = names[request_path]
        path = self.mobile_service.static_root / name
        if not path.is_file():
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "Static asset not found")
            return
        media_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".webmanifest": "application/manifest+json",
            ".svg": "image/svg+xml",
        }.get(path.suffix, "application/octet-stream")
        cache = "no-cache" if name in {"index.html", "sw.js"} else "public, max-age=3600"
        self._bytes(HTTPStatus.OK, path.read_bytes(), media_type, headers={"Cache-Control": cache})

    def _json_error(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._json(status, {"error": {"code": code, "message": message}}, headers=headers)

    def _json(
        self,
        status: HTTPStatus,
        payload: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._bytes(status, data, "application/json; charset=utf-8", headers=headers)

    def _bytes(
        self,
        status: HTTPStatus,
        data: bytes,
        content_type: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        if content_type.startswith("application/json"):
            self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

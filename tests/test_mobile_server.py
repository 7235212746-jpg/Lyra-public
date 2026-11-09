from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import pytest

from lyra.mobile.auth import AuthStore
from lyra.mobile.resources import ResourceRegistry
from lyra.mobile.server import MobileService, ServerConfig
from lyra.mobile.tasks import InMemoryTaskBackend


ORIGIN = "https://lyra.example.ts.net"


@contextmanager
def running_service(tmp_path: Path, *, rate_limit: int = 100):
    auth = AuthStore(state_file=tmp_path / "auth.json")
    backend = InMemoryTaskBackend()
    resources = ResourceRegistry()
    service = MobileService(
        backend=backend,
        auth=auth,
        resources=resources,
        config=ServerConfig(
            host="127.0.0.1",
            port=0,
            trusted_origins=(ORIGIN,),
            rate_limit=rate_limit,
            rate_window_seconds=60,
        ),
    )
    server = service.create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port: int, method: str, path: str, body=None, headers=None):
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = dict(headers or {})
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    for attempt in range(3):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path, body=encoded, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            content_type = response.getheader("Content-Type", "")
            break
        except (ConnectionAbortedError, ConnectionResetError):
            if attempt == 2:
                raise
            time.sleep(0.02)
        finally:
            connection.close()
    if "application/json" in content_type:
        return response.status, json.loads(data), dict(response.getheaders())
    return response.status, data, dict(response.getheaders())


def pair(service: MobileService, port: int):
    code = service.auth.issue_pairing_code()
    status, payload, _ = request(
        port,
        "POST",
        "/api/pair",
        {"code": code, "device_name": "Phone"},
        {"Origin": ORIGIN},
    )
    assert status == 201
    return payload


def auth_headers(credentials, *, origin=ORIGIN, csrf=True):
    headers = {"Authorization": f"Bearer {credentials['token']}", "Origin": origin}
    if csrf:
        headers["X-Lyra-CSRF"] = credentials["csrf_token"]
    return headers


def test_server_refuses_non_loopback_binding() -> None:
    with pytest.raises(ValueError, match="loopback"):
        ServerConfig(host="0.0.0.0")


def test_health_is_public_but_tasks_require_authorization(tmp_path: Path) -> None:
    with running_service(tmp_path) as (_service, port):
        assert request(port, "GET", "/api/health")[0] == 200
        status, payload, _ = request(port, "GET", "/api/tasks")
        assert status == 401
        assert payload["error"]["code"] == "unauthorized"


def test_capabilities_require_authorization_and_are_machine_readable(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        assert request(port, "GET", "/api/capabilities")[0] == 401
        credentials = pair(service, port)
        status, payload, _ = request(
            port, "GET", "/api/capabilities", headers=auth_headers(credentials)
        )
        assert status == 200
        assert payload["capabilities"]["projection"]["supported"] is False


def test_pair_create_list_detail_cancel_and_revoke_flow(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        credentials = pair(service, port)
        headers = auth_headers(credentials)

        status, created, _ = request(
            port,
            "POST",
            "/api/tasks",
            {"kind": "band", "parameters": {"directory_id": "calc-1"}},
            headers,
        )
        assert status == 201
        task_id = created["task"]["id"]
        assert request(port, "GET", "/api/tasks", headers=headers)[1]["tasks"][0]["id"] == task_id
        assert request(port, "GET", f"/api/tasks/{task_id}", headers=headers)[0] == 200
        assert request(port, "POST", f"/api/tasks/{task_id}/cancel", {}, headers)[1]["task"]["state"] == "cancelled"

        device_id = credentials["device_id"]
        assert request(port, "POST", f"/api/devices/{device_id}/revoke", {}, headers)[0] == 200
        assert request(port, "GET", "/api/tasks", headers=headers)[0] == 401


def test_mutations_reject_missing_csrf_and_untrusted_origin(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        credentials = pair(service, port)
        body = {"kind": "dos", "parameters": {"directory_id": "calc-1"}}

        assert request(port, "POST", "/api/tasks", body, auth_headers(credentials, csrf=False))[0] == 403
        assert request(port, "POST", "/api/tasks", body, auth_headers(credentials, origin="https://evil.invalid"))[0] == 403


def test_mobile_device_cannot_revoke_another_device(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        first = pair(service, port)
        second = pair(service, port)

        status, payload, _ = request(
            port,
            "POST",
            f"/api/devices/{second['device_id']}/revoke",
            {},
            auth_headers(first),
        )

        assert status == 403
        assert payload["error"]["code"] == "forbidden"
        assert request(port, "GET", "/api/tasks", headers=auth_headers(second))[0] == 200


def test_task_payload_rejects_file_paths_and_does_not_echo_value(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        credentials = pair(service, port)
        secret = "C:/Users/private/.ssh/id_ed25519"
        status, payload, _ = request(
            port,
            "POST",
            "/api/tasks",
            {"kind": "band", "parameters": {"path": secret}},
            auth_headers(credentials),
        )
        assert status == 400
        assert secret not in json.dumps(payload)


def test_registered_preview_and_download_use_opaque_id(tmp_path: Path) -> None:
    with running_service(tmp_path) as (service, port):
        credentials = pair(service, port)
        image = tmp_path / "private-preview.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
        public = service.resources.register(image, label="Preview")
        headers = auth_headers(credentials)

        preview = request(port, "GET", f"/api/resources/{public.resource_id}/preview", headers=headers)
        download = request(port, "GET", f"/api/resources/{public.resource_id}/download", headers=headers)
        traversal = request(port, "GET", "/api/resources/..%2F..%2Fsecret/download", headers=headers)

        assert preview[0] == 200 and preview[1].startswith(b"\x89PNG")
        assert download[0] == 200
        assert traversal[0] in {400, 404}


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    with running_service(tmp_path, rate_limit=2) as (_service, port):
        assert request(port, "GET", "/api/health")[0] == 200
        assert request(port, "GET", "/api/health")[0] == 200
        status, payload, headers = request(port, "GET", "/api/health")
        assert status == 429
        assert payload["error"]["code"] == "rate_limited"
        assert "Retry-After" in headers


def test_pairing_has_a_stricter_brute_force_limit(tmp_path: Path) -> None:
    with running_service(tmp_path) as (_service, port):
        statuses = []
        for _ in range(9):
            statuses.append(request(
                port,
                "POST",
                "/api/pair",
                {"code": "000000", "device_name": "Unknown"},
                {"Origin": ORIGIN},
            )[0])
            time.sleep(0.01)
        assert statuses[:8] == [401] * 8
        assert statuses[8] == 429


def test_pwa_assets_have_manifest_service_worker_and_security_headers(tmp_path: Path) -> None:
    with running_service(tmp_path) as (_service, port):
        status, html, headers = request(port, "GET", "/")
        assert status == 200
        assert b"manifest.webmanifest" in html
        assert b"serviceWorker" in html
        assert b'id="directory-id"' in html
        assert b'id="band-directory-id"' in html
        assert b'id="dos-directory-id"' in html
        assert b'value="audit"' not in html
        app_js = request(port, "GET", "/app.js")[1]
        assert b"sessionStorage" in app_js
        assert b"localStorage" not in app_js
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert request(port, "GET", "/manifest.webmanifest")[0] == 200
        assert request(port, "GET", "/sw.js")[0] == 200


def test_runtime_private_origin_can_be_trusted_after_tailscale_serve(tmp_path: Path) -> None:
    service = MobileService(
        auth=AuthStore(state_file=tmp_path / "auth.json"),
        config=ServerConfig(host="127.0.0.1", port=0),
    )

    assert not service.origin_is_trusted("https://pc.example.ts.net", 8765)
    service.trust_origin("https://pc.example.ts.net")
    assert service.origin_is_trusted("https://pc.example.ts.net", 8765)
    with pytest.raises(ValueError):
        service.trust_origin("javascript:alert(1)")

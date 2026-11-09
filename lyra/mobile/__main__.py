from __future__ import annotations

import argparse
import os
from pathlib import Path

from .auth import AuthStore
from .lyra_backend import AuthorizedDirectoryRegistry, LyraTaskBackend
from .resources import ResourceRegistry
from .server import MobileService, ServerConfig


def _state_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".lyra"))
    return root / "Lyra" / "mobile" / "auth.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lyra loopback-only mobile service")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LYRA_MOBILE_PORT", "8765")))
    parser.add_argument("--origin", action="append", default=[], help="Trusted Tailscale HTTPS origin")
    parser.add_argument("--directory", action="append", default=[], metavar="LABEL=PATH", help="Desktop-authorized calculation directory")
    args = parser.parse_args()
    environment_origins = [
        item.strip()
        for item in os.environ.get("LYRA_MOBILE_ORIGINS", "").split(",")
        if item.strip()
    ]
    origins = tuple(
        dict.fromkeys(item.rstrip("/") for item in (*environment_origins, *args.origin))
    )
    directories = AuthorizedDirectoryRegistry()
    for item in args.directory:
        label, separator, raw_path = item.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            parser.error("--directory must use LABEL=PATH")
        directories.authorize(Path(raw_path), label=label)
    resources = ResourceRegistry()
    backend = LyraTaskBackend(directories=directories, resources=resources)
    service = MobileService(
        backend=backend,
        auth=AuthStore(state_file=_state_path()),
        resources=resources,
        config=ServerConfig(
            host="127.0.0.1",
            port=args.port,
            trusted_origins=origins,
        ),
    )
    code = service.auth.issue_pairing_code()
    print(f"Lyra Mobile: http://127.0.0.1:{service.config.port}")
    print(f"Pairing code (valid for 5 minutes): {code}")
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()


if __name__ == "__main__":
    main()

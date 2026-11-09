from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
import uuid


class PairingError(ValueError):
    pass


@dataclass(frozen=True)
class PairingCredentials:
    device_id: str
    token: str
    csrf_token: str


@dataclass(frozen=True)
class Device:
    device_id: str
    name: str
    token_hash: str
    csrf_hash: str
    created_at: float
    expires_at: float
    revoked: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {
            "id": self.device_id,
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
        }


def _secret_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthStore:
    """Stores only hashes of high-entropy device credentials."""

    def __init__(
        self,
        *,
        state_file: Path | None = None,
        pairing_ttl: float = 300.0,
        token_ttl: float = 30 * 24 * 60 * 60,
        clock=time.time,
    ) -> None:
        self.state_file = Path(state_file).resolve() if state_file else None
        self.pairing_ttl = float(pairing_ttl)
        self.token_ttl = float(token_ttl)
        if self.token_ttl <= 0:
            raise ValueError("token_ttl must be positive")
        self._clock = clock
        self._lock = threading.RLock()
        self._devices: dict[str, Device] = {}
        self._pairing_codes: dict[str, float] = {}
        self._load()

    def issue_pairing_code(self) -> str:
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self._lock:
            now = self._clock()
            self._purge_codes(now)
            self._pairing_codes.clear()
            self._pairing_codes[_secret_hash(code)] = now + self.pairing_ttl
        return code

    def pair(self, code: str, device_name: str) -> PairingCredentials:
        code_hash = _secret_hash(str(code).strip())
        name = " ".join(str(device_name).split())[:80]
        if not name:
            raise PairingError("device name is required")
        with self._lock:
            now = self._clock()
            self._purge_codes(now)
            expiry = self._pairing_codes.pop(code_hash, None)
            if expiry is None or expiry < now:
                raise PairingError("invalid or expired pairing code")
            device_id = uuid.uuid4().hex
            token = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(32)
            self._devices[device_id] = Device(
                device_id=device_id,
                name=name,
                token_hash=_secret_hash(token),
                csrf_hash=_secret_hash(csrf),
                created_at=now,
                expires_at=now + self.token_ttl,
            )
            self._save()
        return PairingCredentials(device_id, token, csrf)

    def authenticate(self, token: str) -> Device | None:
        if not isinstance(token, str) or len(token) < 32:
            return None
        candidate = _secret_hash(token)
        with self._lock:
            for device in self._devices.values():
                if (
                    not device.revoked
                    and device.expires_at >= self._clock()
                    and hmac.compare_digest(device.token_hash, candidate)
                ):
                    return device
        return None

    @staticmethod
    def verify_csrf(device: Device, csrf_token: str) -> bool:
        if not isinstance(csrf_token, str) or len(csrf_token) < 32:
            return False
        return hmac.compare_digest(device.csrf_hash, _secret_hash(csrf_token))

    def list_devices(self) -> tuple[Device, ...]:
        with self._lock:
            return tuple(sorted(self._devices.values(), key=lambda item: item.created_at))

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            device = self._devices.get(device_id)
            if device is None or device.revoked:
                return False
            self._devices[device_id] = Device(
                device_id=device.device_id,
                name=device.name,
                token_hash=device.token_hash,
                csrf_hash=device.csrf_hash,
                created_at=device.created_at,
                expires_at=device.expires_at,
                revoked=True,
            )
            self._save()
            return True

    def _purge_codes(self, now: float) -> None:
        self._pairing_codes = {
            digest: expiry
            for digest, expiry in self._pairing_codes.items()
            if expiry >= now
        }

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.is_file():
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            devices = payload.get("devices", [])
            for item in devices:
                device = Device(
                    device_id=str(item["device_id"]),
                    name=str(item["name"]),
                    token_hash=str(item["token_hash"]),
                    csrf_hash=str(item["csrf_hash"]),
                    created_at=float(item["created_at"]),
                    expires_at=float(
                        item.get(
                            "expires_at",
                            float(item["created_at"]) + self.token_ttl,
                        )
                    ),
                    revoked=bool(item.get("revoked", False)),
                )
                self._devices[device.device_id] = device
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._devices.clear()

    def _save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "devices": [
                {
                    "device_id": item.device_id,
                    "name": item.name,
                    "token_hash": item.token_hash,
                    "csrf_hash": item.csrf_hash,
                    "created_at": item.created_at,
                    "expires_at": item.expires_at,
                    "revoked": item.revoked,
                }
                for item in self._devices.values()
            ],
        }
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_file)

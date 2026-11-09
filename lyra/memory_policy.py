"""Logical memory-admission accounting for local Lyra workloads."""

import threading


DEFAULT_CAPACITY_SLOTS = 2
DEFAULT_SLOT_BYTES = 4 * 1024**3


class MemoryBudgetExceeded(RuntimeError):
    """Raised when a lease cannot fit in the currently available slot budget."""


class MemoryLease:
    __slots__ = (
        "_pool",
        "_token",
        "_kind",
        "_slots",
        "_closed",
        "_close_lock",
    )

    def __init__(
        self,
        pool: "MemoryLeasePool",
        token: object,
        kind: str,
        slots: int,
    ) -> None:
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_slots", slots)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_close_lock", threading.Lock())

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MemoryLease attributes are read-only.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MemoryLease attributes are read-only.")

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def slots(self) -> int:
        return self._slots

    def close(self) -> None:
        with self._close_lock:
            if not self._closed:
                object.__setattr__(self, "_closed", True)
                self._pool._release(self._token)

    def __enter__(self) -> "MemoryLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


class MemoryLeasePool:
    def __init__(
        self,
        capacity_slots: int = DEFAULT_CAPACITY_SLOTS,
        slot_bytes: int = DEFAULT_SLOT_BYTES,
    ) -> None:
        if isinstance(capacity_slots, bool) or not isinstance(capacity_slots, int):
            raise TypeError("Pool capacity slots must be an integer.")
        if capacity_slots <= 0:
            raise ValueError("Pool capacity slots must be positive.")
        if isinstance(slot_bytes, bool) or not isinstance(slot_bytes, int):
            raise TypeError("Pool slot bytes must be an integer.")
        if slot_bytes <= 0:
            raise ValueError("Pool slot bytes must be positive.")
        self._capacity_slots = capacity_slots
        self._slot_bytes = slot_bytes
        self._used_slots = 0
        self._active_leases: dict[object, int] = {}
        self._lock = threading.Lock()

    @property
    def capacity_slots(self) -> int:
        with self._lock:
            return self._capacity_slots

    @property
    def slot_bytes(self) -> int:
        with self._lock:
            return self._slot_bytes

    @property
    def used_slots(self) -> int:
        with self._lock:
            return self._used_slots

    @property
    def available_slots(self) -> int:
        with self._lock:
            return self._capacity_slots - self._used_slots

    def acquire(self, kind: str, slots: int = 1) -> MemoryLease:
        if not isinstance(kind, str):
            raise TypeError("Lease kind must be a string.")
        if not kind.strip():
            raise ValueError("Lease kind must not be empty.")
        if isinstance(slots, bool) or not isinstance(slots, int):
            raise TypeError("Lease slots must be an integer.")
        if slots <= 0:
            raise ValueError("Lease slots must be positive.")
        if slots > self.capacity_slots:
            raise ValueError("Lease slots cannot exceed pool capacity.")
        token = object()
        with self._lock:
            if self._capacity_slots - self._used_slots < slots:
                raise MemoryBudgetExceeded(
                    "Not enough memory-admission slots are available."
                )
            self._active_leases[token] = slots
            self._used_slots += slots
        return MemoryLease(self, token, kind, slots)

    def _release(self, token: object) -> bool:
        with self._lock:
            slots = self._active_leases.get(token)
            if slots is None:
                return False
            updated_used_slots = self._used_slots - slots
            if not 0 <= updated_used_slots <= self._capacity_slots:
                raise RuntimeError("Memory lease accounting invariant violated.")
            del self._active_leases[token]
            self._used_slots = updated_used_slots
            return True

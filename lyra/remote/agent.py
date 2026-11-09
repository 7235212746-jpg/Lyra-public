from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from uuid import UUID


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def validate_job_id(value: str) -> UUID:
    try:
        result = UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Remote job ID is invalid.") from exc
    if value != str(result):
        raise ValueError("Remote job ID must be canonical.")
    return result


def resolve_runtime_zip(cache_root: str | Path, value: str) -> Path:
    root = Path(cache_root).resolve()
    candidate = Path(value).resolve()
    try:
        relative = candidate.relative_to(root / "runtime")
    except ValueError as exc:
        raise ValueError("Remote runtime is outside the Lyra cache.") from exc
    if len(relative.parts) != 2 or not _DIGEST.fullmatch(relative.parts[0]) or relative.parts[1] != "lyra-runtime.zip":
        raise ValueError("Remote runtime path is not content addressed.")
    if not candidate.is_file():
        raise ValueError("Remote runtime payload is missing.")
    return candidate


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        return 2
    job_id = validate_job_id(argv[0])
    cache_root = Path(__file__).resolve().parent
    job_directory = cache_root / "jobs" / str(job_id)
    cancelled_type = ()
    try:
        raw = (job_directory / "request.json").read_bytes()
        document = json.loads(raw.decode("utf-8"))
        runtime = resolve_runtime_zip(cache_root, document["options"]["runtime_zip"])
        sys.path.insert(0, str(runtime))
        from lyra.remote.protocol import (
            RemoteStatus,
            RemoteTaskState,
            decode_request,
            encode_message,
        )
        from lyra.remote.scientific_worker import RemoteTaskCancelled, execute
        cancelled_type = (RemoteTaskCancelled,)

        request = decode_request(raw)
        result = execute(request, job_directory, cache_root)
        (job_directory / "result.json.tmp").write_bytes(encode_message(result))
        (job_directory / "result.json.tmp").replace(job_directory / "result.json")
        status = RemoteStatus(job_id, 1_000_000, RemoteTaskState.SUCCEEDED, time.time(), None, 0)
        (job_directory / "status.json.tmp").write_bytes(encode_message(status))
        (job_directory / "status.json.tmp").replace(job_directory / "status.json")
        return 0
    except BaseException as exc:
        cancelled = isinstance(exc, cancelled_type)
        if not cancelled:
            _atomic_json(job_directory / "error.json", {"schema_version": 1, "error": f"{type(exc).__name__}: {exc}"})
        _atomic_json(
            job_directory / "status.json",
            {
                "schema_version": 1,
                "job_id": str(job_id),
                "sequence": 1_000_000,
                "state": "cancelled" if cancelled else "failed",
                "timestamp": time.time(),
                "progress": None,
                "exit_code": 130 if cancelled else 1,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

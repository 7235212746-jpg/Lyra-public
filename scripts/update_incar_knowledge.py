"""Explicitly refresh Lyra's offline INCAR tag index from the VASP Wiki API."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://vasp.at/wiki/api.php"
CATEGORY_URL = "https://vasp.at/wiki/Category:INCAR_tag"
DEFAULT_LICENSE_NAME = "GNU FDL 1.2"
DEFAULT_LICENSE_URL = "https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt"
USER_AGENT = "Lyra-INCAR-Knowledge/0.1 (offline index updater)"
MINIMUM_EXPECTED_TAGS = 600


def fetch_json(params: dict[str, str]) -> dict:
    query = dict(params)
    query.update({"format": "json", "formatversion": "2"})
    request = Request(
        f"{API_URL}?{urlencode(query)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=45) as response:
        payload = response.read()
    document = json.loads(payload.decode("utf-8"))
    if not isinstance(document, dict) or "error" in document:
        raise RuntimeError(f"VASP Wiki API returned an invalid response: {document!r}")
    return document


def fetch_official_index(
    fetcher: Callable[[dict[str, str]], dict] = fetch_json,
) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": "Category:INCAR tag",
        "gcmnamespace": "0",
        "gcmlimit": "max",
        "prop": "info|revisions",
        "rvprop": "ids|timestamp",
        "inprop": "url",
    }
    records: dict[str, dict] = {}
    while True:
        response = fetcher(params)
        pages = response.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            raise RuntimeError("VASP Wiki API did not return a page list.")
        for page in pages:
            title = str(page.get("title", "")).strip()
            revisions = page.get("revisions", [])
            revision = revisions[0] if isinstance(revisions, list) and revisions else {}
            url = str(page.get("canonicalurl") or page.get("fullurl") or "").strip()
            revision_id = revision.get("revid")
            revision_timestamp = revision.get("timestamp")
            if (
                not title
                or not url.startswith("https://vasp.at/wiki/")
                or isinstance(revision_id, bool)
                or not isinstance(revision_id, int)
                or revision_id <= 0
                or not isinstance(revision_timestamp, str)
                or not revision_timestamp
            ):
                raise RuntimeError(f"Incomplete VASP Wiki page metadata: {page!r}")
            # MediaWiki renders underscores in page titles as spaces. INCAR uses
            # underscores, while a few documented nested/plugin tags retain '/'.
            tag = title.replace(" ", "_").upper()
            records[tag] = {
                "tag": tag,
                "url": url,
                "revision_id": revision_id,
                "revision_timestamp": revision_timestamp,
            }
        continuation = response.get("continue")
        if not continuation:
            break
        if not isinstance(continuation, dict) or "gcmcontinue" not in continuation:
            raise RuntimeError("Unexpected VASP Wiki continuation response.")
        params = {**params, "gcmcontinue": str(continuation["gcmcontinue"])}
    return [records[tag] for tag in sorted(records)]


def fetch_license(
    fetcher: Callable[[dict[str, str]], dict] = fetch_json,
) -> tuple[str, str]:
    response = fetcher(
        {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "rightsinfo",
        }
    )
    rights = response.get("query", {}).get("rightsinfo", {})
    name = str(rights.get("text", "")).strip()
    url = str(rights.get("url", "")).strip().replace("http://", "https://", 1)
    if name != "GNU Free Documentation License 1.2" or not url:
        raise RuntimeError(f"Unexpected VASP Wiki license metadata: {rights!r}")
    return DEFAULT_LICENSE_NAME, url


def make_index_document(
    records: list[dict],
    *,
    fetched_at: str,
    license_name: str = DEFAULT_LICENSE_NAME,
    license_url: str = DEFAULT_LICENSE_URL,
) -> dict:
    return {
        "schema_version": 1,
        "dataset": {
            "source_url": CATEGORY_URL,
            "fetched_at": fetched_at,
            "license_name": license_name,
            "license_url": license_url,
            "content_note_zh": (
                "标签索引和页面修订元数据来自 VASP Wiki；中文知识正文由 Lyra 整理，"
                "不代表 VASP 官方翻译或官方建议。"
            ),
        },
        "records": sorted(records, key=lambda item: item["tag"]),
    }


def _validate_document(document: dict) -> None:
    records = document.get("records")
    if not isinstance(records, list) or len(records) < MINIMUM_EXPECTED_TAGS:
        count = len(records) if isinstance(records, list) else 0
        raise ValueError(
            f"Official INCAR index must contain at least {MINIMUM_EXPECTED_TAGS} "
            f"records; received {count}. Existing data was not replaced."
        )
    tags: set[str] = set()
    for record in records:
        tag = record.get("tag") if isinstance(record, dict) else None
        if not isinstance(tag, str) or not tag or tag in tags:
            raise ValueError(f"Invalid or duplicate INCAR tag: {tag!r}")
        tags.add(tag)
        if not str(record.get("url", "")).startswith("https://vasp.at/wiki/"):
            raise ValueError(f"Invalid VASP Wiki URL for {tag}.")
        if not isinstance(record.get("revision_id"), int):
            raise ValueError(f"Missing revision ID for {tag}.")
        if not record.get("revision_timestamp"):
            raise ValueError(f"Missing revision timestamp for {tag}.")


def write_index_atomic(document: dict, destination: str | Path) -> None:
    _validate_document(document)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _default_output() -> Path:
    return Path(__file__).resolve().parents[1] / "lyra" / "resources" / "incar" / "index.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly update Lyra's offline INCAR index from VASP Wiki."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="confirm that network access and replacement of the local index are intended",
    )
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="download and validate without replacing the local index",
    )
    args = parser.parse_args(argv)
    if not args.update:
        parser.error("network access is disabled by default; pass --update explicitly")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    records = fetch_official_index()
    license_name, license_url = fetch_license()
    document = make_index_document(
        records,
        fetched_at=fetched_at,
        license_name=license_name,
        license_url=license_url,
    )
    _validate_document(document)
    if not args.dry_run:
        write_index_atomic(document, args.output)
    print(
        f"Validated {len(records)} INCAR tags from VASP Wiki"
        + (" (dry run)." if args.dry_run else f" and wrote {args.output}.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

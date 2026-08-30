"""Dataset SHA256 integrity verification (Pillar IV / hard constraint 3).

- Only verifies the frozen toolkit's catalog.jsonl / public_set.jsonl.
- Never downloads any upstream full raw Amazon Reviews data.
- A failed check raises by default; with SKIP_DATA_VERIFY=1 or an explicit env skip, it only warns.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from config import constants


def sha256_of(path: str | Path) -> str:
    """Compute a file's SHA256 in chunks (large-file friendly)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_file(path: str | Path, expected: str, label: str, skip: bool = False) -> bool:
    """Verify one file: True on pass; on failure warns or raises per skip."""
    path = Path(path)
    if not path.exists():
        msg = f"[data_verify] missing data file {label}: {path}"
        if skip:
            print(f"[WARN] {msg}", file=sys.stderr)
            return False
        raise FileNotFoundError(msg)
    actual = sha256_of(path)
    ok = actual == expected.upper()
    if ok:
        print(f"[data_verify] {label}: SHA256 OK ({path.name})")
    else:
        msg = (
            f"[data_verify] {label} SHA256 mismatch!\n"
            f"  expected: {expected.upper()}\n  actual  : {actual}\n"
            f"  path: {path} (if you use a different official release, override via SKIP_DATA_VERIFY=1 or env vars)"  # noqa: E501
        )
        if skip:
            print(f"[WARN] {msg}", file=sys.stderr)
        else:
            raise ValueError(msg)
    return ok


def verify_dataset(skip: bool = False) -> bool:
    """Verify the two core frozen-toolkit files plus a sanity check on row counts."""
    ok_cat = verify_file(
        constants.CATALOG_PATH,
        constants.EXPECTED_SHA256_CATALOG,
        "catalog.jsonl",
        skip=skip,
    )
    ok_pub = verify_file(
        constants.PUBLIC_SET_PATH,
        constants.EXPECTED_SHA256_PUBLIC_SET,
        "public_set.jsonl",
        skip=skip,
    )
    # row-count spot check (informational only, not a hard gate)
    if ok_cat:
        rows = sum(1 for _ in open(constants.CATALOG_PATH, encoding="utf-8"))
        if rows != constants.CATALOG_EXPECTED_ROWS:
            print(
                f"[WARN] catalog row count {rows} != expected {constants.CATALOG_EXPECTED_ROWS}",
                file=sys.stderr,
            )
    return ok_cat and ok_pub

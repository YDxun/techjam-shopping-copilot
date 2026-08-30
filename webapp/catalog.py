from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


class CatalogError(ValueError):
    """The selected presentation catalog is not safe to serve."""


class CatalogPresenter:
    def __init__(self, path: Path, offsets: dict[str, int]) -> None:
        self.path = path
        self._offsets = offsets

    @classmethod
    def build(cls, path: Path) -> "CatalogPresenter":
        offsets: dict[str, int] = {}
        try:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    row = json.loads(raw.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise CatalogError("catalog row must be a JSON object")
                    asin = str(row.get("parent_asin", "")).strip()
                    if not asin or asin in offsets:
                        raise CatalogError("catalog has a missing or duplicate parent_asin")
                    offsets[asin] = offset
        except CatalogError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("catalog cannot be indexed") from exc
        return cls(path, offsets)

    def summaries(self, asins: Sequence[str]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for asin in dict.fromkeys(asins):
            row = self._read(asin)
            if row is not None:
                result[asin] = self._summary(row)
        return result

    def detail(self, parent_asin: str) -> dict[str, object] | None:
        row = self._read(parent_asin)
        if row is None:
            return None
        return {key: row.get(key) for key in (
            "parent_asin", "title", "price", "average_rating", "rating_number",
            "store", "categories", "features", "description", "details",
        )}

    def _read(self, asin: str) -> dict[str, object] | None:
        offset = self._offsets.get(asin)
        if offset is None:
            return None
        try:
            with self.path.open("rb") as handle:
                handle.seek(offset)
                row = json.loads(handle.readline().decode("utf-8"))
                if not isinstance(row, dict):
                    raise CatalogError("indexed catalog record is not a JSON object")
                return row
        except CatalogError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("indexed catalog record cannot be read") from exc

    @staticmethod
    def _summary(row: dict[str, object]) -> dict[str, object]:
        categories = [str(value) for value in row.get("categories") or []]
        features = [str(value) for value in row.get("features") or []]
        return {
            "parent_asin": str(row["parent_asin"]),
            "title": str(row.get("title") or "Untitled product"),
            "price": row.get("price"),
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "store": str(row.get("store") or ""),
            "categories": categories[-2:],
            "features": features[:2],
        }

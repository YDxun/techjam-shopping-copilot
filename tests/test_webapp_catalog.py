import json
from pathlib import Path

import pytest

from webapp.catalog import CatalogError, CatalogPresenter


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_catalog_returns_order_independent_summary_mapping_and_full_detail(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    write_rows(path, [
        {
            "parent_asin": "A1",
            "title": "Café cotton shirt",
            "price": None,
            "average_rating": 4.6,
            "rating_number": 42,
            "store": "Demo",
            "categories": ["Clothing", "Men", "Shirts"],
            "features": ["Cotton", "Machine washable", "Regular fit"],
            "description": ["A lightweight shirt."],
            "details": {"Department": "mens"},
        },
        {"parent_asin": "A2", "title": "Trail jacket", "features": []},
    ])

    catalog = CatalogPresenter.build(path)
    summaries = catalog.summaries(["A2", "missing", "A1"])

    assert list(summaries) == ["A2", "A1"]
    assert summaries["A1"]["features"] == ["Cotton", "Machine washable"]
    assert summaries["A1"]["categories"] == ["Men", "Shirts"]
    assert summaries["A1"]["price"] is None
    assert catalog.detail("A1")["description"] == ["A lightweight shirt."]
    assert catalog.detail("missing") is None


@pytest.mark.parametrize(
    "rows",
    [
        [{"title": "missing id"}],
        [{"parent_asin": None}],
        [{"parent_asin": 123}],
        [{"parent_asin": "A1"}, {"parent_asin": "A1"}],
    ],
)
def test_catalog_rejects_missing_and_duplicate_asins(tmp_path: Path, rows: list[dict]) -> None:
    path = tmp_path / "catalog.jsonl"
    write_rows(path, rows)
    with pytest.raises(CatalogError):
        CatalogPresenter.build(path)


def test_catalog_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        CatalogPresenter.build(path)

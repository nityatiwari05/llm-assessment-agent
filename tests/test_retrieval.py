import json
from pathlib import Path

import pytest

from app.catalog import Catalog, load_catalog
from app.retrieval import Retriever

SAMPLE = [
    {
        "entity_id": "1",
        "name": "Core Java (Advanced Level) (New)",
        "link": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "job_levels": ["Mid-Professional", "Professional Individual Contributor"],
        "languages": ["English (USA)"],
        "duration": "13 minutes",
        "status": "ok",
        "remote": "yes",
        "adaptive": "no",
        "description": "Multi-choice test that measures advanced Java concepts like generics, collections, threads.",
        "keys": ["Knowledge & Skills"],
    },
    {
        "entity_id": "2",
        "name": "Occupational Personality Questionnaire OPQ32r",
        "link": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "job_levels": ["Manager", "Director"],
        "languages": ["English International"],
        "duration": "25 minutes",
        "status": "ok",
        "remote": "yes",
        "adaptive": "no",
        "description": "Measures workplace behavioural style across 32 dimensions.",
        "keys": ["Personality & Behavior"],
    },
    {
        "entity_id": "3",
        "name": "SQL (New)",
        "link": "https://www.shl.com/products/product-catalog/view/sql-new/",
        "job_levels": ["Mid-Professional"],
        "languages": ["English (USA)"],
        "duration": "9 minutes",
        "status": "ok",
        "remote": "yes",
        "adaptive": "no",
        "description": "Measures knowledge of SQL queries, data manipulation and transaction processing.",
        "keys": ["Knowledge & Skills"],
    },
]


@pytest.fixture()
def sample_catalog_path(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return p


def test_load_catalog_derives_test_type(sample_catalog_path: Path):
    items = load_catalog(sample_catalog_path)
    assert len(items) == 3
    java = next(i for i in items if "Java" in i.name)
    assert java.test_type == "K"
    opq = next(i for i in items if "OPQ" in i.name)
    assert opq.test_type == "P"


def test_catalog_dedupes_and_indexes(sample_catalog_path: Path):
    catalog = Catalog(sample_catalog_path)
    assert catalog.get("1").name == "Core Java (Advanced Level) (New)"
    assert catalog.get("nonexistent") is None


def test_retrieval_finds_relevant_item(sample_catalog_path: Path):
    catalog = Catalog(sample_catalog_path)
    retriever = Retriever(catalog)
    results = retriever.search("Java developer generics collections", top_k=3)
    assert results
    top_item, _score = results[0]
    assert "Java" in top_item.name


def test_retrieval_empty_query_returns_empty(sample_catalog_path: Path):
    catalog = Catalog(sample_catalog_path)
    retriever = Retriever(catalog)
    assert retriever.search("   ", top_k=5) == []


def test_fuzzy_name_match_for_compare(sample_catalog_path: Path):
    catalog = Catalog(sample_catalog_path)
    matches = catalog.find_by_name_fuzzy("SQL (New)")
    assert len(matches) == 1
    assert matches[0].entity_id == "3"

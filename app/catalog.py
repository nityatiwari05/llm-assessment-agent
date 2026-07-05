"""
Loads data/catalog.json (the scraped SHL product catalog, restricted to Individual
Test Solutions) and normalizes it into a canonical in-memory structure used by
retrieval and the agent.

Expected raw record shape (matches the assignment's scrape output):
{
  "entity_id": "4084",
  "name": "Java 8 (New)",
  "link": "https://www.shl.com/products/product-catalog/view/java-8-new/",
  "job_levels": [...],
  "languages": [...],
  "duration": "18 minutes",
  "status": "ok",
  "remote": "yes",
  "adaptive": "no",
  "description": "...",
  "keys": ["Knowledge & Skills"]
}

We additionally derive a `test_type` field: the single-letter SHL taxonomy codes
(A/B/C/D/E/K/P/S) computed from `keys`, since the API response and several eval
probes reference test_type.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.config import CATALOG_PATH

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Names/name-fragments that identify pre-packaged "Job Solutions" bundles rather
# than Individual Test Solutions. The assignment scopes the catalog to Individual
# Test Solutions only; if your scrape source already filters this, this list is a
# no-op safety net. Edit freely — it's intentionally conservative (exact bundle
# families only), since several traces legitimately use "Solution"-named products
# that behave like individual assessments in this dataset (e.g. entry-level
# behavioral solutions used as-is in SHL's Individual Test Solutions catalog).
EXCLUDE_NAME_PATTERNS: List[str] = [
    # Intentionally empty by default — see README "Catalog scoping" section.
    # Populate this if your scrape mixes in true multi-assessment Job Solutions
    # pages (distinguishable on shl.com by a "Job Solutions" breadcrumb) that
    # you want excluded from retrieval.
]


@dataclass
class CatalogItem:
    entity_id: str
    name: str
    url: str
    description: str
    keys: List[str]
    test_type: str  # e.g. "K" or "K,S"
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration: str = ""
    adaptive: str = "no"
    remote: str = "yes"

    def search_text(self) -> str:
        """Text blob used for lexical retrieval (BM25)."""
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            self.test_type,
        ]
        return " ".join(p for p in parts if p)

    def to_public_dict(self) -> Dict:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


def _derive_test_type(keys: List[str]) -> str:
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k.strip())
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def _is_excluded(name: str) -> bool:
    return any(pat.lower() in name.lower() for pat in EXCLUDE_NAME_PATTERNS)


def load_catalog(path: Optional[Path] = None) -> List[CatalogItem]:
    path = path or CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {path}. Place your scraped SHL catalog "
            f"JSON there (see README 'Getting the catalog data'), or set the "
            f"CATALOG_PATH env var to point at it."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[CatalogItem] = []
    seen_ids = set()
    for rec in raw:
        # Skip records that failed to scrape.
        if rec.get("status") and rec["status"] != "ok":
            continue
        name = (rec.get("name") or "").strip()
        # Defensive cleanup: some scrapes leave stray newlines/whitespace in names.
        name = re.sub(r"\s+", " ", name).strip()
        if not name or _is_excluded(name):
            continue

        entity_id = str(rec.get("entity_id") or rec.get("id") or "")
        url = (rec.get("link") or rec.get("url") or "").strip()
        if not url:
            continue

        # De-dupe on entity_id (fall back to URL) — some catalog dumps repeat rows.
        dedupe_key = entity_id or url
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)

        keys = rec.get("keys") or []
        item = CatalogItem(
            entity_id=entity_id or url,
            name=name,
            url=url,
            description=(rec.get("description") or "").strip(),
            keys=keys,
            test_type=_derive_test_type(keys),
            job_levels=rec.get("job_levels") or [],
            languages=rec.get("languages") or [],
            duration=(rec.get("duration") or "").strip(),
            adaptive=(rec.get("adaptive") or "no").strip(),
            remote=(rec.get("remote") or "yes").strip(),
        )
        items.append(item)

    if not items:
        raise ValueError(
            f"Loaded {path} but found zero usable catalog rows. Check the file's "
            f"shape against the docstring in app/catalog.py."
        )
    return items


class Catalog:
    """Thin in-memory index wrapper, loaded once at process startup."""

    def __init__(self, path: Optional[Path] = None):
        self.items: List[CatalogItem] = load_catalog(path)
        self.by_id: Dict[str, CatalogItem] = {i.entity_id: i for i in self.items}
        self.by_name_lower: Dict[str, CatalogItem] = {i.name.lower(): i for i in self.items}

    def get(self, entity_id: str) -> Optional[CatalogItem]:
        return self.by_id.get(entity_id)

    def find_by_name_fuzzy(self, query: str, limit: int = 5) -> List[CatalogItem]:
        """Cheap substring match used to force-include named entities (e.g. for
        'what's the difference between OPQ and GSA' comparison requests) even if
        BM25 ranks them lower."""
        q = query.lower().strip()
        if len(q) < 2:
            return []
        exact = [i for i in self.items if i.name.lower() == q]
        if exact:
            return exact
        contains = [i for i in self.items if q in i.name.lower()]
        return contains[:limit]


_catalog_singleton: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton

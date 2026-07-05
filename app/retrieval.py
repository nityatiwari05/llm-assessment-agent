"""
Retrieval layer. Deliberately lexical (BM25) rather than embedding-based:

  - No model download needed at build/deploy time (works on free tiers with no
    GPU and no access to a model hub), which matters since assessment names are
    short, precise skill/product tokens ("Core Java", "OPQ32r", "SVAR") that BM25
    handles very well — this is a keyword-heavy domain, not a paraphrase-heavy one.
  - Deterministic and fast (<10ms for ~500 items), which matters under the 30s
    per-call budget.

If you want to swap in embeddings later, implement the same `search()` interface
and swap the import in app/agent.py.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.catalog import Catalog, CatalogItem

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# A handful of domain synonyms so that JD language maps onto SHL's product
# vocabulary. Extend this as you observe retrieval misses in eval.
#
# The leadership/executive/director/cxo entries below exist because of a real
# miss found during evaluation: a "senior leadership, CXO, 15+ years, selection
# against a benchmark" query scored "Enterprise Leadership Report 1.0/2.0" far
# above "Occupational Personality Questionnaire OPQ32r" and its Leadership/UCF
# report pairing, purely because the Enterprise Leadership Report's name and
# description repeat the literal word "leadership" while OPQ32r's description
# talks about "workplace behavioural style" and never uses that word at all.
# Lexical retrieval has no notion that OPQ32r is SHL's standard instrument for
# almost every senior/selection scenario unless that's spelled out somewhere in
# the corpus, so we spell it out here rather than in the LLM prompt, since a
# prompt only helps once an item is already in the candidate pool, whereas the
# retrieval score decides whether it gets into the pool at all.
SYNONYMS = {
    "js": "javascript",
    "reactjs": "react",
    "node": "nodejs",
    "postgres": "sql",
    "mysql": "sql",
    "aws": "amazon web services aws",
    "k8s": "kubernetes",
    "oop": "object oriented programming",
    "personality": "personality behavior opq",
    "cognitive": "ability aptitude reasoning verify",
    "sjt": "situational judgment biodata scenarios",
    "leadership": "leadership opq occupational personality questionnaire competency",
    "executive": "executive opq occupational personality questionnaire leadership",
    "director": "director opq occupational personality questionnaire leadership",
    "cxo": "executive director opq occupational personality questionnaire",
    "benchmark": "benchmark competency norm opq",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in SYNONYMS:
            expanded.extend(_TOKEN_RE.findall(SYNONYMS[t]))
    return expanded


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.corpus_tokens = [_tokenize(i.search_text()) for i in catalog.items]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 25) -> List[Tuple[CatalogItem, float]]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self.catalog.items, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(item, score) for item, score in ranked[:top_k] if score > 0]

    def search_multi(
        self, queries: List[str], top_k: int = 25
    ) -> List[Tuple[CatalogItem, float]]:
        """Union of per-query top hits, useful when a conversation covers several
        distinct constraints (e.g. 'Java' + 'situational judgement' + 'graduate')
        that a single bag-of-words query would blur together."""
        best: dict[str, float] = {}
        item_by_id = {}
        for q in queries:
            for item, score in self.search(q, top_k=top_k):
                item_by_id[item.entity_id] = item
                best[item.entity_id] = max(best.get(item.entity_id, 0.0), score)
        ranked_ids = sorted(best, key=lambda eid: best[eid], reverse=True)
        return [(item_by_id[eid], best[eid]) for eid in ranked_ids[:top_k]]


_retriever_singleton: "Retriever | None" = None


def get_retriever(catalog: Catalog) -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever(catalog)
    return _retriever_singleton
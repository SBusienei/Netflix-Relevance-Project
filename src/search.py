"""Compare lexical, semantic, fused, and reranked retrieval without inventing relevance."""

from typing import Literal, TypedDict

import numpy as np

from src.index import Index, load_ranker, tokens

Mode = Literal["bm25", "vector", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid", "rerank")
RRF_C = 60


class Hit(TypedDict):
    show_id: str
    score: float


def hits(index: Index, scores: dict[int, float], k: int) -> list[Hit]:
    order = sorted(scores, key=lambda row: (-scores[row], row))[:k]
    return [{"show_id": index.records[row]["show_id"], "score": scores[row]} for row in order]


def bm25_search(index: Index, query: str, k: int) -> list[Hit]:
    scores = index.bm25.get_scores(tokens(query))
    return hits(index, {row: float(score) for row, score in enumerate(scores) if score > 0}, k)


def vector_search(index: Index, query: str, k: int) -> list[Hit]:
    vector = np.asarray(index.encoder.encode([query], normalize_embeddings=True), dtype=np.float32)
    scores, rows = index.ann.search(vector, min(k, len(index.records)))
    return hits(index, {int(row): float(score) for row, score in zip(rows[0], scores[0]) if row >= 0}, k)


def hybrid(index: Index, query: str, k: int, depth: int) -> list[Hit]:
    scores: dict[str, float] = {}
    for ranking in (bm25_search(index, query, depth), vector_search(index, query, depth)):
        for rank, hit in enumerate(ranking, 1):
            scores[hit["show_id"]] = scores.get(hit["show_id"], 0) + 1 / (RRF_C + rank)
    return hits(index, {row: scores[record["show_id"]] for row, record in enumerate(index.records)
                        if record["show_id"] in scores}, k)


def rerank(index: Index, query: str, candidates: list[Hit], k: int) -> list[Hit]:
    if not candidates:
        return []
    ids = [hit["show_id"] for hit in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Reranking candidates must have unique IDs")
    rows = {record["show_id"]: row for row, record in enumerate(index.records)}
    if any(show_id not in rows for show_id in ids):
        raise ValueError("Unknown reranking candidate")
    if index.ranker is None:
        index.ranker = load_ranker()
    scores = index.ranker.predict([(query, index.texts[rows[show_id]]) for show_id in ids])
    if len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Reranker returned invalid scores")
    return hits(index, {rows[show_id]: float(score) for show_id, score in zip(ids, scores)}, k)


def validate_query(query: str) -> None:
    if not query.strip() or len(query) > 500 or not tokens(query):
        raise ValueError("Supply a query containing words, at most 500 characters")


def search(index: Index, query: str, mode: Mode, k: int = 10, depth: int = 50) -> list[Hit]:
    validate_query(query)
    if mode not in MODES or type(k) is not int or type(depth) is not int or not 1 <= k <= depth <= 100:
        raise ValueError("Choose a valid mode and 1 <= k <= candidate depth <= 100")
    with index.lock:
        if mode == "bm25":
            return bm25_search(index, query, k)
        if mode == "vector":
            return vector_search(index, query, k)
        candidates = hybrid(index, query, depth if mode == "rerank" else k, depth)
        return rerank(index, query, candidates, k) if mode == "rerank" else candidates

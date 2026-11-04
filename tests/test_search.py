"""Check candidate selection, rank fusion, reranking, and invalid-input boundaries."""

from pathlib import Path

import numpy as np
import pytest

from src.index import Index, document, tokens
from src.search import MODES, bm25_search, hybrid, rerank, search, vector_search


def test_shared_fields_and_tokenization(small_index: Index) -> None:
    assert tokens("ORBIT,") == ["orbit"]
    assert "title: Orbit" in document(small_index.records[0])
    assert "country:" not in document(small_index.records[0])


def test_bm25_and_real_hnsw(small_index: Index) -> None:
    assert bm25_search(small_index, "orbit", 10)[0]["show_id"] == "s1"
    assert bm25_search(small_index, "unmatchableword", 10) == []
    vector = vector_search(small_index, "orbit", 10)
    assert len(vector) == 3 and vector[0] == {"show_id": "s1", "score": 1.0}


def test_rrf_uses_ranks_not_score_scales(small_index: Index) -> None:
    fused = hybrid(small_index, "orbit", 3, 3)
    # s1 is rank 1 in both lists: 1/(60+1) + 1/(60+1) = 2/61.
    assert fused[0]["score"] == pytest.approx(2 / 61)
    assert len({hit["show_id"] for hit in fused}) == len(fused)
    reranked = search(small_index, "orbit", "rerank", 2, 3)
    assert reranked[0]["show_id"] == "s2"
    assert {hit["show_id"] for hit in reranked} <= {hit["show_id"] for hit in fused}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("query,k,depth", [("", 1, 3), ("!!!", 1, 3), ("x" * 501, 1, 3),
                                          ("orbit", 4, 3), ("orbit", 0, 3), ("orbit", 1, 101)])
def test_invalid_search_is_explicit(small_index: Index, mode: str, query: str, k: int, depth: int) -> None:
    with pytest.raises(ValueError):
        search(small_index, query, mode, k, depth)


def test_rerank_rejects_unknown_and_duplicate_candidates(small_index: Index) -> None:
    for ids in (["unknown"], ["s1", "s1"]):
        with pytest.raises(ValueError):
            rerank(small_index, "orbit", [{"show_id": show_id, "score": 1.0} for show_id in ids], 1)


def test_disk_embeddings_are_reused_and_bound_to_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.index

    source = tmp_path / "corpus.csv"
    source.write_text("first", encoding="utf-8")
    monkeypatch.setattr(src.index, "DATA_PATH", source)
    calls = []

    class Encoder:
        def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
            calls.append(sentences)
            values = np.zeros((len(sentences), 384), dtype=np.float32)
            values[:, 0] = 1
            return values

    encoder = Encoder()
    first = src.index.embeddings(encoder, ["Orbit"])
    assert np.array_equal(first, src.index.embeddings(encoder, ["Orbit"])) and len(calls) == 1
    old_files = set((tmp_path / ".hybrid_cache").glob("*.npy"))
    source.write_text("changed", encoding="utf-8")
    src.index.embeddings(encoder, ["Orbit"])
    assert len(calls) == 2
    target = next(iter(set((tmp_path / ".hybrid_cache").glob("*.npy")) - old_files))
    np.save(target, np.zeros((1, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid embeddings"):
        src.index.embeddings(encoder, ["Orbit"])

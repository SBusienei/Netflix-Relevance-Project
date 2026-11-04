"""Supply artificial vectors and records so unit tests never download models or label films."""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.data import FIELDS
from src.index import Index, document, tokens


class Encoder:
    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[1, 0] if "orbit" in text.casefold() else [0, 1]
                         for text in sentences], dtype=np.float32)


class Ranker:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.array([2 if "Kitchen" in text else 1 for _, text in pairs])


@pytest.fixture
def small_index() -> Index:
    import faiss

    records = [dict.fromkeys(FIELDS, "") | {
        "show_id": f"s{number}", "title": title, "description": description,
    } for number, (title, description) in enumerate([
        ("Orbit", "A stranded space crew."), ("Kitchen", "A cooking competition."),
        ("Forest", "Wild animals in nature."),
    ], 1)]
    texts = [document(row) for row in records]
    ann = faiss.IndexHNSWFlat(2, 8, faiss.METRIC_INNER_PRODUCT)
    ann.add(np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32))
    return Index(records, texts, Encoder(), ann,
                 BM25Okapi([tokens(row["description"] + " " + row["title"]) for row in records]), ranker=Ranker())

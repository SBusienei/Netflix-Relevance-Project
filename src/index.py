"""Build lexical and approximate vector indexes over the same audited film text."""

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from src.data import DATA_PATH, FilmRecord, load_records

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
FIELDS = ("title", "description", "listed_in", "cast", "director", "country")
HNSW_M, EF_CONSTRUCTION, EF_SEARCH = 32, 80, 64
EMBED_MAXLEN, RERANK_MAXLEN = 256, 512
BM25_SETTINGS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}
BUILD_LOCK = RLock()


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> NDArray[np.float32]: ...


class ANN(Protocol):
    def search(self, vectors: NDArray[np.float32], k: int) -> tuple[NDArray, NDArray]: ...


class Ranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> NDArray: ...


@dataclass
class Index:
    records: list[FilmRecord]
    texts: list[str]
    encoder: Encoder
    ann: ANN
    bm25: BM25Okapi
    lock: RLock = field(default_factory=RLock)
    ranker: Ranker | None = None


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def document(record: FilmRecord) -> str:
    return "\n".join(f"{name}: {record[name]}" for name in FIELDS if record[name])


def validate_vectors(vectors: NDArray, count: int) -> None:
    if (vectors.shape != (count, 384) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)):
        raise ValueError("Invalid embeddings/cache; remove a corrupt named .npy file and rebuild")


def embeddings(encoder: Encoder, texts: list[str]) -> NDArray[np.float32]:
    key = hashlib.sha256(DATA_PATH.read_bytes() +
                         repr((EMBED_REVISION, FIELDS, EMBED_MAXLEN, "documents-v1")).encode()).hexdigest()
    directory = DATA_PATH.parent / ".hybrid_cache"
    directory.mkdir(exist_ok=True)
    target = directory / f"{key}.npy"
    if target.exists():
        vectors = np.load(target, allow_pickle=False)
    else:
        vectors = np.asarray(encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)
        validate_vectors(vectors, len(texts))
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as temporary:
            name = temporary.name
        try:
            np.save(name, vectors, allow_pickle=False)
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    validate_vectors(vectors, len(texts))
    return vectors


@lru_cache(maxsize=1)
def _build_index() -> Index:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    faiss.omp_set_num_threads(1)
    records, _ = load_records(DATA_PATH)
    texts = [document(record) for record in records]
    encoder = SentenceTransformer(EMBEDDER, revision=EMBED_REVISION, device="cpu")
    encoder.max_seq_length = EMBED_MAXLEN
    vectors = embeddings(encoder, texts)
    ann = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction, ann.hnsw.efSearch = EF_CONSTRUCTION, EF_SEARCH
    ann.add(vectors)
    lexical = [" ".join(record[name] for name in FIELDS) for record in records]
    return Index(records, texts, encoder, ann, BM25Okapi([tokens(text) for text in lexical], **BM25_SETTINGS))


def build_index() -> Index:
    with BUILD_LOCK:
        return _build_index()


def load_ranker() -> Ranker:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER, revision=RERANK_REVISION, device="cpu", max_length=RERANK_MAXLEN)

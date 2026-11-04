"""Bind human judgments and blind repeat passes to a query split frozen before tuning."""

import csv
import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from src.data import FIELDS, load_records, normalize_text


LABEL_COLUMNS = (
    "query_id", "show_id", "grade", "timestamp", "pass_id",
    "split_sha256", "corpus_sha256", "pool_sha256",
)
Pair = tuple[str, str]
LabelKey = tuple[str, str, str]


@dataclass(frozen=True)
class Pool:
    kind: str
    pass_id: str
    partition: str
    split_sha256: str
    corpus_sha256: str
    source_pool_sha256: str


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path}: expected CSV columns {', '.join(columns)}")
        rows: list[dict[str, str]] = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV record {number}")
            rows.append({column: row[column] for column in columns})
    return rows


def write_json(path: Path, value: object) -> None:
    # Exclusive creation protects frozen splits and sampled queues from replacement.
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=True)
        target.write("\n")


def read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def freeze_queries(
    source: Path, output: Path, corpus: Path, heldout_fraction: float = 0.2, seed: int = 42,
) -> None:
    if not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise ValueError("Held-out fraction must be strictly between zero and one")
    rows = read_table(source, ("query_id", "query"))
    if len(rows) < 2:
        raise ValueError("Supply at least two human-authored queries to split")
    ids: set[str] = set()
    texts: set[str] = set()
    for row in rows:
        row["query_id"] = row["query_id"].strip()
        row["query"] = normalize_text(row["query"])
        if not row["query_id"] or not row["query"]:
            raise ValueError("Query IDs and query text must not be blank")
        if row["query_id"] in ids or row["query"].casefold() in texts:
            raise ValueError("Duplicate query ID or text could leak across the query split")
        ids.add(row["query_id"])
        texts.add(row["query"].casefold())
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    count = min(len(rows) - 1, max(1, math.ceil(len(rows) * heldout_fraction)))
    heldout = set(shuffled[:count])
    write_json(output, {
        "version": 1, "source_sha256": fingerprint(source),
        "corpus_sha256": fingerprint(corpus), "seed": seed,
        "heldout_fraction": heldout_fraction,
        "queries": [
            row | {"split": "heldout" if row["query_id"] in heldout else "dev"}
            for row in sorted(rows, key=lambda row: row["query_id"])
        ],
    })


def load_queries(path: Path, corpus: Path) -> dict[str, dict[str, str]]:
    frozen = read_json(path)
    if frozen.get("version") != 1 or frozen.get("corpus_sha256") != fingerprint(corpus):
        raise ValueError("Query split version or corpus fingerprint does not match")
    rows = frozen.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen split contains no queries")
    queries: dict[str, dict[str, str]] = {}
    texts: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id", "query", "split"}:
            raise ValueError("Invalid frozen query schema")
        query_id, text, partition = row["query_id"], row["query"], row["split"]
        if not isinstance(query_id, str) or not isinstance(text, str) or not isinstance(partition, str):
            raise ValueError("Frozen query values must be strings")
        if not query_id.strip() or not text.strip() or not partition.strip():
            raise ValueError("Frozen query values must be nonempty strings")
        if query_id in queries or text.casefold() in texts:
            raise ValueError("Duplicate query ID or text in frozen split")
        if partition not in ("dev", "heldout"):
            raise ValueError("Invalid query partition")
        queries[query_id] = {"query_id": query_id, "query": text, "split": partition}
        texts.add(text.casefold())
    if {row["split"] for row in queries.values()} != {"dev", "heldout"}:
        raise ValueError("Frozen split must contain development and held-out queries")
    return queries


def load_pool(
    path: Path, split_file: Path, corpus: Path,
) -> tuple[Pool, list[Pair], dict[str, dict[str, str]]]:
    queries = load_queries(split_file, corpus)
    pool = read_json(path)
    if (pool.get("version") != 1
            or pool.get("split_sha256") != fingerprint(split_file)
            or pool.get("corpus_sha256") != fingerprint(corpus)):
        raise ValueError("Pool does not match the frozen query split and corpus")
    kind = pool.get("kind")
    if not isinstance(kind, str) or kind not in {"initial", "relabel"}:
        raise ValueError("Unknown pool kind")
    pass_id = pool.get("pass_id")
    if not isinstance(pass_id, str) or not pass_id.strip():
        raise ValueError("Pool is missing its pass ID")
    if (kind == "initial") != (pass_id == "initial"):
        raise ValueError("Pool kind and pass ID disagree")
    source_hash = pool.get("source_pool_sha256", "")
    if not isinstance(source_hash, str):
        raise ValueError("Original pool fingerprint must be a string")
    if kind == "relabel" and (len(source_hash) != 64 or source_hash == fingerprint(path)):
        raise ValueError("Re-label pool is missing its original pool fingerprint")
    partition = pool.get("partition")
    if not isinstance(partition, str) or partition not in {"dev", "heldout"}:
        raise ValueError("Pool is missing its query partition")
    rows = pool.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Pool contains no query-record pairs")
    records, _ = load_records(corpus)
    record_ids = {row["show_id"] for row in records}
    pairs: list[Pair] = []
    seen: set[Pair] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id", "show_id"}:
            raise ValueError("Pool pairs must contain only query_id and show_id, never grades")
        query_id, show_id = row["query_id"], row["show_id"]
        if not isinstance(query_id, str) or not isinstance(show_id, str):
            raise ValueError("Pool identifiers must be strings")
        if query_id not in queries or show_id not in record_ids:
            raise ValueError("Pool refers to an unknown query or record")
        if queries[query_id]["split"] != partition:
            raise ValueError("Pool mixes development and held-out queries")
        pair = (query_id, show_id)
        if pair in seen:
            raise ValueError("Pool contains duplicate query-record pairs")
        seen.add(pair)
        pairs.append(pair)
    return Pool(
        kind=kind, pass_id=pass_id, partition=partition,
        split_sha256=fingerprint(split_file), corpus_sha256=fingerprint(corpus),
        source_pool_sha256=source_hash,
    ), pairs, queries


def load_labels(
    path: Path, split_file: Path, corpus: Path,
) -> dict[LabelKey, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[LabelKey, dict[str, str]] = {}
    queries = load_queries(split_file, corpus)
    records, _ = load_records(corpus)
    record_ids = {row["show_id"] for row in records}
    split_hash, corpus_hash = fingerprint(split_file), fingerprint(corpus)
    for row in read_table(path, LABEL_COLUMNS):
        if row["grade"] not in {"0", "1", "2"}:
            raise ValueError("Judgment CSV contains a grade outside 0-2")
        if row["split_sha256"] != split_hash or row["corpus_sha256"] != corpus_hash:
            raise ValueError("Judgment CSV belongs to a different split or corpus")
        if row["query_id"] not in queries or row["show_id"] not in record_ids:
            raise ValueError("Judgment CSV contains an unknown query or record")
        if any(not row[name].strip() for name in LABEL_COLUMNS):
            raise ValueError("Judgment CSV contains an empty required value")
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp.utcoffset() is None:
            raise ValueError("Judgment timestamps must include a timezone")
        if len(row["pool_sha256"]) != 64:
            raise ValueError("Invalid pool fingerprint in judgment CSV")
        key = (row["pass_id"], row["query_id"], row["show_id"])
        if key in labels:
            raise ValueError("Duplicate grade for the same pair and pass")
        labels[key] = row
    return labels


def require_labels(
    labels: dict[LabelKey, dict[str, str]], pairs: list[Pair], pass_id: str, pool_hash: str,
) -> list[int]:
    missing = [pair for pair in pairs if (pass_id, *pair) not in labels]
    if missing:
        raise ValueError(
            f"Missing human labels: {len(missing)} pair(s) in pass {pass_id}. "
            "Complete the labelling pass; no grades will be inferred."
        )
    rows = [labels[(pass_id, *pair)] for pair in pairs]
    if any(row["pool_sha256"] != pool_hash for row in rows):
        raise ValueError("Existing grades refer to a different candidate pool")
    return [int(row["grade"]) for row in rows]


def sample_relabel(
    initial_pool: Path, split_file: Path, corpus: Path, labels_file: Path,
    output: Path, size: int | None, seed: int, query_count: int | None = None,
) -> None:
    pool, pairs, _ = load_pool(initial_pool, split_file, corpus)
    if pool.kind != "initial":
        raise ValueError("Sample from an initial pool, not another re-label pass")
    if (size is None) == (query_count is None):
        raise ValueError("Choose either a pair sample size or a query sample count")
    query_ids = sorted({query_id for query_id, _ in pairs})
    if size is not None and not 1 <= size <= len(pairs):
        raise ValueError(f"Sample size must be between 1 and {len(pairs)}")
    if query_count is not None and not 1 <= query_count <= len(query_ids):
        raise ValueError(f"Query sample count must be between 1 and {len(query_ids)}")
    labels = load_labels(labels_file, split_file, corpus)
    require_labels(labels, pairs, "initial", fingerprint(initial_pool))
    rng = random.Random(seed)
    if query_count is not None:
        selected = set(rng.sample(query_ids, query_count))
        sample = [pair for pair in sorted(pairs) if pair[0] in selected]
        rng.shuffle(sample)
    else:
        assert size is not None
        sample = rng.sample(sorted(pairs), size)
    write_json(output, {
        "version": 1, "kind": "relabel", "pass_id": f"relabel-{uuid.uuid4().hex}",
        "split_sha256": fingerprint(split_file), "corpus_sha256": fingerprint(corpus),
        "source_pool_sha256": fingerprint(initial_pool),
        "partition": pool.partition, "seed": seed,
        "sampling_unit": "queries" if query_count is not None else "pairs",
        "pairs": [{"query_id": query_id, "show_id": show_id} for query_id, show_id in sample],
    })


def append_label(path: Path, row: dict[str, str]) -> None:
    if row.get("grade") not in {"0", "1", "2"}:
        raise ValueError("Human relevance grades must be integers from 0 to 2")
    new_file = not path.exists()
    with path.open("x" if new_file else "a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        target.flush()
        os.fsync(target.fileno())


def label_pool(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path,
    read: Callable[[str], str] = input, display: Callable[[str], None] = print,
) -> None:
    pool, pairs, queries = load_pool(pool_file, split_file, corpus)
    labels = load_labels(labels_file, split_file, corpus)
    pool_hash = fingerprint(pool_file)
    done = [pair for pair in pairs if (pool.pass_id, *pair) in labels]
    if done:
        require_labels(labels, done, pool.pass_id, pool_hash)
    if pool.kind == "relabel":
        require_labels(labels, pairs, "initial", pool.source_pool_sha256)
    records, _ = load_records(corpus)
    by_id = {row["show_id"]: row for row in records}
    pending = [pair for pair in pairs if (pool.pass_id, *pair) not in labels]
    display(f"{pool.partition} pass: {len(pending)} unlabelled pair(s). Earlier grades and scores are hidden.")
    for query_id, show_id in pending:
        display(f"\nQuery [{query_id}]: {queries[query_id]['query']}")
        for name in FIELDS:
            display(f"{name}: {by_id[show_id][name]}")
        while True:
            try:
                answer = read("Your relevance grade (0-2), or q to save and quit: ").strip()
            except (EOFError, KeyboardInterrupt):
                display("\nStopped. Previously submitted grades are saved; no grade entered for this pair.")
                return
            if answer.lower() == "q":
                display("Stopped. Previously submitted grades are saved.")
                return
            if answer not in {"0", "1", "2"}:
                display("Enter an explicit integer 0, 1, or 2. There is no default grade.")
                continue
            append_label(labels_file, {
                "query_id": query_id, "show_id": show_id, "grade": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pass_id": pool.pass_id, "split_sha256": pool.split_sha256,
                "corpus_sha256": pool.corpus_sha256, "pool_sha256": pool_hash,
            })
            break
    display("Pass complete.")


def agreement(first: list[int], second: list[int]) -> dict[str, int | float | None | str]:
    if not first or len(first) != len(second):
        raise ValueError("Consistency requires nonempty, equally sized paired human grades")
    if any(type(grade) is not int or grade not in range(3) for grade in first + second):
        raise ValueError("Consistency grades must be integers from 0 to 2")
    count = len(first)
    exact = sum(a == b for a, b in zip(first, second)) / count
    # Quadratic disagreement weights: (a-b)^2 / (2-0)^2.
    observed = sum((a - b) ** 2 / 4 for a, b in zip(first, second)) / count
    a_counts, b_counts = Counter(first), Counter(second)
    expected = sum(
        (a - b) ** 2 / 4 * a_counts[a] * b_counts[b] / count ** 2
        for a in range(3) for b in range(3)
    )
    kappa = None if expected == 0 else 1 - observed / expected
    return {
        "paired_count": count, "exact_agreement": exact,
        "quadratic_weighted_kappa": kappa,
        "kappa_note": "undefined: both passes use the same single grade" if kappa is None else "",
    }


def consistency(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path,
) -> dict[str, int | float | None | str]:
    pool, pairs, _ = load_pool(pool_file, split_file, corpus)
    if pool.kind != "relabel":
        raise ValueError("Consistency requires a re-label pool")
    labels = load_labels(labels_file, split_file, corpus)
    first = require_labels(labels, pairs, "initial", pool.source_pool_sha256)
    second = require_labels(labels, pairs, pool.pass_id, fingerprint(pool_file))
    return agreement(first, second) | {"partition": pool.partition}

"""Provide one CLI from data audit and human grading to frozen, fully judged evaluation."""

import argparse
import csv
import io
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from src.data import DATA_PATH, load_records
from src.study import (
    consistency, fingerprint, freeze_queries, label_pool, load_labels, load_pool,
    load_queries, read_json, read_table, require_labels, sample_relabel, write_json,
)
from src.index import (
    BM25_SETTINGS, EF_CONSTRUCTION, EF_SEARCH, EMBED_MAXLEN, EMBED_REVISION,
    FIELDS, HNSW_M, RERANK_MAXLEN, RERANK_REVISION, build_index,
)
from src.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.search import MODES, RRF_C, search

PROTOCOL = {"embedding_revision": EMBED_REVISION, "reranker_revision": RERANK_REVISION,
            "fields": list(FIELDS), "tokenizer": "casefold Unicode words", "rrf_c": RRF_C,
            "hnsw_m": HNSW_M, "ef_construction": EF_CONSTRUCTION, "ef_search": EF_SEARCH,
            "embedding_max_length": EMBED_MAXLEN, "reranker_max_length": RERANK_MAXLEN,
            "bm25": BM25_SETTINGS, "version": 1}


def frozen_settings(path: Path, split: Path) -> dict:
    value = read_json(path)
    if (value.get("split_sha256") != fingerprint(split)
            or value.get("corpus_sha256") != fingerprint(DATA_PATH)
            or value.get("protocol") != PROTOCOL or value.get("selected_on") != "dev"
            or value.get("mode") not in MODES or type(value.get("relevant_grade")) is not int
            or value.get("relevant_grade") not in (1, 2)
            or type(value.get("k")) is not int or type(value.get("depth")) is not int
            or not 1 <= value["k"] <= value["depth"] <= 100):
        raise ValueError("Incompatible frozen hybrid settings")
    return value


def prepare(split: Path, output: Path, partition: str, k: int, depth: int,
            frozen: Path | None = None, additional_pairs: Path | None = None) -> None:
    queries = load_queries(split, DATA_PATH)
    if partition not in ("dev", "heldout") or type(k) is not int or type(depth) is not int or not 1 <= k <= depth <= 100:
        raise ValueError("Choose dev/heldout and 1 <= k <= candidate depth <= 100")
    modes = MODES
    if partition == "heldout":
        if frozen is None:
            raise ValueError("Held-out pooling requires frozen development settings")
        settings = frozen_settings(frozen, split)
        modes, k, depth = (settings["mode"],), settings["k"], settings["depth"]
    index = build_index()
    runs = [{"mode": mode, "query_id": query_id,
             "ranking": [hit["show_id"] for hit in search(index, row["query"], mode, k, depth)]}
            for query_id, row in queries.items() if row["split"] == partition for mode in modes]
    candidates = {(run["query_id"], show_id) for run in runs for show_id in run["ranking"]}
    if additional_pairs is not None:
        record_ids = {row["show_id"] for row in index.records}
        for row in read_table(additional_pairs, ("query_id", "show_id")):
            query_id, show_id = row["query_id"], row["show_id"]
            if query_id not in queries or queries[query_id]["split"] != partition:
                raise ValueError("Additional candidates must belong to the selected query partition")
            if show_id not in record_ids:
                raise ValueError(f"Unknown additional candidate record: {show_id}")
            candidates.add((query_id, show_id))
    pairs = sorted(candidates)
    random.Random(42).shuffle(pairs)
    write_json(output, {
        "version": 1, "kind": "initial", "pass_id": "initial", "partition": partition,
        "split_sha256": fingerprint(split), "corpus_sha256": fingerprint(DATA_PATH),
        "protocol": PROTOCOL, "k": k, "depth": depth, "runs": runs,
        "frozen_sha256": fingerprint(frozen) if frozen else None,
        "additional_pairs_sha256": fingerprint(additional_pairs) if additional_pairs else None,
        "pairs": [{"query_id": query_id, "show_id": show_id} for query_id, show_id in pairs],
    })


def evaluate_pool(split: Path, pool_file: Path, labels_file: Path, grade: int,
                  frozen: Path | None = None) -> tuple[str, dict]:
    pool, pairs, queries = load_pool(pool_file, split, DATA_PATH)
    saved = read_json(pool_file)
    if pool.kind != "initial" or saved.get("protocol") != PROTOCOL or type(grade) is not int or grade not in (1, 2):
        raise ValueError("Supply an initial hybrid pool and an explicit threshold of 1 or 2")
    k, depth = saved.get("k"), saved.get("depth")
    if type(k) is not int or type(depth) is not int or not 1 <= k <= depth <= 100:
        raise ValueError("Invalid hybrid pool depth")
    modes = MODES
    if pool.partition == "heldout":
        if frozen is None:
            raise ValueError("Held-out evaluation requires frozen development settings")
        settings = frozen_settings(frozen, split)
        if (saved.get("frozen_sha256") != fingerprint(frozen) or grade != settings["relevant_grade"]
                or (k, depth) != (settings["k"], settings["depth"])):
            raise ValueError("Held-out protocol differs from frozen settings")
        modes = (settings["mode"],)
    grades = require_labels(load_labels(labels_file, split, DATA_PATH), pairs, "initial", fingerprint(pool_file))
    judgments: dict[str, dict[str, int]] = {}
    for (query_id, show_id), value in zip(pairs, grades):
        judgments.setdefault(query_id, {})[show_id] = value
    expected = {(mode, key) for mode in modes for key, row in queries.items() if row["split"] == pool.partition}
    runs = saved.get("runs")
    if not isinstance(runs, list) or any(
        not isinstance(run, dict) or set(run) != {"mode", "query_id", "ranking"}
        or run["mode"] not in modes or not isinstance(run["query_id"], str)
        or not isinstance(run["ranking"], list)
        or any(not isinstance(show_id, str) or not show_id.strip() for show_id in run["ranking"])
        for run in runs
    ):
        raise ValueError("Invalid frozen query/method rankings")
    actual = [(run["mode"], run["query_id"]) for run in runs]
    if len(actual) != len(set(actual)) or set(actual) != expected or set(judgments) != {key for _, key in expected}:
        raise ValueError("Pool must contain every frozen query/method exactly once")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query_id", "method", "partition", "k", "relevant_grade", "precision", "recall", "rr_at_k", "ndcg"])
    for run in runs:
        ranking, query_id = run["ranking"], run["query_id"]
        if len(ranking) > k:
            raise ValueError("Saved ranking exceeds its frozen cutoff")
        judged = judgments.get(query_id, {})
        writer.writerow([query_id, run["mode"], pool.partition, k, grade,
                         precision_at_k(ranking, judged, k, relevant_grade=grade),
                         recall_at_k(ranking, judged, k, relevant_grade=grade),
                         reciprocal_rank(ranking, judged, relevant_grade=grade), ndcg_at_k(ranking, judged, k)])
    return output.getvalue(), saved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("audit", "split", "pool", "label", "sample", "consistency", "evaluate", "freeze"):
        command = commands.add_parser(name)
        if name not in ("audit", "split"):
            command.add_argument("--split", type=Path, required=True)
        if name in ("split", "pool", "sample", "evaluate", "freeze"):
            command.add_argument("--output", type=Path, required=True)
        if name in ("label", "sample", "consistency", "evaluate", "freeze"):
            command.add_argument("--pool", type=Path, required=True)
            command.add_argument("--labels", type=Path, required=True)
        if name == "split":
            command.add_argument("--queries", type=Path, required=True)
            command.add_argument("--heldout-fraction", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=42)
        if name == "pool":
            command.add_argument("--partition", choices=("dev", "heldout"), default="dev")
            command.add_argument("-k", type=int, default=10)
            command.add_argument("--depth", type=int, default=50)
            command.add_argument("--frozen", type=Path)
            command.add_argument("--additional-pairs", type=Path, help="Ungraded query_id,show_id CSV")
        if name == "sample":
            size = command.add_mutually_exclusive_group(required=True)
            size.add_argument("--size", type=int, help="Number of individual pairs")
            size.add_argument("--query-count", type=int, help="Repeat all pairs for this many queries")
            command.add_argument("--seed", type=int, default=43)
        if name in ("evaluate", "freeze"):
            command.add_argument("--relevant-grade", type=int, choices=(1, 2), required=True)
            command.add_argument("--frozen", type=Path)
        if name == "freeze":
            command.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "audit":
            _, report = load_records(DATA_PATH)
            print(json.dumps(asdict(report) | {"source": str(DATA_PATH), "sha256": fingerprint(DATA_PATH)}, indent=2))
        elif args.action == "split":
            load_records(DATA_PATH)
            freeze_queries(args.queries, args.output, DATA_PATH, args.heldout_fraction, args.seed)
            print("Query split frozen. Use only dev queries for tuning; keep heldout sealed.")
        elif args.action == "pool":
            prepare(args.split, args.output, args.partition, args.k, args.depth, args.frozen, args.additional_pairs)
            print("Ungraded pool saved. Held-out runs use frozen development settings; no labels were generated.")
        elif args.action == "label":
            label_pool(args.pool, args.split, DATA_PATH, args.labels)
        elif args.action == "sample":
            sample_relabel(args.pool, args.split, DATA_PATH, args.labels, args.output,
                           args.size, args.seed, args.query_count)
            print("Blind re-label queue saved. Return later to label this same queue.")
        elif args.action == "consistency":
            print(json.dumps(consistency(args.pool, args.split, DATA_PATH, args.labels), indent=2))
        else:
            text, saved = evaluate_pool(args.split, args.pool, args.labels, args.relevant_grade, args.frozen)
            if args.action == "evaluate":
                with args.output.open("x", encoding="utf-8", newline="") as target:
                    target.write(text)
                rows = list(csv.DictReader(io.StringIO(text)))
                summary = {}
                for mode in MODES:
                    group = [row for row in rows if row["method"] == mode]
                    if group:
                        summary[mode] = {
                            name: sum(float(row[column]) for row in group) / len(group)
                            for name, column in (("precision", "precision"), ("recall", "recall"),
                                                 ("mrr_at_k", "rr_at_k"), ("ndcg", "ndcg"))
                        }
                print(json.dumps({"partition": saved["partition"], "k": saved["k"],
                                 "relevant_grade": args.relevant_grade, "macro_average": summary}, indent=2))
            else:
                if saved["partition"] != "dev":
                    raise ValueError("Freeze one --mode selected using development results only")
                write_json(args.output, {key: saved[key] for key in ("protocol", "k", "depth", "split_sha256",
                                                                    "corpus_sha256")} | {
                    "selected_on": "dev", "mode": args.mode, "relevant_grade": args.relevant_grade,
                    "dev_pool_sha256": fingerprint(args.pool), "labels_sha256": fingerprint(args.labels),
                })
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

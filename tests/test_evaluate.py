"""Protect pooled human-only evaluation and the development-to-held-out boundary."""

import csv
from pathlib import Path

import pytest

from src.study import append_label, load_pool, sample_relabel, fingerprint, freeze_queries, read_json, write_json
from src import evaluate
from src.index import Index


@pytest.fixture
def study(tmp_path: Path, small_index: Index, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    from src.data import FIELDS

    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(small_index.records)
    paths = {name: tmp_path / name for name in ("queries.csv", "split.json", "pool.json", "labels.csv", "frozen.json")}
    paths["queries.csv"].write_text("query_id,query\nq1,orbit\nq2,cooking\nq3,nature\n", encoding="utf-8")
    freeze_queries(paths["queries.csv"], paths["split.json"], corpus, 0.34)
    monkeypatch.setattr(evaluate, "DATA_PATH", corpus)
    monkeypatch.setattr(evaluate, "build_index", lambda: small_index)
    evaluate.prepare(paths["split.json"], paths["pool.json"], "dev", 2, 3)
    paths["corpus.csv"] = corpus
    return paths


def label_fixture(paths: dict[str, Path], pool_path: Path) -> None:
    pool, pairs, _ = load_pool(pool_path, paths["split.json"], paths["corpus.csv"])
    for query_id, show_id in pairs:
        # Artificial records only: this is not a human Netflix judgment list.
        append_label(paths["labels.csv"], {
            "query_id": query_id, "show_id": show_id, "grade": "2",
            "timestamp": "2026-01-01T00:00:00+00:00", "pass_id": "initial",
            "split_sha256": pool.split_sha256, "corpus_sha256": pool.corpus_sha256,
            "pool_sha256": fingerprint(pool_path),
        })


def test_four_method_union_has_no_grades_and_requires_labels(study: dict[str, Path]) -> None:
    saved = read_json(study["pool.json"])
    assert {run["mode"] for run in saved["runs"]} == set(evaluate.MODES)
    assert all(set(pair) == {"query_id", "show_id"} for pair in saved["pairs"])
    assert {(pair["query_id"], pair["show_id"]) for pair in saved["pairs"]} == {
        (run["query_id"], show_id) for run in saved["runs"] for show_id in run["ranking"]}
    with pytest.raises(ValueError, match="Missing human labels"):
        evaluate.evaluate_pool(study["split.json"], study["pool.json"], study["labels.csv"], 1)
    label_fixture(study, study["pool.json"])
    text, _ = evaluate.evaluate_pool(study["split.json"], study["pool.json"], study["labels.csv"], 1)
    assert len(list(csv.DictReader(text.splitlines()))) == 4
    repeat = study["pool.json"].with_name("repeat.json")
    sample_relabel(study["pool.json"], study["split.json"], study["corpus.csv"], study["labels.csv"],
                   repeat, None, 43, query_count=1)
    assert len(read_json(repeat)["pairs"]) == len(saved["pairs"])


def test_heldout_only_uses_frozen_winner(study: dict[str, Path]) -> None:
    saved = read_json(study["pool.json"])
    settings = {key: saved[key] for key in ("protocol", "k", "depth", "split_sha256", "corpus_sha256")} | {
        "selected_on": "dev", "mode": "hybrid", "relevant_grade": 1,
    }
    heldout = study["pool.json"].with_name("heldout.json")
    with pytest.raises(ValueError, match="frozen"):
        evaluate.prepare(study["split.json"], heldout, "heldout", 2, 3)
    write_json(study["frozen.json"], settings)
    evaluate.prepare(study["split.json"], heldout, "heldout", 2, 3, study["frozen.json"])
    assert {run["mode"] for run in read_json(heldout)["runs"]} == {"hybrid"}
    label_fixture(study, heldout)
    text, _ = evaluate.evaluate_pool(study["split.json"], heldout, study["labels.csv"], 1, study["frozen.json"])
    assert len(list(csv.DictReader(text.splitlines()))) == 2
    with pytest.raises(ValueError, match="protocol"):
        evaluate.evaluate_pool(study["split.json"], heldout, study["labels.csv"], 2, study["frozen.json"])


def test_extra_human_candidates_are_ungraded_deduplicated_and_checked(study: dict[str, Path]) -> None:
    baseline = study["pool.json"].with_name("baseline.json")
    evaluate.prepare(study["split.json"], baseline, "dev", 1, 3)
    saved = read_json(baseline)
    query_id = saved["pairs"][0]["query_id"]
    existing = saved["pairs"][0]["show_id"]
    included = {pair["show_id"] for pair in saved["pairs"]}
    missed = next(show_id for show_id in ("s1", "s2", "s3") if show_id not in included)
    extra = study["pool.json"].with_name("extra.csv")
    extra.write_text(f"query_id,show_id\n{query_id},{existing}\n{query_id},{missed}\n{query_id},{missed}\n",
                     encoding="utf-8")
    extended = study["pool.json"].with_name("extended.json")
    evaluate.prepare(study["split.json"], extended, "dev", 1, 3, additional_pairs=extra)
    extended_pool = read_json(extended)
    assert len(extended_pool["pairs"]) == len(saved["pairs"]) + 1
    assert all(set(pair) == {"query_id", "show_id"} for pair in extended_pool["pairs"])
    assert extended_pool["additional_pairs_sha256"] == fingerprint(extra)
    for content in (f"query_id,show_id\n{query_id},missing\n",
                    "query_id,show_id\nunknown,s1\n",
                    f"query_id,show_id,grade\n{query_id},s1,0\n"):
        extra.write_text(content, encoding="utf-8")
        bad = extra.with_name("bad.json")
        with pytest.raises(ValueError):
            evaluate.prepare(study["split.json"], bad, "dev", 1, 3, additional_pairs=extra)
        assert not bad.exists()


def test_single_cli_evaluates_macro_metrics_and_freezes_dev_only(
    study: dict[str, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    label_fixture(study, study["pool.json"])
    output = study["pool.json"].with_name("results.csv")
    arguments = ["--split", str(study["split.json"]), "--pool", str(study["pool.json"]),
                 "--labels", str(study["labels.csv"]), "--relevant-grade", "1"]
    evaluate.main(["evaluate", *arguments, "--output", str(output)])
    summary = json.loads(capsys.readouterr().out)
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert set(summary["macro_average"]) == set(evaluate.MODES)
    for mode, values in summary["macro_average"].items():
        group = [row for row in rows if row["method"] == mode]
        assert values["mrr_at_k"] == sum(float(row["rr_at_k"]) for row in group) / len(group)
    evaluate.main(["freeze", *arguments, "--mode", "hybrid", "--output", str(study["frozen.json"])])
    settings = read_json(study["frozen.json"])
    assert settings["selected_on"] == "dev" and settings["mode"] == "hybrid"
    assert settings["dev_pool_sha256"] == fingerprint(study["pool.json"])
    assert settings["labels_sha256"] == fingerprint(study["labels.csv"])
    with pytest.raises(SystemExit):
        evaluate.main(["freeze", *arguments, "--mode", "hybrid", "--output", str(study["frozen.json"])])


@pytest.mark.parametrize("change", [
    {"mode": "unknown"}, {"relevant_grade": 3}, {"relevant_grade": True},
    {"k": 0}, {"depth": 101}, {"selected_on": "heldout"}, {"protocol": {}},
])
def test_incompatible_frozen_settings_cannot_be_reused(study: dict[str, Path], change: dict) -> None:
    saved = read_json(study["pool.json"])
    settings = {key: saved[key] for key in ("protocol", "k", "depth", "split_sha256", "corpus_sha256")} | {
        "selected_on": "dev", "mode": "hybrid", "relevant_grade": 1,
    }
    write_json(study["frozen.json"], settings | change)
    with pytest.raises(ValueError, match="Incompatible"):
        evaluate.frozen_settings(study["frozen.json"], study["split.json"])

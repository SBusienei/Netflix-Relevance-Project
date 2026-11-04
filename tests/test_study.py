"""Use synthetic records and grades to protect the human-only labelling protocol."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sklearn.metrics import cohen_kappa_score

from src.data import FIELDS
from src.study import (
    LABEL_COLUMNS, agreement, append_label, consistency, label_pool, load_labels,
    load_pool, sample_relabel, fingerprint, freeze_queries, load_queries, read_table, write_json,
)
from src import evaluate
from src.evaluate import main
from src.index import Index


def test_append_rejects_grade_three_without_creating_a_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    with pytest.raises(ValueError, match="integers from 0 to 2"):
        append_label(path, {"grade": "3"})
    assert not path.exists()


def test_query_relabel_sampling_repeats_complete_queries(workspace: dict[str, Path]) -> None:
    complete_initial(workspace)
    sample_relabel(workspace["pool.json"], workspace["split.json"], workspace["corpus.csv"],
                   workspace["labels.csv"], workspace["repeat.json"], None, 43, query_count=2)
    initial = json.loads(workspace["pool.json"].read_text(encoding="utf-8"))["pairs"]
    saved = json.loads(workspace["repeat.json"].read_text(encoding="utf-8"))
    selected = {pair["query_id"] for pair in saved["pairs"]}
    assert len(selected) == 2 and saved["sampling_unit"] == "queries"
    assert {(pair["query_id"], pair["show_id"]) for pair in saved["pairs"]} == {
        (pair["query_id"], pair["show_id"]) for pair in initial if pair["query_id"] in selected}
    assert all(set(pair) == {"query_id", "show_id"} for pair in saved["pairs"])


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in (
        "corpus.csv", "queries.csv", "split.json", "pool.json", "labels.csv", "repeat.json",
    )}
    with paths["corpus.csv"].open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for number, title in enumerate(("Orbit", "Kitchen", "Forest", "Ocean"), 1):
            writer.writerow(dict.fromkeys(FIELDS, "") | {
                "show_id": f"s{number}", "title": title, "description": f"{title} tale",
            })
    paths["queries.csv"].write_text(
        "query_id,query\nq1,Orbit\nq2,Kitchen\nq3,Forest\nq4,Ocean\n", encoding="utf-8",
    )
    freeze_queries(paths["queries.csv"], paths["split.json"], paths["corpus.csv"], 0.25, 42)
    # The human-tool tests use an ungraded synthetic queue, not a retrieval model or Netflix labels.
    queries = load_queries(paths["split.json"], paths["corpus.csv"])
    write_json(paths["pool.json"], {
        "version": 1, "kind": "initial", "pass_id": "initial", "partition": "dev",
        "split_sha256": fingerprint(paths["split.json"]), "corpus_sha256": fingerprint(paths["corpus.csv"]),
        "pairs": [{"query_id": query_id, "show_id": show_id}
                  for query_id, row in queries.items() if row["split"] == "dev"
                  for show_id in ("s1", "s2")],
    })
    return paths


def label(
    paths: dict[str, Path], answers: list[str], pool: str = "pool.json",
) -> list[str]:
    responses = iter(answers)
    messages: list[str] = []
    label_pool(
        paths[pool], paths["split.json"], paths["corpus.csv"], paths["labels.csv"],
        read=lambda _: next(responses), display=messages.append,
    )
    return messages


def complete_initial(paths: dict[str, Path]) -> None:
    # These are artificial test-fixture grades, never labels for Netflix records.
    label(paths, ["0", "1", "2", "2", "0", "1"])


def make_repeat(paths: dict[str, Path], size: int = 3) -> None:
    sample_relabel(
        paths["pool.json"], paths["split.json"], paths["corpus.csv"],
        paths["labels.csv"], paths["repeat.json"], size, 43,
    )


def test_frozen_split_is_disjoint_complete_and_unchangeable(workspace: dict[str, Path]) -> None:
    p = workspace
    queries = load_queries(p["split.json"], p["corpus.csv"])
    dev = {key for key, row in queries.items() if row["split"] == "dev"}
    heldout = {key for key, row in queries.items() if row["split"] == "heldout"}
    assert len(dev) == 3 and len(heldout) == 1
    assert not dev & heldout and dev | heldout == {"q1", "q2", "q3", "q4"}
    original = p["split.json"].read_bytes()
    with pytest.raises(FileExistsError):
        freeze_queries(p["queries.csv"], p["split.json"], p["corpus.csv"])
    assert p["split.json"].read_bytes() == original


def test_split_membership_does_not_depend_on_csv_order(workspace: dict[str, Path], tmp_path: Path) -> None:
    p = workspace
    p["queries.csv"].write_text(
        "query_id,query\nq4,Ocean\nq3,Forest\nq2,Kitchen\nq1,Orbit\n", encoding="utf-8",
    )
    other = tmp_path / "other.json"
    freeze_queries(p["queries.csv"], other, p["corpus.csv"], 0.25, 42)
    assert load_queries(other, p["corpus.csv"]) == load_queries(p["split.json"], p["corpus.csv"])


@pytest.mark.parametrize("queries", [
    "query_id,query\n", "query_id,query\nq1,Only one\n",
    "query_id,query\nq1,One\nq1,Two\n",
    "query_id,query\nq1,One\nq2, ONE \n",
    "query_id,query\nq1, \nq2,Two\n",
    "query_id,query\n,One\nq2,Two\n",
    "query_id,text\nq1,One\nq2,Two\n",
])
def test_invalid_query_inventory_is_rejected(
    workspace: dict[str, Path], tmp_path: Path, queries: str,
) -> None:
    workspace["queries.csv"].write_text(queries, encoding="utf-8")
    output = tmp_path / "bad-split.json"
    with pytest.raises(ValueError):
        freeze_queries(workspace["queries.csv"], output, workspace["corpus.csv"])
    assert not output.exists()


@pytest.mark.parametrize("fraction", [0, 1, -0.5, float("nan")])
def test_invalid_split_fraction(workspace: dict[str, Path], tmp_path: Path, fraction: float) -> None:
    with pytest.raises(ValueError):
        freeze_queries(workspace["queries.csv"], tmp_path / "bad.json", workspace["corpus.csv"], fraction)


def test_changed_corpus_or_split_cannot_reuse_pool(workspace: dict[str, Path]) -> None:
    p = workspace
    p["split.json"].write_text(p["split.json"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_pool(p["pool.json"], p["split.json"], p["corpus.csv"])
    p["corpus.csv"].write_text(p["corpus.csv"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_queries(p["split.json"], p["corpus.csv"])


def test_pool_only_contains_selected_partition_and_no_grades(workspace: dict[str, Path]) -> None:
    p = workspace
    pool, pairs, queries = load_pool(p["pool.json"], p["split.json"], p["corpus.csv"])
    assert pool.partition == "dev"
    assert len(pairs) == 6 and len(set(pairs)) == 6
    assert all(queries[query_id]["split"] == "dev" for query_id, _ in pairs)
    raw = json.loads(p["pool.json"].read_text())
    assert all(set(pair) == {"query_id", "show_id"} for pair in raw["pairs"])
    assert not p["labels.csv"].exists()


@pytest.mark.parametrize("mutate", ["grade", "unknown", "duplicate", "heldout", "kind", "empty"])
def test_bad_candidate_pool_is_rejected(workspace: dict[str, Path], mutate: str) -> None:
    p = workspace
    raw = json.loads(p["pool.json"].read_text())
    if mutate == "grade":
        raw["pairs"][0]["grade"] = "0"
    elif mutate == "unknown":
        raw["pairs"][0]["show_id"] = "missing"
    elif mutate == "duplicate":
        raw["pairs"].append(raw["pairs"][0])
    elif mutate == "heldout":
        queries = load_queries(p["split.json"], p["corpus.csv"])
        raw["pairs"][0]["query_id"] = next(key for key, row in queries.items() if row["split"] == "heldout")
    elif mutate == "kind":
        raw["kind"] = []
    else:
        raw["pairs"] = []
    p["pool.json"].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pool(p["pool.json"], p["split.json"], p["corpus.csv"])


def test_explicit_input_timestamp_and_resume(workspace: dict[str, Path]) -> None:
    p = workspace
    messages = label(p, ["", "3", "4", "1.5", "2", "q"])
    assert sum("no default" in message for message in messages) == 4
    rows = read_table(p["labels.csv"], LABEL_COLUMNS)
    assert len(rows) == 1 and rows[0]["grade"] == "2"
    assert rows[0]["timestamp"].endswith("+00:00")
    assert rows[0]["pool_sha256"] == fingerprint(p["pool.json"])
    label(p, ["1", "0", "2", "2", "1"])
    assert len(read_table(p["labels.csv"], LABEL_COLUMNS)) == 6
    messages = label(p, [])
    assert "0 unlabelled" in messages[0] and messages[-1] == "Pass complete."


def test_quit_or_eof_does_not_create_grades(workspace: dict[str, Path]) -> None:
    p = workspace
    label(p, ["q"])
    assert not p["labels.csv"].exists()

    def eof(_: str) -> str:
        raise EOFError

    label_pool(p["pool.json"], p["split.json"], p["corpus.csv"], p["labels.csv"],
               read=eof, display=lambda _: None)
    assert not p["labels.csv"].exists()


def test_missing_initial_labels_stop_sampling(workspace: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="Missing human labels: 6"):
        make_repeat(workspace)
    assert not workspace["repeat.json"].exists()


def test_blind_saved_subset_resumes_and_preserves_initial_labels(workspace: dict[str, Path]) -> None:
    p = workspace
    complete_initial(p)
    original = p["labels.csv"].read_bytes()
    make_repeat(p)
    pool, pairs, _ = load_pool(p["repeat.json"], p["split.json"], p["corpus.csv"])
    _, initial, _ = load_pool(p["pool.json"], p["split.json"], p["corpus.csv"])
    assert len(pairs) == 3 and set(pairs) <= set(initial)
    assert pool.pass_id != "initial"
    sample_bytes = p["repeat.json"].read_bytes()
    assert "grade" not in p["repeat.json"].read_text()
    with pytest.raises(FileExistsError):
        make_repeat(p)
    assert p["repeat.json"].read_bytes() == sample_bytes
    messages = label(p, ["1", "q"], pool="repeat.json")
    assert not any("grade:" in message or "score:" in message for message in messages)
    assert all(not message.startswith("timestamp:") for message in messages)
    assert p["labels.csv"].read_bytes().startswith(original)
    with pytest.raises(ValueError, match="Missing human labels: 2"):
        consistency(p["repeat.json"], p["split.json"], p["corpus.csv"], p["labels.csv"])
    label(p, ["0", "2"], pool="repeat.json")
    result = consistency(p["repeat.json"], p["split.json"], p["corpus.csv"], p["labels.csv"])
    assert result["paired_count"] == 3 and result["partition"] == "dev"
    rows = read_table(p["labels.csv"], LABEL_COLUMNS)
    assert len(rows) == 9
    with p["labels.csv"].open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(reversed(rows))
    assert consistency(p["repeat.json"], p["split.json"], p["corpus.csv"], p["labels.csv"]) == result


def test_sampling_is_reproducible_without_reusing_pass_ids(
    workspace: dict[str, Path], tmp_path: Path,
) -> None:
    p = workspace
    complete_initial(p)
    make_repeat(p)
    other = tmp_path / "other-repeat.json"
    sample_relabel(p["pool.json"], p["split.json"], p["corpus.csv"], p["labels.csv"], other, 3, 43)
    first, first_pairs, _ = load_pool(p["repeat.json"], p["split.json"], p["corpus.csv"])
    second, second_pairs, _ = load_pool(other, p["split.json"], p["corpus.csv"])
    assert first_pairs == second_pairs and first.pass_id != second.pass_id


def test_initial_grades_cannot_be_reused_after_pool_changes(workspace: dict[str, Path]) -> None:
    p = workspace
    complete_initial(p)
    p["pool.json"].write_text(p["pool.json"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different candidate pool"):
        label(p, [])


def test_invalid_or_empty_judgment_csv_fails_explicitly(workspace: dict[str, Path]) -> None:
    p = workspace
    label(p, ["1", "q"])
    original = p["labels.csv"].read_text()
    row = read_table(p["labels.csv"], LABEL_COLUMNS)[0]
    for column, value in (("grade", "3"), ("grade", "4"), ("timestamp", "2026-10-28T00:00:00"),
                          ("show_id", "missing"), ("pool_sha256", "bad"),
                          ("split_sha256", "different")):
        with p["labels.csv"].open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerow(row | {column: value})
        with pytest.raises(ValueError):
            load_labels(p["labels.csv"], p["split.json"], p["corpus.csv"])
    p["labels.csv"].write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected CSV columns"):
        load_labels(p["labels.csv"], p["split.json"], p["corpus.csv"])
    p["labels.csv"].write_text(original, encoding="utf-8")


def test_duplicate_csv_grades_are_rejected(workspace: dict[str, Path]) -> None:
    p = workspace
    label(p, ["1", "q"])
    text = p["labels.csv"].read_text()
    p["labels.csv"].write_text(text + text.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate grade"):
        load_labels(p["labels.csv"], p["split.json"], p["corpus.csv"])


def test_hand_computed_agreement() -> None:
    # Synthetic passes [0,1,2] vs [0,2,1]: 1/3 exact.
    # Observed quadratic disagreement = (0+1+1)/(4*3) = 1/6.
    # Both marginals uniform; expected squared distance = 2*Var(0,1,2)
    # = 2*(2/3) = 4/3, so expected weighted disagreement = (4/3)/4 = 1/3.
    # Kappa = 1-(1/6)/(1/3) = 0.5.
    result = agreement([0, 1, 2], [0, 2, 1])
    assert result["exact_agreement"] == 1 / 3
    assert result["quadratic_weighted_kappa"] == pytest.approx(0.5)


def test_kappa_cross_check_is_separate_from_definition_test() -> None:
    first, second = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2]
    assert agreement(first, second)["quadratic_weighted_kappa"] == pytest.approx(
        cohen_kappa_score(first, second, labels=[0, 1, 2], weights="quadratic"),
    )


def test_degenerate_kappa_is_explicitly_undefined() -> None:
    result = agreement([2, 2], [2, 2])
    assert result["exact_agreement"] == 1
    assert result["quadratic_weighted_kappa"] is None
    assert "undefined" in str(result["kappa_note"])
    assert agreement([0, 0], [2, 2])["quadratic_weighted_kappa"] == 0


@pytest.mark.parametrize("first,second", [
    ([], []), ([0], []), ([3], [0]), ([4], [0]), ([True], [1]),
])
def test_invalid_consistency_inputs(first: list[int], second: list[int]) -> None:
    with pytest.raises(ValueError):
        agreement(first, second)


def test_cli_missing_labels_are_an_error_not_a_result(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = workspace
    complete_initial(p)
    make_repeat(p)
    monkeypatch.setattr(evaluate, "DATA_PATH", p["corpus.csv"])
    with pytest.raises(SystemExit) as error:
        main(["consistency", "--pool", str(p["repeat.json"]), "--split", str(p["split.json"]),
              "--labels", str(p["labels.csv"])])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Missing human labels" in captured.err
    assert "quadratic_weighted_kappa" not in captured.out


def test_actual_cli_accepts_stdin_and_quits_cleanly(workspace: dict[str, Path]) -> None:
    p = workspace
    # Inject a synthetic corpus into the child; exercise the real CLI parser and stdin.
    script = ("import sys; from pathlib import Path; from src import evaluate; "
              "evaluate.DATA_PATH = Path(sys.argv.pop(1)); evaluate.main()")
    result = subprocess.run(
        [sys.executable, "-c", script, str(p["corpus.csv"]), "label", "--pool", str(p["pool.json"]),
         "--split", str(p["split.json"]),
         "--labels", str(p["labels.csv"])],
        input="2\nq\n", capture_output=True, text=True, check=True,
    )
    assert "Query [" in result.stdout and "description:" in result.stdout
    assert "score:" not in result.stdout
    assert len(read_table(p["labels.csv"], LABEL_COLUMNS)) == 1


def test_cli_split_pool_sample_and_consistency_round_trip(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    small_index: Index, monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = workspace
    split_file, pool_file, repeat_file = (
        tmp_path / "cli-split.json", tmp_path / "cli-pool.json", tmp_path / "cli-repeat.json",
    )
    monkeypatch.setattr(evaluate, "DATA_PATH", p["corpus.csv"])
    monkeypatch.setattr(evaluate, "build_index", lambda: small_index)
    main(["split", "--queries", str(p["queries.csv"]), "--output", str(split_file),
          "--heldout-fraction", "0.25"])
    main(["pool", "--split", str(split_file), "--output", str(pool_file), "-k", "2", "--depth", "3"])
    pool = json.loads(pool_file.read_text(encoding="utf-8"))
    responses = iter(["2"] * len(pool["pairs"]))
    label_pool(pool_file, split_file, p["corpus.csv"], p["labels.csv"],
               read=lambda _: next(responses), display=lambda _: None)
    main(["sample", "--split", str(split_file),
          "--pool", str(pool_file), "--labels", str(p["labels.csv"]), "--size", "2",
          "--output", str(repeat_file)])
    responses = iter(["1", "2"])
    label_pool(repeat_file, split_file, p["corpus.csv"], p["labels.csv"],
               read=lambda _: next(responses), display=lambda _: None)
    capsys.readouterr()
    main(["consistency", "--split", str(split_file),
          "--pool", str(repeat_file), "--labels", str(p["labels.csv"])])
    assert json.loads(capsys.readouterr().out)["paired_count"] == 2

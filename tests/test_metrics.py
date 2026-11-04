"""Check metric definitions against hand-worked artificial examples, not research labels."""

import pytest

from src.metrics import (
    mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)


def test_precision_and_recall_hand_calculation() -> None:
    judgments = {"a": 0, "b": 2, "c": 1, "d": 2, "e": 0}
    ranking = ["a", "c", "b"]
    # At threshold 1, b/c/d are relevant. Top two are a/c: one relevant hit.
    # P@2 = 1/2; R@2 = 1/3. At three, P@3 = 2/3 and R@3 = 2/3.
    assert precision_at_k(ranking, judgments, 2, relevant_grade=1) == 1 / 2
    assert recall_at_k(ranking, judgments, 2, relevant_grade=1) == 1 / 3
    assert precision_at_k(ranking, judgments, 3, relevant_grade=1) == 2 / 3
    assert recall_at_k(ranking, judgments, 3, relevant_grade=1) == 2 / 3


def test_binary_threshold_is_explicit_and_inclusive() -> None:
    judgments = {"a": 0, "b": 1, "c": 2}
    ranking = ["a", "b", "c"]
    # Threshold 1: b/c relevant, a not. P@2 = 1/2; R@2 = 1/2; RR = 1/2.
    # Threshold 2: only c relevant, at rank 3; P@2 = R@2 = 0; RR = 1/3.
    assert precision_at_k(ranking, judgments, 2, relevant_grade=1) == 1 / 2
    assert recall_at_k(ranking, judgments, 2, relevant_grade=1) == 1 / 2
    assert reciprocal_rank(ranking, judgments, relevant_grade=1) == 1 / 2
    assert precision_at_k(ranking, judgments, 2, relevant_grade=2) == 0
    assert recall_at_k(ranking, judgments, 2, relevant_grade=2) == 0
    assert reciprocal_rank(ranking, judgments, relevant_grade=2) == 1 / 3


def test_short_ranking_does_not_change_precision_denominator() -> None:
    judgments = {"a": 2, "b": 1}
    # One returned relevant document, k=5: P@5=1/5, not 1.
    # There are two judged relevant documents, so R@5=1/2.
    assert precision_at_k(["a"], judgments, 5, relevant_grade=1) == 1 / 5
    assert recall_at_k(["a"], judgments, 5, relevant_grade=1) == 1 / 2


def test_reciprocal_rank_hand_calculation() -> None:
    judgments = {"a": 0, "b": 0, "c": 1, "d": 2}
    # First relevant result c is at rank 3: RR=1/3. d at rank 4 adds nothing.
    assert reciprocal_rank(["a", "b", "c", "d"], judgments, relevant_grade=1) == 1 / 3
    # No relevant hit retrieved: RR=0, even though c/d are relevant in the list.
    assert reciprocal_rank(["a", "b"], judgments, relevant_grade=1) == 0
    assert reciprocal_rank(["d", "c"], judgments, relevant_grade=1) == 1


def test_mrr_is_an_equal_query_average() -> None:
    rankings = {
        "first": ["a", "b"], "second": ["a", "b", "c"], "third": ["a"],
    }
    judgments = {
        "first": {"a": 0, "b": 2},
        "second": {"a": 0, "b": 0, "c": 2},
        "third": {"a": 0, "d": 1},
    }
    # RRs are 1/2, 1/3, and 0. MRR=(1/2+1/3+0)/3 = 5/18.
    # Different run lengths do not change the weight of each query.
    assert mean_reciprocal_rank(rankings, judgments, relevant_grade=1) == pytest.approx(5 / 18)


def test_ndcg_hand_calculation_with_a_missed_high_grade() -> None:
    judgments = {"a": 0, "b": 2, "c": 1, "d": 1}
    # Linear gains, base-2 discounts. At k=2, returned grades [0,1]:
    # DCG=0/log2(2)+1/log2(3)=0.6309297535714575.
    # Ideal grades [2,1]: IDCG=2+1/log2(3)=2.6309297535714575.
    # nDCG=0.6309297535714575/2.6309297535714575.
    assert ndcg_at_k(["a", "c", "b"], judgments, 2) == pytest.approx(
        0.6309297535714575 / 2.6309297535714575,
    )


def test_ndcg_at_three_hand_calculation() -> None:
    judgments = {"a": 0, "b": 2, "c": 1, "d": 1}
    # Returned [2,0,1]: DCG=2+0+1/log2(4)=2.5.
    # Ideal [2,1,1], including missed d: IDCG=2+1/log2(3)+1/2
    # =3.1309297535714575. nDCG@3=2.5/3.1309297535714575.
    assert ndcg_at_k(["b", "a", "c"], judgments, 3) == pytest.approx(
        2.5 / 3.1309297535714575,
    )


def test_ndcg_ideal_ranking_and_cutoff_one() -> None:
    judgments = {"a": 2, "b": 1, "c": 1, "d": 0}
    # Ideal ordering gives DCG=IDCG at any cutoff, including k>list length.
    assert ndcg_at_k(["a", "b", "c", "d"], judgments, 10) == 1
    # At k=1 there is no rank discount: returned gain 1 / ideal gain 2.
    assert ndcg_at_k(["b"], judgments, 1) == 1 / 2


def test_empty_ranking_is_explicitly_a_run_with_no_hits() -> None:
    judgments = {"a": 2}
    # Zero hits: P=0/k=0, R=0/1=0, RR=0, nDCG=0/2=0.
    assert precision_at_k([], judgments, 3, relevant_grade=1) == 0
    assert recall_at_k([], judgments, 3, relevant_grade=1) == 0
    assert reciprocal_rank([], judgments, relevant_grade=1) == 0
    assert ndcg_at_k([], judgments, 3) == 0
    assert mean_reciprocal_rank({"q": []}, {"q": judgments}, relevant_grade=1) == 0


def test_no_judged_positives_has_explicit_undefined_metrics() -> None:
    judgments = {"a": 0}
    assert precision_at_k(["a"], judgments, 1, relevant_grade=1) == 0
    assert reciprocal_rank(["a"], judgments, relevant_grade=1) == 0
    with pytest.raises(ValueError, match="Recall is undefined"):
        recall_at_k(["a"], judgments, 1, relevant_grade=1)
    with pytest.raises(ValueError, match="nDCG is undefined"):
        ndcg_at_k(["a"], judgments, 1)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_cutoffs_are_rejected(k: int) -> None:
    for measure in (precision_at_k, recall_at_k):
        with pytest.raises(ValueError, match="positive integer"):
            measure(["a"], {"a": 1}, k, relevant_grade=1)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k(["a"], {"a": 1}, k)


@pytest.mark.parametrize("threshold", [0, 3, 4, 1.5, True])
def test_invalid_thresholds_are_rejected(threshold: int) -> None:
    for measure in (precision_at_k, recall_at_k):
        with pytest.raises(ValueError, match="explicit relevance threshold"):
            measure(["a"], {"a": 1}, 1, relevant_grade=threshold)
    with pytest.raises(ValueError, match="explicit relevance threshold"):
        reciprocal_rank(["a"], {"a": 1}, relevant_grade=threshold)
    with pytest.raises(ValueError, match="explicit relevance threshold"):
        mean_reciprocal_rank({"q": ["a"]}, {"q": {"a": 1}}, relevant_grade=threshold)


@pytest.mark.parametrize("grade", [-1, 3, 4, 1.0, True, None, "1"])
def test_invalid_grades_are_rejected(grade: int) -> None:
    with pytest.raises(ValueError, match="grades must be integers"):
        ndcg_at_k(["a"], {"a": grade}, 1)


@pytest.mark.parametrize("ranking,judgments,match", [
    (["a"], {}, "Missing human labels"),
    (["a", "missing"], {"a": 1}, "Missing human labels"),
    (["a", "a"], {"a": 1}, "duplicate"),
    ([""], {"a": 1}, "nonempty strings"),
    (["a"], {"": 1, "a": 1}, "nonempty strings"),
    ("a", {"a": 1}, "not a single string"),
])
def test_invalid_or_unjudged_runs_never_produce_results(
    ranking: list[str], judgments: dict[str, int], match: str,
) -> None:
    for measure in (precision_at_k, recall_at_k):
        with pytest.raises(ValueError, match=match):
            measure(ranking, judgments, 1, relevant_grade=1)
    with pytest.raises(ValueError, match=match):
        reciprocal_rank(ranking, judgments, relevant_grade=1)
    with pytest.raises(ValueError, match=match):
        ndcg_at_k(ranking, judgments, 1)


def test_missing_grade_beyond_cutoff_is_not_silently_discarded() -> None:
    # Full supplied runs must be judged, even when a particular metric cuts at 1.
    with pytest.raises(ValueError, match="Missing human labels"):
        precision_at_k(["a", "unjudged"], {"a": 1}, 1, relevant_grade=1)


@pytest.mark.parametrize("rankings,judgments", [
    ({}, {}), ({"q": ["a"]}, {}), ({}, {"q": {"a": 1}}),
    ({"q": ["a"]}, {"other": {"a": 1}}),
    ({"q": ["a"]}, {"q": {}}),
    ({"": ["a"]}, {"": {"a": 1}}),
])
def test_mrr_never_drops_missing_queries(
    rankings: dict[str, list[str]], judgments: dict[str, dict[str, int]],
) -> None:
    with pytest.raises(ValueError):
        mean_reciprocal_rank(rankings, judgments, relevant_grade=1)

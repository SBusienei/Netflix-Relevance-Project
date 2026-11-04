"""Cross-check definitions against an IR library, isolated from production metrics."""

import ir_measures
import pytest

from src.metrics import (
    mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)


@pytest.mark.parametrize("threshold", [1, 2])
@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_metrics_agree_with_ir_measures(threshold: int, k: int) -> None:
    # Fully judged synthetic runs avoid the library's implicit unjudged=0 policy.
    judgments = {
        "q1": {"a": 0, "b": 2, "c": 1, "d": 1, "e": 0},
        "q2": {"a": 1, "b": 0, "c": 2, "d": 1, "e": 0},
        "q3": {"a": 0, "b": 0, "c": 2, "d": 1, "e": 1},
    }
    rankings = {
        "q1": ["a", "c", "b"], "q2": ["c", "d", "b", "a"], "q3": ["a", "b"],
    }
    # Unique descending scores encode the exact input ordering, with no score ties.
    run = {
        query_id: {document_id: float(len(ranking) - rank) for rank, document_id in enumerate(ranking)}
        for query_id, ranking in rankings.items()
    }
    precision = ir_measures.P(rel=threshold) @ k
    recall = ir_measures.R(rel=threshold) @ k
    rr = ir_measures.RR(rel=threshold)
    ndcg = ir_measures.nDCG @ k
    measures = [precision, recall, rr, ndcg]
    per_query = {
        (result.query_id, result.measure): result.value
        for result in ir_measures.iter_calc(measures, judgments, run)
    }
    for query_id, ranking in rankings.items():
        qrels = judgments[query_id]
        assert precision_at_k(ranking, qrels, k, relevant_grade=threshold) == pytest.approx(
            per_query[(query_id, precision)],
        )
        assert recall_at_k(ranking, qrels, k, relevant_grade=threshold) == pytest.approx(
            per_query[(query_id, recall)],
        )
        assert reciprocal_rank(ranking, qrels, relevant_grade=threshold) == pytest.approx(
            per_query[(query_id, rr)],
        )
        assert ndcg_at_k(ranking, qrels, k) == pytest.approx(per_query[(query_id, ndcg)])
    aggregate = ir_measures.calc_aggregate([rr], judgments, run)
    assert mean_reciprocal_rank(rankings, judgments, relevant_grade=threshold) == pytest.approx(
        aggregate[rr],
    )

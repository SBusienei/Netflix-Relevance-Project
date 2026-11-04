"""Make retrieval measurements inspectable and refuse to infer missing human grades."""

import math
from collections.abc import Mapping, Sequence


Judgments = Mapping[str, int]
Rankings = Mapping[str, Sequence[str]]
QueryJudgments = Mapping[str, Judgments]


def _validate_cutoff(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")


def _validate_threshold(relevant_grade: int) -> None:
    if type(relevant_grade) is not int or relevant_grade not in (1, 2):
        raise ValueError("Choose an explicit relevance threshold of 1 or 2")


def _grades(ranking: Sequence[str], judgments: Judgments) -> list[int]:
    if not judgments:
        raise ValueError("Missing human labels: supply a nonempty judgment list")
    if isinstance(ranking, (str, bytes)):
        raise ValueError("Supply a sequence of document IDs, not a single string")
    for document_id, grade in judgments.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Judgment document IDs must be nonempty strings")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError("Human relevance grades must be integers from 0 to 2")
    if any(not isinstance(document_id, str) or not document_id.strip() for document_id in ranking):
        raise ValueError("Ranked document IDs must be nonempty strings")
    if len(set(ranking)) != len(ranking):
        raise ValueError("A ranking must not contain duplicate document IDs")
    missing = [document_id for document_id in ranking if document_id not in judgments]
    if missing:
        raise ValueError(
            f"Missing human labels for {len(missing)} ranked document(s): "
            f"{', '.join(missing)}. Stop and label them; no grades will be inferred."
        )
    return [judgments[document_id] for document_id in ranking]


def precision_at_k(
    ranking: Sequence[str], judgments: Judgments, k: int, *, relevant_grade: int,
) -> float:
    """Divide relevant returned documents in the first k positions by k."""
    _validate_cutoff(k)
    _validate_threshold(relevant_grade)
    grades = _grades(ranking, judgments)
    return sum(grade >= relevant_grade for grade in grades[:k]) / k


def recall_at_k(
    ranking: Sequence[str], judgments: Judgments, k: int, *, relevant_grade: int,
) -> float:
    """Measure coverage of relevant documents in the supplied human judgment list."""
    _validate_cutoff(k)
    _validate_threshold(relevant_grade)
    grades = _grades(ranking, judgments)
    relevant_count = sum(grade >= relevant_grade for grade in judgments.values())
    if relevant_count == 0:
        raise ValueError("Recall is undefined: no judged document meets the relevance threshold")
    return sum(grade >= relevant_grade for grade in grades[:k]) / relevant_count


def reciprocal_rank(
    ranking: Sequence[str], judgments: Judgments, *, relevant_grade: int,
) -> float:
    """Return the inverse rank of the first relevant hit, or zero when there is none."""
    _validate_threshold(relevant_grade)
    grades = _grades(ranking, judgments)
    for rank, grade in enumerate(grades, 1):
        if grade >= relevant_grade:
            return 1 / rank
    return 0.0


def mean_reciprocal_rank(
    rankings: Rankings, judgments: QueryJudgments, *, relevant_grade: int,
) -> float:
    """Macro-average reciprocal rank over exactly the explicitly supplied query set."""
    _validate_threshold(relevant_grade)
    if not rankings or set(rankings) != set(judgments):
        raise ValueError("MRR requires the same nonempty query IDs in rankings and judgments")
    if any(not isinstance(query_id, str) or not query_id.strip() for query_id in rankings):
        raise ValueError("Query IDs must be nonempty strings")
    return sum(
        reciprocal_rank(ranking, judgments[query_id], relevant_grade=relevant_grade)
        for query_id, ranking in rankings.items()
    ) / len(rankings)


def _dcg(grades: Sequence[int]) -> float:
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def ndcg_at_k(ranking: Sequence[str], judgments: Judgments, k: int) -> float:
    """Normalize discounted linear graded gain by the ideal judged ranking at k."""
    _validate_cutoff(k)
    grades = _grades(ranking, judgments)
    ideal = _dcg(sorted(judgments.values(), reverse=True)[:k])
    if ideal == 0:
        raise ValueError("nDCG is undefined: the judgment list contains no positive graded gain")
    return _dcg(grades[:k]) / ideal

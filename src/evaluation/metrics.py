"""MIND-style binary ranking metrics, without runtime third-party dependencies.

Reference formulas:
https://recommenders-team.github.io/recommenders/_modules/recommenders/models/deeprec/deeprec_utils.html
Ranking ties preserve input order (an explicit repository policy). Undefined
metrics return None, not zero. No rounding occurs inside metric calculations.
"""

from collections.abc import Sequence
from math import fsum, log2
from numbers import Integral

from src.data_loader.schema import validate_label_score

METRIC_NAMES = ("auc", "mrr", "ndcg@5", "ndcg@10")


def _validate(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("labels and scores must have equal, nonzero lengths")
    for label, score in zip(labels, scores):
        validate_label_score(label, score)


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Count positive-negative wins, giving tied pairs half credit; O(n log n)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    negative_below = 0
    wins = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group_positive = sum(labels[i] for i in order[start:end])
        group_negative = end - start - group_positive
        wins += group_positive * (negative_below + 0.5 * group_negative)
        negative_below += group_negative
        start = end
    return float(wins / (positives * negatives))


def _ranked(labels: Sequence[int], scores: Sequence[float]) -> list[int]:
    # Python's sort is stable, including reverse=True. Never inspect labels here.
    return [labels[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]


def _mrr(ranked: Sequence[int]) -> float | None:
    positives = sum(ranked)
    if not positives:
        return None
    return fsum(label / rank for rank, label in enumerate(ranked, 1)) / positives


def _ndcg(ranked: Sequence[int], k: int) -> float | None:
    positives = sum(ranked)
    if not positives:
        return None
    actual = fsum(label / log2(rank + 1) for rank, label in enumerate(ranked[:k], 1))
    ideal = fsum(1 / log2(rank + 1) for rank in range(1, min(k, positives) + 1))
    return actual / ideal


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    _validate(labels, scores)
    return _auc(labels, scores)


def mrr_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    _validate(labels, scores)
    return _mrr(_ranked(labels, scores))


def ndcg_score(labels: Sequence[int], scores: Sequence[float], k: int = 10) -> float | None:
    _validate(labels, scores)
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    return _ndcg(_ranked(labels, scores), k)


def compute_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | None]:
    _validate(labels, scores)
    ranked = _ranked(labels, scores)
    return {"auc": _auc(labels, scores), "mrr": _mrr(ranked),
            "ndcg@5": _ndcg(ranked, 5), "ndcg@10": _ndcg(ranked, 10)}

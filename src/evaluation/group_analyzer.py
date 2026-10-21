"""Impression cohorts: grouping never filters candidates or re-ranks an impression."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import floor

from src.data_loader.schema import ImpressionPrediction, require_count, same_time_basis


@dataclass(frozen=True)
class HistoryThresholds:
    q25: float | None
    q75: float | None
    n_train: int
    n_positive_history: int
    training_sources: tuple[str, ...]
    max_train_time: datetime

    @classmethod
    def fit(cls, train: Sequence[ImpressionPrediction]) -> "HistoryThresholds":
        """Fit inclusive linear quantiles of positive histories, per train impression.

        Missing lengths are excluded; zero lengths get their own group. No positive
        history => thresholds absent, positive evaluation histories remain unknown.
        """
        if not train or any(row.split != "train" for row in train):
            raise ValueError("threshold fitting requires nonempty train-only impressions")
        if len({row.key for row in train}) != len(train):
            raise ValueError("duplicate training impression key")
        for row in train:
            same_time_basis(row.timestamp, train[0].timestamp)
        values = sorted(row.history_length for row in train if row.history_length is not None
                        and row.history_length > 0)

        def quantile(q: float) -> float | None:
            if not values:
                return None
            position = (len(values) - 1) * q
            low = floor(position)
            high = min(low + 1, len(values) - 1)
            return float(values[low] + (values[high] - values[low]) * (position - low))

        return cls(quantile(0.25), quantile(0.75), len(train), len(values),
                   tuple(sorted({row.source for row in train})), max(row.timestamp for row in train))

    def group(self, length: int | None) -> str:
        if length is None:
            return "unknown"
        require_count(length, "history_length")
        if length == 0:
            return "empty"
        if self.q25 is None or self.q75 is None:
            return "unknown"
        if length <= self.q25:
            return "short"
        if length <= self.q75:
            return "medium"
        return "long"


def cohort_memberships(
    impression: ImpressionPrediction, thresholds: HistoryThresholds | None,
) -> dict[str, tuple[str, ...]]:
    counts = [c.prior_click_count for c in impression.candidates]
    if any(count is None for count in counts):
        cold_group = "unknown"
    elif all(count == 0 for count in counts):
        cold_group = "cold_only"
    elif all(count > 0 for count in counts):
        cold_group = "warm_only"
    else:
        cold_group = "mixed"
    # Prefix real categories so a literal category named "unknown" cannot collide.
    categories = {f"category:{c.category}" if c.category is not None else "unknown"
                  for c in impression.candidates}
    history = thresholds.group(impression.history_length) if thresholds else (
        "empty" if impression.history_length == 0 else "unknown")
    return {"history_length": (history,), "cold_warm": (cold_group,),
            "category_presence": tuple(sorted(categories))}

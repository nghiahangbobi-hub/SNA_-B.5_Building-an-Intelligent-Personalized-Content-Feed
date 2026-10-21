"""Intentionally leaky E1 counts. Never imported by the clean B2 implementation.

A positive label contributes one event at its IMPRESSION timestamp. MIND does
not provide the actual click time. History never contributes events.
"""

from collections import Counter
from collections.abc import Iterable

from src.baselines.base import BaseRecommender, ScoringContext
from src.data_loader.mind_reader import Behavior
from src.data_loader.temporal_split import TRAIN_START, TEST_END

CLICK_TIME_POLICY = (
    "positive_label_event_time=impression_timestamp; actual_click_time_unavailable; History_excluded"
)
LEAKY_PROTOCOLS = ("random", "gts_static")


class E1StaticPopularity(BaseRecommender):
    """Count the entire explicitly supplied source, without a prediction cutoff."""

    name = "B2_cumulative"

    def __init__(self, rows: Iterable[Behavior], *, protocol: str) -> None:
        if protocol not in LEAKY_PROTOCOLS:
            raise ValueError("E1 static counts require an explicitly leaky protocol")
        self.protocol = protocol
        self.counts: Counter[str] = Counter()
        self.impression_count = 0
        self.event_count = 0
        seen = set()
        for row in rows:
            if protocol == "random" and row.split not in {"train", "train_final"}:
                raise ValueError("random counts only accept train/train_final sources")
            if row.timestamp.utcoffset() is not None or not TRAIN_START <= row.timestamp < TEST_END:
                raise ValueError("E1 source must be in 09-15/11/2019, timezone naive")
            if row.key in seen:
                raise ValueError("duplicate click-source impression")
            seen.add(row.key)
            if (not row.candidate_ids or len(row.candidate_ids) != len(row.labels)
                    or len(set(row.candidate_ids)) != len(row.candidate_ids)):
                raise ValueError("invalid candidate/label alignment in E1 source")
            self.impression_count += 1
            for news_id, label in zip(row.candidate_ids, row.labels):
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("click labels must be integer 0/1")
                if label:
                    self.counts[news_id] += 1
                    self.event_count += 1

    def score(self, context: ScoringContext) -> tuple[float, ...]:
        return tuple(float(self.counts[news_id]) for news_id in context.candidate_ids)

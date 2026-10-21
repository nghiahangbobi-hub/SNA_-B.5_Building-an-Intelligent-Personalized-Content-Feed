"""Popularity of timestamped impression clicks only, queried at daily snapshots."""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from src.baselines.base import BaseRecommender, ScoringContext
from src.data_loader.mind_reader import Behavior
from src.data_loader.schema import same_time_basis
from src.data_loader.temporal_split import day_cutoff


class ClickIndex:
    """Immutable event source restricted to training partitions, with unique events.

    A repeated user-news click in a different impression is a distinct event.
    Duplicate impressions are errors, not silently deduplicated observations.
    """

    def __init__(self, rows: Iterable[Behavior], *, source_end: datetime) -> None:
        self.source_end = source_end
        self.event_count = 0
        self.impression_count = 0
        self.source_names: set[str] = set()
        self._times: dict[str, list[datetime]] = defaultdict(list)
        seen = set()
        for row in rows:
            if row.split not in {"train", "train_final"}:
                raise ValueError("click index only accepts train/train_final rows, never evaluation labels")
            same_time_basis(row.timestamp, source_end)
            if row.timestamp >= source_end:
                raise ValueError("event source must strictly precede source_end")
            if row.key in seen:
                raise ValueError("duplicate click-source impression")
            seen.add(row.key)
            if len(row.candidate_ids) != len(row.labels) or len(set(row.candidate_ids)) != len(row.labels):
                raise ValueError("invalid candidate/label alignment in click source")
            self.impression_count += 1
            self.source_names.add(row.source)
            for news_id, label in zip(row.candidate_ids, row.labels):
                if type(label) is not int or label not in {0, 1}:
                    raise ValueError("click labels must be integer 0/1")
                if label == 1:
                    self._times[news_id].append(row.timestamp)
                    self.event_count += 1
        for times in self._times.values():
            times.sort()

    def count(self, news_id: str, cutoff: datetime, *, window_hours: int | None = None) -> int:
        same_time_basis(cutoff, self.source_end)
        if window_hours not in (None, 24) or isinstance(window_hours, bool):
            raise ValueError("approved windows are cumulative (None) or 24 hours")
        times = self._times.get(news_id, ())
        upper = bisect_left(times, cutoff)
        lower = 0 if window_hours is None else bisect_left(times, cutoff - timedelta(hours=24))
        return upper - lower


class PopularityBaseline(BaseRecommender):
    def __init__(self, index: ClickIndex, *, window_hours: int | None = None) -> None:
        if window_hours not in (None, 24) or isinstance(window_hours, bool):
            raise ValueError("approved windows are cumulative (None) or 24 hours")
        self.index = index
        self.window_hours = window_hours
        self.name = "B2_cumulative" if window_hours is None else "B2_24h"

    def score(self, context: ScoringContext) -> tuple[float, ...]:
        cutoff = day_cutoff(context.timestamp)
        return tuple(float(self.index.count(news_id, cutoff, window_hours=self.window_hours))
                     for news_id in context.candidate_ids)

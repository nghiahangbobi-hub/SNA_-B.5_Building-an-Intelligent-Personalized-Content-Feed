"""Streaming MIND reader. History IDs have no invented click timestamps."""

import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator, Iterable

from src.baselines.base import ScoringContext


@dataclass(frozen=True, slots=True)
class Behavior:
    source: str
    impression_id: str
    user_id: str
    timestamp: datetime
    history_length: int
    candidate_ids: tuple[str, ...]
    labels: tuple[int, ...]
    split: str = "unassigned"
    history_ids: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.impression_id

    def context(self) -> ScoringContext:
        return ScoringContext(self.source, self.impression_id, self.timestamp,
                              self.candidate_ids, self.history_ids)


def read_behaviors(path: str | Path, *, source: str) -> Iterator[Behavior]:
    """Parse exactly five columns, fail with location; do not skip malformed rows."""
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            try:
                if len(fields) != 5:
                    raise ValueError("expected five behaviors columns")
                impression_id, user_id, time_text, history, impressions = fields
                if not source or not impression_id or not user_id or not impressions.strip():
                    raise ValueError("missing source/ID/candidates")
                if impression_id in seen:
                    raise ValueError("duplicate impression ID in source")
                seen.add(impression_id)
                candidate_ids, labels = [], []
                for token in impressions.split():
                    news_id, label = token.rsplit("-", 1)
                    if not news_id or label not in {"0", "1"}:
                        raise ValueError("expected news-ID-0/1 candidate")
                    candidate_ids.append(sys.intern(news_id))
                    labels.append(int(label))
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError("duplicate candidate ID")
                yield Behavior(source, impression_id, sys.intern(user_id),
                               datetime.strptime(time_text, "%m/%d/%Y %I:%M:%S %p"),
                               len(history.split()), tuple(candidate_ids), tuple(labels),
                               history_ids=tuple(sys.intern(n) for n in history.split()))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def read_categories(paths: Iterable[str | Path]) -> dict[str, str | None]:
    """Read only news ID/category; conflicting cross-split metadata is an error."""
    categories: dict[str, str | None] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for line_number, fields in enumerate(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
                if len(fields) != 8 or not fields[0]:
                    raise ValueError(f"{path}:{line_number}: expected eight news columns and ID")
                news_id = sys.intern(fields[0])
                category = sys.intern(fields[1]) if fields[1] else None
                if news_id in categories and categories[news_id] != category:
                    raise ValueError(f"{path}:{line_number}: conflicting category for {news_id}")
                categories[news_id] = category
    return categories


@dataclass(frozen=True, slots=True)
class NewsMetadata:
    category: str | None
    subcategory: str | None
    title: str
    abstract: str

    @property
    def text(self) -> str:
        """Strip outer whitespace only; preserve internal punctuation and spacing."""
        return ". ".join(part.strip() for part in (self.title, self.abstract) if part.strip())


def read_news(paths: Iterable[str | Path]) -> dict[str, NewsMetadata]:
    """Union content metadata; conflicting records are errors, never last-wins."""
    news: dict[str, NewsMetadata] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for line_number, fields in enumerate(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
                if len(fields) != 8 or not fields[0]:
                    raise ValueError(f"{path}:{line_number}: expected eight news columns and ID")
                item = NewsMetadata(fields[1].strip() or None, fields[2].strip() or None,
                                    fields[3], fields[4])
                news_id = sys.intern(fields[0])
                if news_id in news and news[news_id] != item:
                    raise ValueError(f"{path}:{line_number}: conflicting content for {news_id}")
                news[news_id] = item
    if not news:
        raise ValueError("news metadata is empty")
    return news

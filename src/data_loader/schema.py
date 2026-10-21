"""Validated prediction contracts; no TSV loading or click-history inference."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Integral, Real


def require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def require_count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def validate_label_score(label: object, score: object) -> None:
    if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
        raise ValueError("label must be integer 0 or 1")
    if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score):
        raise ValueError("score must be a finite real number")


def same_time_basis(left: datetime, right: datetime) -> None:
    """Do not silently mix timezone-naive MIND timestamps with aware timestamps."""
    if (left.utcoffset() is None) != (right.utcoffset() is None):
        raise ValueError("timestamps must share a naive/aware time basis")


@dataclass(frozen=True)
class CandidatePrediction:
    news_id: str
    label: int
    score: float
    category: str | None = None
    prior_click_count: int | None = None

    def __post_init__(self) -> None:
        require_text(self.news_id, "news_id")
        validate_label_score(self.label, self.score)
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "score", float(self.score))
        if self.category is not None:
            require_text(self.category, "category")
        if self.prior_click_count is not None:
            require_count(self.prior_click_count, "prior_click_count")
            object.__setattr__(self, "prior_click_count", int(self.prior_click_count))


@dataclass(frozen=True)
class ImpressionPrediction:
    """Candidates must retain their original order and contain every displayed item.

    source + impression_id is the unique key. The producer guarantees candidate
    completeness and that prior_click_count counts ONLY events < metadata_cutoff.
    This contract validates cutoff ordering, not the unprovided event history.
    """

    impression_id: str
    user_id: str
    timestamp: datetime
    candidates: tuple[CandidatePrediction, ...]
    split: str
    source: str
    history_length: int | None = None
    metadata_cutoff: datetime | None = None
    metadata_source: str | None = None

    def __post_init__(self) -> None:
        for name in ("impression_id", "user_id", "source"):
            require_text(getattr(self, name), name)
        if self.split not in {"train", "validation", "test", "train_final"}:
            raise ValueError("split must be train, validation, test, or train_final")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be datetime")
        try:
            candidates = tuple(self.candidates)
        except TypeError as exc:
            raise ValueError("candidates must be a nonempty sequence") from exc
        if not candidates or any(not isinstance(c, CandidatePrediction) for c in candidates):
            raise ValueError("candidates must contain CandidatePrediction objects")
        if len({c.news_id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate news_id within impression")
        object.__setattr__(self, "candidates", candidates)
        if self.history_length is not None:
            require_count(self.history_length, "history_length")
            object.__setattr__(self, "history_length", int(self.history_length))
        if self.metadata_cutoff is not None:
            if not isinstance(self.metadata_cutoff, datetime):
                raise ValueError("metadata_cutoff must be datetime")
            same_time_basis(self.timestamp, self.metadata_cutoff)
            if self.metadata_cutoff > self.timestamp:
                raise ValueError("metadata_cutoff must not be after impression timestamp")
            require_text(self.metadata_source, "metadata_source")
        elif self.metadata_source is not None:
            raise ValueError("metadata_source requires metadata_cutoff")
        if any(c.prior_click_count is not None for c in candidates):
            if self.metadata_cutoff is None:
                raise ValueError("prior click counts require metadata_cutoff and provenance")

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.impression_id

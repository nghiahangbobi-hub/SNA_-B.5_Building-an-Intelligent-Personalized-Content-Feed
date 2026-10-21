"""Label-free scoring interface; History is local to the current impression."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from src.data_loader.schema import require_text


@dataclass(frozen=True, slots=True)
class ScoringContext:
    source: str
    impression_id: str
    timestamp: datetime
    candidate_ids: tuple[str, ...]
    history_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.source, "source")
        require_text(self.impression_id, "impression_id")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be datetime")
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be nonempty and unique")
        for news_id in self.candidate_ids:
            require_text(news_id, "news_id")
        object.__setattr__(self, "history_ids", tuple(self.history_ids))
        for news_id in self.history_ids:
            require_text(news_id, "history news_id")


class BaseRecommender(ABC):
    """Scores are aligned to original candidates; ground-truth labels are not inputs."""

    name: str

    @abstractmethod
    def score(self, context: ScoringContext) -> tuple[float, ...]:
        raise NotImplementedError

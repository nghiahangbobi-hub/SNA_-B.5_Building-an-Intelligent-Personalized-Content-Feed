"""Seeded, order-independent pseudorandom scores using SHA-256 (not Python hash)."""

import hashlib
import json

from src.baselines.base import BaseRecommender, ScoringContext
from src.data_loader.schema import require_count


class RandomBaseline(BaseRecommender):
    name = "B1_random"

    def __init__(self, seed: int = 0) -> None:
        require_count(seed, "seed")
        self.seed = int(seed)

    def score(self, context: ScoringContext) -> tuple[float, ...]:
        scores = []
        for news_id in context.candidate_ids:
            identity = json.dumps([self.seed, context.source, context.impression_id, news_id],
                                  ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            bits = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") >> 11
            scores.append(bits / 2**53)
        return tuple(scores)

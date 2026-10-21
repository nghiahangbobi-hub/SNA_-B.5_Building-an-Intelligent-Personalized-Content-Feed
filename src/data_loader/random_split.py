"""E1-only impression split, independent of input iteration order."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace

from src.data_loader.mind_reader import Behavior
from src.data_loader.schema import require_count
from src.data_loader.temporal_split import TRAIN_START, TEST_END, TemporalPartitions

RANDOM_ALGORITHM = (
    "sort by SHA-256(UTF-8 compact JSON [seed,source,impression_id]), "
    "then (source,impression_id); floor(8*N/10), floor(N/10), remainder"
)


def random_impression_split(rows: Iterable[Behavior], *, seed: int) -> TemporalPartitions:
    """Keep whole impressions; never inspect labels, History, or candidate order."""
    require_count(seed, "seed")
    records = list(rows)
    if len({row.key for row in records}) != len(records):
        raise ValueError("duplicate (source, impression_id) in random split")
    for row in records:
        if row.timestamp.utcoffset() is not None or not TRAIN_START <= row.timestamp < TEST_END:
            raise ValueError("E1 random split requires naive timestamps in 09-15/11/2019")

    def order(row: Behavior) -> tuple[bytes, tuple[str, str]]:
        payload = json.dumps([int(seed), *row.key], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).digest(), row.key

    records.sort(key=order)
    n_train, n_validation = len(records) * 8 // 10, len(records) // 10
    boundary = n_train + n_validation
    if min(n_train, n_validation, len(records) - boundary) == 0:
        raise ValueError("E1 random split requires nonempty 80/10/10 partitions")
    return TemporalPartitions(
        [replace(row, split="train") for row in records[:n_train]],
        [replace(row, split="validation") for row in records[n_train:boundary]],
        [replace(row, split="test") for row in records[boundary:]],
    )

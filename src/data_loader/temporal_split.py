"""Proposal 4.2 GTS only: no random or intentionally leaky protocols here."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

from src.data_loader.mind_reader import Behavior

TRAIN_START = datetime(2019, 11, 9)
VALIDATION_START = datetime(2019, 11, 14)
TEST_START = datetime(2019, 11, 15)
TEST_END = datetime(2019, 11, 16)


def day_cutoff(timestamp: datetime) -> datetime:
    return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class TemporalPartitions:
    train: list[Behavior]
    validation: list[Behavior]
    test: list[Behavior]

    def final_train(self) -> list[Behavior]:
        return [replace(row, split="train_final") for row in self.train + self.validation]


def global_temporal_split(
    official_train: Iterable[Behavior], official_dev: Iterable[Behavior],
) -> TemporalPartitions:
    train, validation, test = [], [], []
    seen = set()
    for rows, origin in ((official_train, "train"), (official_dev, "dev")):
        for row in rows:
            if row.key in seen:
                raise ValueError("duplicate (source, impression_id) across inputs")
            seen.add(row.key)
            if row.timestamp.utcoffset() is not None:
                raise ValueError("MIND GTS expects naive source timestamps, no inferred timezone")
            if origin == "train" and TRAIN_START <= row.timestamp < TEST_START:
                target = train if row.timestamp < VALIDATION_START else validation
                target.append(replace(row, split="train" if target is train else "validation"))
            elif origin == "dev" and TEST_START <= row.timestamp < TEST_END:
                test.append(replace(row, split="test"))
            else:
                raise ValueError(f"{row.source}/{row.impression_id}: timestamp outside approved {origin} period")
    if not train or not validation or not test:
        raise ValueError("GTS requires nonempty train, validation, and test partitions")
    return TemporalPartitions(train, validation, test)

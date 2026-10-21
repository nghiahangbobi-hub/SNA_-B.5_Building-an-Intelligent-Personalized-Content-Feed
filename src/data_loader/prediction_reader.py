"""Strict external scores joined to authoritative MIND impressions."""

import csv
import json
from collections.abc import Iterable, Mapping, Iterator
from pathlib import Path

from src.baselines.b2_popularity import ClickIndex
from src.data_loader.mind_reader import Behavior
from src.data_loader.schema import CandidatePrediction, ImpressionPrediction, validate_label_score
from src.data_loader.temporal_split import day_cutoff

ScoreMap = dict[tuple[str, str], tuple[float, ...]]


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_prediction_scores(path: str | Path, expected: Iterable[Behavior]) -> ScoreMap:
    """Allow exported diagnostic columns, but never use them as ground truth.

    Require exactly one prediction per expected impression. Arrays must follow
    original candidate order, so tie behavior cannot change through reordering.
    If optional labels are present, they must also agree with the raw data.
    """
    rows = list(expected)
    truth = {row.key: row for row in rows}
    if not rows or len(truth) != len(rows):
        raise ValueError("expected impressions must be nonempty with unique keys")
    scores: ScoreMap = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        required = {"source", "impression_id", "candidate_ids", "scores"}
        if not fields or len(fields) != len(set(fields)) or not required <= set(fields):
            raise ValueError("prediction CSV requires unique source/impression_id/candidate_ids/scores columns")
        for line, record in enumerate(reader, 2):
            try:
                if None in record or any(value is None for value in record.values()):
                    raise ValueError("CSV row has missing/extra fields")
                key = record["source"], record["impression_id"]
                if key not in truth or key in scores:
                    raise ValueError("unexpected or duplicate impression key")
                original = truth[key]
                ids = json.loads(record["candidate_ids"], parse_constant=_reject_constant)
                values = json.loads(record["scores"], parse_constant=_reject_constant)
                if not isinstance(ids, list) or ids != list(original.candidate_ids):
                    raise ValueError("candidate IDs/order must exactly match raw impression")
                if not isinstance(values, list) or len(values) != len(ids):
                    raise ValueError("scores must be an array aligned with all candidates")
                for label, value in zip(original.labels, values):
                    validate_label_score(label, value)
                if "labels" in record:
                    labels = json.loads(record["labels"], parse_constant=_reject_constant)
                    if (not isinstance(labels, list) or any(type(v) is not int for v in labels)
                            or labels != list(original.labels)):
                        raise ValueError("optional labels disagree with raw impression")
                scores[key] = tuple(float(v) for v in values)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if len(scores) != len(truth):
        raise ValueError(f"missing predictions: expected {len(truth)}, received {len(scores)}")
    return scores


def external_predictions(
    rows: Iterable[Behavior], scores: Mapping[tuple[str, str], tuple[float, ...]],
    index: ClickIndex, categories: Mapping[str, str | None],
) -> Iterator[ImpressionPrediction]:
    """Rebuild group metadata with the existing clean observed-click definition."""
    provenance = ("adapter_verified_impression_label_counts;actual_click_time_unavailable;"
                  "History_excluded;source_end=" + index.source_end.isoformat())
    for row in rows:
        cutoff = day_cutoff(row.timestamp)
        values = scores[row.key]
        if len(values) != len(row.candidate_ids):
            raise ValueError("misaligned external scores")
        candidates = tuple(
            CandidatePrediction(nid, label, value, categories.get(nid), index.count(nid, cutoff))
            for nid, label, value in zip(row.candidate_ids, row.labels, values)
        )
        yield ImpressionPrediction(row.impression_id, row.user_id, row.timestamp, candidates,
                                   row.split, row.source, row.history_length, cutoff, provenance)

"""Shared evaluation API and auditable JSON/CSV reports, independent of models."""

import csv
import json
import platform
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import fsum
from pathlib import Path

from src.data_loader.schema import ImpressionPrediction, same_time_basis
from src.evaluation.group_analyzer import HistoryThresholds, cohort_memberships
from src.evaluation.metrics import METRIC_NAMES, compute_metrics

DEFAULT_CONFIG = {
    "metrics": list(METRIC_NAMES),
    "aggregation": "impression_macro",
    "ranking_ties": "candidate_order",
    "undefined_metric": "null_and_count",
    "invalid_prediction": "error",
    "history_quantiles": [0.25, 0.75],
    "history_quantile_method": "linear",
    "history_fit_split": "train",
    "cold_warm_grouping": "impression_composition",
    "category_grouping": "impression_presence_overlapping",
    "metadata_click_boundary": "strict_before_cutoff",
}


def _resolve_config(config: Mapping | None) -> dict:
    resolved = dict(DEFAULT_CONFIG if config is None else config)
    if resolved != DEFAULT_CONFIG:
        raise ValueError("config must match the approved PHUC-01 policies and keys")
    return json.loads(json.dumps(resolved))


def load_eval_config(path: str | Path = "configs/eval_config.yaml") -> dict:
    """YAML is optional for the core API; fail clearly if PyYAML is unavailable."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading YAML requires the approved PyYAML dependency") from exc
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    return _resolve_config(config)


def _summary(records: list[dict]) -> list[dict]:
    rows = []
    for metric in METRIC_NAMES:
        values = [record[metric] for record in records if record[metric] is not None]
        rows.append({"metric": metric, "mean": fsum(values) / len(values) if values else None,
                     "n_impressions": len(records), "n_valid": len(values),
                     "n_undefined": len(records) - len(values)})
    return rows


@dataclass
class EvaluationResult:
    per_impression: list[dict]
    overall: list[dict]
    groups: list[dict]
    metadata: dict


def evaluate(
    impressions: Iterable[ImpressionPrediction],
    *,
    train_impressions: Iterable[ImpressionPrediction] | None = None,
    config: Mapping | None = None,
    run_metadata: Mapping[str, str] | None = None,
) -> EvaluationResult:
    """Evaluate one split. Train inputs are used ONLY to fit grouping thresholds.

    Metadata assertions cannot prove that an upstream producer used no future
    events. Report metadata_source and cutoff for that producer's audit.
    """
    resolved = _resolve_config(config)
    rows = list(impressions)
    if not rows or any(not isinstance(row, ImpressionPrediction) for row in rows):
        raise ValueError("evaluation requires nonempty ImpressionPrediction inputs")
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("duplicate evaluation impression key (source, impression_id)")
    if len({row.split for row in rows}) != 1:
        raise ValueError("evaluate one split at a time")
    for row in rows:
        same_time_basis(row.timestamp, rows[0].timestamp)
    thresholds = None
    train = None
    if train_impressions is not None:
        train = list(train_impressions)
        if any(not isinstance(row, ImpressionPrediction) for row in train):
            raise ValueError("training rows must be ImpressionPrediction objects")
        thresholds = HistoryThresholds.fit(train)
        for row in rows:
            same_time_basis(row.timestamp, thresholds.max_train_time)
            if row.split != "train" and thresholds.max_train_time >= row.timestamp:
                raise ValueError("threshold training data must strictly precede evaluation")
        if rows[0].split != "train" and {r.key for r in train} & {r.key for r in rows}:
            raise ValueError("train and evaluation impression keys overlap")

    records = []
    cohorts: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        labels = [c.label for c in row.candidates]
        scores = [c.score for c in row.candidates]
        memberships = cohort_memberships(row, thresholds)
        record = {
            "source": row.source, "impression_id": row.impression_id,
            "user_id": row.user_id, "split": row.split, "timestamp": row.timestamp.isoformat(),
            "history_length": row.history_length, "n_candidates": len(row.candidates),
            "n_positive": sum(labels),
            "metadata_cutoff": row.metadata_cutoff.isoformat() if row.metadata_cutoff else None,
            "metadata_source": row.metadata_source,
            "candidate_ids": [c.news_id for c in row.candidates],
            "labels": labels, "scores": scores,
            "categories": [c.category for c in row.candidates],
            "prior_click_counts": [c.prior_click_count for c in row.candidates],
            "groups": memberships, **compute_metrics(labels, scores),
        }
        records.append(record)
        for dimension, names in memberships.items():
            for name in names:
                cohorts.setdefault((dimension, name), []).append(record)
    group_rows = []
    for (dimension, name), members in sorted(cohorts.items()):
        group_rows.extend({"dimension": dimension, "group": name, **summary}
                          for summary in _summary(members))

    fitted = asdict(thresholds) if thresholds else None
    if fitted:
        fitted["max_train_time"] = thresholds.max_train_time.isoformat()
        fitted["training_impression_keys"] = [row.key for row in train]
    libraries = {}
    for package in ("numpy", "scikit-learn", "PyYAML", "pytest"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    metadata = {"schema_version": "phuc01-v1", "config": resolved,
                "history_thresholds": fitted, "run": dict(run_metadata or {}),
                "python": platform.python_version(), "libraries": libraries,
                "invalid_predictions": "error; no silent candidate/impression removal",
                "category_groups_overlap": True,
                "undefined_reasons": {"auc": "no_positive_or_no_negative",
                                      "mrr": "no_positive", "ndcg@5": "no_positive",
                                      "ndcg@10": "no_positive"},
                "temporal_limitation": "producer guarantees event_time < metadata_cutoff"}
    return EvaluationResult(records, _summary(records), group_rows, metadata)


def write_tables(result: EvaluationResult, directory: str | Path) -> dict[str, Path]:
    """Write into a NEW run directory to avoid silently overwriting prior evidence.

    JSON contains nulls and full provenance. CSV uses empty fields for null;
    nested candidate arrays and group memberships are JSON-encoded cells.
    """
    payload = asdict(result)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, rows in (("per_impression", result.per_impression), ("overall", result.overall),
                       ("groups", result.groups)):
        path = directory / f"{name}.csv"
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False)
                                 if isinstance(value, (list, tuple, dict)) else value
                                 for key, value in row.items()})
        paths[name] = path
    path = directory / "report.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json_text + "\n")
    paths["report"] = path
    return paths

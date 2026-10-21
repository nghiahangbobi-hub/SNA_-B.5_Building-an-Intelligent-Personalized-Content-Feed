"""E1 aggregate comparisons; no changes to the shared metric definitions."""

from collections.abc import Iterable, Mapping
from math import isclose

from src.evaluation.metrics import METRIC_NAMES


def metric_delta(baseline: float | None, value: float | None) -> dict:
    if baseline is None or value is None:
        return {"absolute_delta": None, "relative_delta_pct": None, "reason": "undefined_metric"}
    delta = value - baseline
    return {
        "absolute_delta": delta,
        "relative_delta_pct": None if baseline == 0 else 100 * delta / baseline,
        "reason": "zero_baseline" if baseline == 0 else None,
    }


def compare_protocols(metrics: Iterable[Mapping]) -> list[dict]:
    rows = list(metrics)
    indexed = {(r["protocol"], r["split"], r["metric"]): r for r in rows}
    expected = {(p, s, m) for p in ("gts_clean", "random", "gts_static")
                for s in ("validation", "test") for m in METRIC_NAMES}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("comparison requires exactly four metrics for each E1 protocol/split")
    output = []
    for protocol in ("random", "gts_static"):
        for split in ("validation", "test"):
            for metric in METRIC_NAMES:
                baseline = indexed["gts_clean", split, metric]["mean"]
                value = indexed[protocol, split, metric]["mean"]
                output.append({
                    "protocol": protocol, "split": split, "metric": metric,
                    "reference_protocol": "gts_clean", "reference_mean": baseline, "mean": value,
                    **metric_delta(baseline, value),
                    "same_evaluation_set": protocol == "gts_static",
                    "interpretation": ("same_impressions; includes_evaluation_labels" if protocol == "gts_static"
                                       else "different_impressions; split_composition_and_temporal_leakage"),
                })
    return output


def compare_phuc02(metrics: Iterable[Mapping], reference: Iterable[Mapping], *, seed: int) -> list[dict]:
    """Compare all eight clean aggregates and valid/undefined denominators.

    CSV reference means are read without rounding. Tolerance is 1e-12 absolute,
    zero relative; report exact equality separately.
    """
    actual = {(r["split"], r["metric"]): r for r in metrics if r["protocol"] == "gts_clean"}
    expected = {}
    for row in reference:
        if row["model"] != "B2_cumulative" or int(row["seed"]) != seed:
            continue
        key = row["split"], row["metric"]
        if key in expected:
            raise ValueError("duplicate PHUC-02 reference metric")
        expected[key] = row
    keys = {(s, m) for s in ("validation", "test") for m in METRIC_NAMES}
    if set(actual) != keys or set(expected) != keys:
        raise ValueError("PHUC-02 comparison requires all eight B2 cumulative metrics")
    output = []
    for split, metric in sorted(keys):
        a, ref = actual[split, metric], expected[split, metric]
        baseline = None if ref["mean"] in ("", None) else float(ref["mean"])
        value = a["mean"]
        matching_mean = (baseline is value if baseline is None or value is None
                         else isclose(value, baseline, rel_tol=0, abs_tol=1e-12))
        matching_counts = all(int(ref[k]) == a[k] for k in ("n_impressions", "n_valid", "n_undefined"))
        output.append({
            "split": split, "metric": metric, "reference_mean": baseline, "mean": value,
            "absolute_difference": None if baseline is None or value is None else value - baseline,
            "exact_equal": value == baseline, "counts_match": matching_counts,
            "passed": matching_mean and matching_counts,
            "reason": None if matching_mean and matching_counts else "mean_or_denominator_mismatch",
        })
    return output

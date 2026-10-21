"""PHUC-04: cumulative Popularity under clean GTS, random, and leaky static GTS.

Run from repository root: python -m scripts.run_e1_leakage --help.
Positive labels are dated at the impression timestamp, NOT actual click time.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import gc
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.baselines.b2_popularity import ClickIndex, PopularityBaseline
from src.baselines.e1_popularity import CLICK_TIME_POLICY, E1StaticPopularity
from src.data_loader.mind_reader import read_behaviors, read_categories
from src.data_loader.random_split import RANDOM_ALGORITHM, random_impression_split
from src.data_loader.schema import CandidatePrediction, ImpressionPrediction
from src.data_loader.temporal_split import (
    VALIDATION_START, TEST_START, day_cutoff, global_temporal_split,
)
from src.evaluation.evaluator import evaluate, load_eval_config
from src.evaluation.leakage_comparison import compare_phuc02, compare_protocols

POLICY = {
    "ticket": "PHUC-04", "seed": 0, "model": "B2_cumulative",
    "protocols": ["gts_clean", "random", "gts_static"],
    "period": ["2019-11-09", "2019-11-16"], "clean_cutoff": "day_start_strict",
    "random_ratios": [80, 10, 10], "random_algorithm": "sha256_seed_source_impression_id",
    "random_test_source": "train_plus_validation",
    "static_source": "all_09_15_including_evaluation_labels",
    "click_timestamp": "impression_timestamp_not_actual_click_time",
    "history_in_popularity": False, "group_thresholds": "not_fitted_overall_comparison_only",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Exact types matter: bool/float must not silently stand in for an integer.
    if json.dumps(config, sort_keys=True) != json.dumps(POLICY, sort_keys=True):
        raise ValueError("E1 config differs from approved PHUC-04 policy")
    return config


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict))
                             else v for k, v in row.items()})


def partition_stats(rows) -> dict:
    return {
        "impressions": len(rows), "clicks": sum(sum(r.labels) for r in rows),
        "candidates": sum(len(r.labels) for r in rows),
        "min_impression_time": min(r.timestamp for r in rows).isoformat(),
        "max_impression_time": max(r.timestamp for r in rows).isoformat(),
    }


def source_audit(source_rows, evaluation_rows) -> dict:
    """Count source exposure, not just events belonging to a scored candidate.

    Pair counts count an event again for each evaluation impression exposed to it.
    Temporal violation measures use impression timestamps as an event-time proxy.
    """
    source_keys = {r.key for r in source_rows}
    events = sorted(r.timestamp for r in source_rows for label in r.labels if label == 1)
    at_or_after_day = [len(events) - bisect_left(events, day_cutoff(r.timestamp))
                       for r in evaluation_rows]
    after_impression = [len(events) - bisect_right(events, r.timestamp) for r in evaluation_rows]
    return {
        "source_impressions": len(source_rows), "source_positive_events": len(events),
        "source_evaluation_overlap_impressions": sum(r.key in source_keys for r in evaluation_rows),
        "evaluation_impressions_exposed_to_events_at_or_after_day_cutoff":
            sum(n > 0 for n in at_or_after_day),
        "event_evaluation_pairs_at_or_after_day_cutoff": sum(at_or_after_day),
        "evaluation_impressions_exposed_to_events_after_impression_time":
            sum(n > 0 for n in after_impression),
        "event_evaluation_pairs_after_impression_time": sum(after_impression),
        "timestamp_policy": CLICK_TIME_POLICY,
    }


def prediction_rows(rows, model, categories):
    """Score without labels; don't disguise leaky scores as prior-click metadata."""
    for row in rows:
        scores = model.score(row.context())
        if len(scores) != len(row.candidate_ids):
            raise ValueError("E1 scores must preserve the full candidate list")
        candidates = tuple(CandidatePrediction(nid, label, score, categories.get(nid))
                           for nid, label, score in zip(row.candidate_ids, row.labels, scores))
        yield ImpressionPrediction(row.impression_id, row.user_id, row.timestamp, candidates,
                                   row.split, row.source, row.history_length)


def _save_splits(directory, partitions, prefix):
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    stats = {}
    for split in ("train", "validation", "test"):
        rows = getattr(partitions, split)
        records.extend({"source": r.source, "impression_id": r.impression_id, "split": split,
                        "timestamp": r.timestamp.isoformat()} for r in rows)
        stats[split] = partition_stats(rows)
    path = directory / "membership.csv"
    write_csv(path, records)
    stats["train_final"] = partition_stats(partitions.train + partitions.validation)
    return {"identity": prefix, "membership": fingerprint(path), "partitions": stats}


def run_e1(
    *, train_path="data/raw/train/behaviors.tsv", dev_path="data/raw/dev/behaviors.tsv",
    news_paths=("data/raw/train/news.tsv", "data/raw/dev/news.tsv"),
    output_dir, predictions_dir, splits_dir,
    config_path="configs/e1_protocols.yaml", eval_config_path="configs/eval_config.yaml",
    reference_dir="outputs/tables/phuc02_mindsmall_seed0_2026-10-19",
    data_kind="MIND-small",
) -> dict:
    config, eval_config = load_config(config_path), load_eval_config(eval_config_path)
    destinations = [Path(p).resolve() for p in (output_dir, predictions_dir, splits_dir)]
    for i, path in enumerate(destinations):
        if path.exists():
            raise FileExistsError("use new E1 run directories; never overwrite prior evidence")
        if any(path == other or path in other.parents or other in path.parents
               for other in destinations[i + 1:]):
            raise ValueError("E1 output/prediction/split directories must be separate and nonnested")
    output_dir, predictions_dir, splits_dir = destinations
    if not news_paths:
        raise ValueError("at least one news.tsv required")
    input_paths = [Path(train_path), Path(dev_path), *(Path(p) for p in news_paths)]
    inputs = [fingerprint(p) for p in input_paths]
    reference_dir = Path(reference_dir)
    reference_manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    with (reference_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    # Same source files and evaluation policy are prerequisites for a useful regression comparison.
    if (reference_manifest["status"] != "completed" or reference_manifest["data_kind"] != data_kind
            or reference_manifest["seed"] != config["seed"]
            or reference_manifest["evaluation_config"] != eval_config
            or [i["sha256"] for i in reference_manifest["inputs"]] != [i["sha256"] for i in inputs]):
        raise ValueError("PHUC-02 reference provenance mismatch: inspect data/config/seed/status before acceptance")
    temporal = global_temporal_split(read_behaviors(train_path, source="MINDsmall_train"),
                                     read_behaviors(dev_path, source="MINDsmall_dev"))
    all_rows = temporal.train + temporal.validation + temporal.test
    random = random_impression_split(all_rows, seed=config["seed"])
    categories = read_categories(news_paths)
    repo = Path(__file__).resolve().parents[1]
    code_files = [
        "scripts/run_e1_leakage.py", "src/data_loader/random_split.py",
        "src/baselines/e1_popularity.py", "src/evaluation/leakage_comparison.py",
        "src/baselines/base.py", "src/baselines/b2_popularity.py", "src/data_loader/mind_reader.py",
        "src/data_loader/schema.py", "src/data_loader/temporal_split.py",
        "src/evaluation/evaluator.py", "src/evaluation/metrics.py", "src/evaluation/group_analyzer.py",
        "requirements.txt", "tests/test_e1_leakage.py",
    ]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    manifest = {
        "status": "running", "ticket": "PHUC-04", "data_kind": data_kind, "seed": config["seed"],
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "resolved_config": config, "evaluation_config": eval_config, "inputs": inputs,
        "config_files": [fingerprint(Path(p)) for p in (config_path, eval_config_path)],
        "code_sha256": {p: fingerprint(repo / p)["sha256"] for p in code_files},
        "git_base_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "timestamp_policy": CLICK_TIME_POLICY, "random_algorithm": RANDOM_ALGORITHM,
        "group_policy": "overall only; no thresholds fitted; prior_click_count and metadata_cutoff absent",
        "reference_files": [fingerprint(reference_dir / p) for p in ("metrics.csv", "manifest.json")],
        "reference_comparison_tolerance": {"absolute": 1e-12, "relative": 0},
        "splits": {}, "runs": [],
    }
    for path in destinations:
        path.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    combined = []
    try:
        for protocol in config["protocols"]:
            partitions = random if protocol == "random" else temporal
            manifest["splits"][protocol] = _save_splits(splits_dir / protocol, partitions, protocol)
            for split in ("validation", "test"):
                evaluation_rows = getattr(partitions, split)
                if protocol == "gts_static":
                    source_rows = all_rows
                else:
                    source_rows = partitions.train if split == "validation" else partitions.final_train()
                audit = source_audit(source_rows, evaluation_rows)
                if protocol != "gts_static" and audit["source_evaluation_overlap_impressions"]:
                    raise ValueError("unexpected evaluation-label inclusion outside static E1")
                if protocol == "gts_clean":
                    source_end = VALIDATION_START if split == "validation" else TEST_START
                    model = PopularityBaseline(ClickIndex(source_rows, source_end=source_end))
                    if audit["event_evaluation_pairs_at_or_after_day_cutoff"]:
                        raise ValueError("clean E1 source contains events at/after daily cutoff")
                    source_policy = "train_09_13" if split == "validation" else "train_09_14"
                else:
                    model = E1StaticPopularity(source_rows, protocol=protocol)
                    source_policy = ("all_09_15_including_evaluation_labels" if protocol == "gts_static"
                                     else "random_train" if split == "validation" else "random_train_plus_validation")
                metadata = {
                    "protocol": protocol, "model": "B2_cumulative", "data_kind": data_kind,
                    "seed": str(config["seed"]), "leakage": str(protocol != "gts_clean"),
                    "click_source": source_policy, "timestamp_policy": CLICK_TIME_POLICY,
                    "manifest": str(manifest_path),
                }
                result = evaluate(prediction_rows(evaluation_rows, model, categories),
                                  config=eval_config, run_metadata=metadata)
                table_path, pred_path = output_dir / protocol / split, predictions_dir / protocol / split
                table_path.mkdir(parents=True, exist_ok=False)
                pred_path.mkdir(parents=True, exist_ok=False)
                write_csv(table_path / "overall.csv", result.overall)
                # Groups are not a PHUC-04 deliverable; no thresholds or leaky prior counts are supplied.
                write_csv(pred_path / "per_impression.csv", result.per_impression)
                write_json(pred_path / "metadata.json", result.metadata)
                combined.extend({"protocol": protocol, "split": split, "model": "B2_cumulative",
                                 "seed": config["seed"], "intentional_leakage": protocol != "gts_clean",
                                 **row} for row in result.overall)
                manifest["runs"].append({
                    **metadata, "split": split, "source_audit": audit, "overall": result.overall,
                    "libraries": result.metadata["libraries"],
                    "predictions": fingerprint(pred_path / "per_impression.csv"),
                    "prediction_metadata": fingerprint(pred_path / "metadata.json"),
                    "overall_file": fingerprint(table_path / "overall.csv"),
                })
                write_json(manifest_path, manifest)
                print(json.dumps({"completed": f"{protocol}/{split}", "overall": result.overall}), flush=True)
                del result, model, source_rows
                gc.collect()
            if protocol == "gts_clean":
                comparison = compare_phuc02(combined, reference, seed=config["seed"])
                write_csv(output_dir / "phuc02_comparison.csv", comparison)
                manifest["phuc02_comparison"] = {
                    "passed": all(r["passed"] for r in comparison),
                    "all_exact_equal": all(r["exact_equal"] for r in comparison), "rows": comparison,
                }
                write_json(manifest_path, manifest)
                if not manifest["phuc02_comparison"]["passed"]:
                    raise ValueError("clean GTS differs from PHUC-02; inspect phuc02_comparison.csv before acceptance")
        write_csv(output_dir / "metrics.csv", combined)
        write_csv(output_dir / "inflation.csv", compare_protocols(combined))
        manifest["summary_files"] = [fingerprint(output_dir / p) for p in
                                     ("metrics.csv", "inflation.csv", "phuc02_comparison.csv")]
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-behaviors", default="data/raw/train/behaviors.tsv")
    parser.add_argument("--dev-behaviors", default="data/raw/dev/behaviors.tsv")
    parser.add_argument("--news", nargs="+", default=["data/raw/train/news.tsv", "data/raw/dev/news.tsv"])
    parser.add_argument("--config", default="configs/e1_protocols.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--reference-dir", default="outputs/tables/phuc02_mindsmall_seed0_2026-10-19",
                        help="Completed PHUC-02 run with matching inputs/config/seed; required for acceptance")
    parser.add_argument("--output-dir", required=True, help="New summary directory")
    parser.add_argument("--predictions-dir", required=True, help="New E1-only prediction directory")
    parser.add_argument("--splits-dir", required=True, help="New E1-only split-membership directory")
    parser.add_argument("--data-kind", choices=["MIND-small", "synthetic"], default="MIND-small")
    args = parser.parse_args()
    run_e1(train_path=args.train_behaviors, dev_path=args.dev_behaviors, news_paths=args.news,
           output_dir=args.output_dir, predictions_dir=args.predictions_dir, splits_dir=args.splits_dir,
           config_path=args.config, eval_config_path=args.eval_config,
           reference_dir=args.reference_dir, data_kind=args.data_kind)


if __name__ == "__main__":
    main()

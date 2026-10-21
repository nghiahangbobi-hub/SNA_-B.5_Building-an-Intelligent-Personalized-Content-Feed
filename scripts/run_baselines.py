"""GTS CLI: B1/two B2s by default; explicit content config selects B3/B4.

Run from repository root: python -m scripts.run_baselines --help.
Only timestamped training-impression clicks enter popularity; never History/dev labels.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.baselines.base import BaseRecommender
from src.baselines.b1_random import RandomBaseline
from src.baselines.b2_popularity import ClickIndex, PopularityBaseline
from src.baselines.b3_category import CategoryBaseline
from src.baselines.b4_semantic_plm import EmbeddingStore, SemanticBaseline, load_content_config
from src.data_loader.mind_reader import Behavior, read_behaviors, read_categories, read_news
from src.data_loader.schema import CandidatePrediction, ImpressionPrediction, require_count
from src.data_loader.temporal_split import global_temporal_split, day_cutoff, VALIDATION_START, TEST_START
from src.evaluation.evaluator import evaluate, load_eval_config

POLICY = {
    "seed": 0, "models": ["B1_random", "B2_cumulative", "B2_24h"],
    "cutoff": "day_start", "window_hours": 24,
    "click_source": "timestamped_impression_clicks_only", "history_in_popularity": False,
    "ranking_ties": "candidate_order", "train_start": "2019-11-09",
    "validation_start": "2019-11-14", "test_start": "2019-11-15", "test_end": "2019-11-16",
}


def load_config(path: str | Path, seed: int | None = None) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or set(config) != set(POLICY):
        raise ValueError("baseline config must contain exactly the approved PHUC-02 keys")
    require_count(config["seed"], "seed")
    if seed is not None:
        require_count(seed, "seed")
        config["seed"] = seed
    if {k: v for k, v in config.items() if k != "seed"} != {k: v for k, v in POLICY.items() if k != "seed"}:
        raise ValueError("baseline protocol differs from the approved PHUC-02 policy")
    return config


def prediction_rows(
    rows: list[Behavior], model: BaseRecommender, index: ClickIndex,
    categories: dict[str, str | None],
):
    """Label-free scoring first; labels are joined only for evaluator output."""
    provenance = ("timestamped_impression_clicks_only;History_excluded;source_end="
                  + index.source_end.isoformat() + ";sources=" + ",".join(sorted(index.source_names)))
    for row in rows:
        context = row.context()
        scores = model.score(context)
        if len(scores) != len(row.candidate_ids):
            raise ValueError("model returned misaligned candidate scores")
        cutoff = day_cutoff(row.timestamp)
        candidates = tuple(CandidatePrediction(news_id, label, score, categories.get(news_id),
                                               index.count(news_id, cutoff))
                           for news_id, label, score in zip(row.candidate_ids, row.labels, scores))
        yield ImpressionPrediction(row.impression_id, row.user_id, row.timestamp, candidates,
                                   row.split, row.source, row.history_length, cutoff, provenance)


def threshold_inputs(rows: list[Behavior], categories: dict[str, str | None]) -> list[ImpressionPrediction]:
    """Retain full real candidate lists for the existing evaluator's train schema.

    These rows only fit history quantiles; their neutral scores are NOT evaluated
    or reported as baseline predictions. Share frozen candidates to bound memory.
    """
    cache: dict[tuple[str, int], CandidatePrediction] = {}
    result = []
    for row in rows:
        candidates = []
        for news_id, label in zip(row.candidate_ids, row.labels):
            key = news_id, label
            if key not in cache:
                cache[key] = CandidatePrediction(news_id, label, 0.0, categories.get(news_id))
            candidates.append(cache[key])
        result.append(ImpressionPrediction(row.impression_id, row.user_id, row.timestamp,
                                           tuple(candidates), row.split, row.source, row.history_length))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, tuple, dict)) else value
                             for key, value in row.items()})


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def partition_stats(rows: list[Behavior]) -> dict:
    return {"impressions": len(rows), "clicks": sum(sum(row.labels) for row in rows),
            "candidates": sum(len(row.labels) for row in rows),
            "min_time": min(row.timestamp for row in rows).isoformat(),
            "max_time": max(row.timestamp for row in rows).isoformat()}


def run_baselines(
    *, train_path: str | Path, dev_path: str | Path, news_paths: list[str | Path],
    output_dir: str | Path, predictions_dir: str | Path,
    config_path: str | Path = "configs/baselines_config.yaml",
    eval_config_path: str | Path = "configs/eval_config.yaml",
    seed: int | None = None, data_kind: str = "MIND-small",
    content_config_path: str | Path | None = None,
    embedding_cache: str | Path | None = None, models: list[str] | None = None,
) -> dict:
    config = load_config(config_path, seed)
    eval_config = load_eval_config(eval_config_path)
    content_config = load_content_config(content_config_path) if content_config_path is not None else None
    if content_config is None and (models is not None or embedding_cache is not None):
        raise ValueError("--models/--embedding-cache require --content-config")
    selected = config["models"] if content_config is None else (
        content_config["models"] if models is None else models)
    if content_config is not None:
        if config["seed"] != content_config["seed"]:
            raise ValueError("content seed differs from approved policy")
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(content_config["models"]):
            raise ValueError("select a nonempty unique subset of approved content models")
        if "B4_semantic_plm" in selected and embedding_cache is None:
            raise ValueError("B4 requires a completed embedding cache")
    output_dir, predictions_dir = Path(output_dir).resolve(), Path(predictions_dir).resolve()
    if output_dir == predictions_dir or output_dir in predictions_dir.parents or predictions_dir in output_dir.parents:
        raise ValueError("summary and prediction directories must be separate, nonnested paths")
    if output_dir.exists() or predictions_dir.exists():
        raise FileExistsError("use new run directories; existing evidence will not be overwritten")
    files = [Path(train_path), Path(dev_path), *(Path(p) for p in news_paths)]
    if not news_paths:
        raise ValueError("at least one news.tsv is required")
    inputs = [fingerprint(path) for path in files]
    news = read_news(news_paths) if content_config is not None else None
    store, cache_manifest = None, None
    if content_config is not None and "B4_semantic_plm" in selected:
        store, cache_manifest = EmbeddingStore.load(embedding_cache, news, content_config, data_kind=data_kind)
        cache_manifest = {**cache_manifest, "path": str(Path(embedding_cache).resolve()),
                          "manifest_sha256": fingerprint(Path(embedding_cache) / "manifest.json")["sha256"]}
    partitions = global_temporal_split(read_behaviors(train_path, source="MINDsmall_train"),
                                       read_behaviors(dev_path, source="MINDsmall_dev"))
    categories = read_categories(news_paths) if news is None else {n: item.category for n, item in news.items()}
    train_for_groups = threshold_inputs(partitions.train, categories)
    repo = Path(__file__).resolve().parents[1]
    code_files = ["scripts/run_baselines.py", "src/baselines/base.py", "src/baselines/b1_random.py",
                  "src/baselines/b2_popularity.py", "src/data_loader/mind_reader.py",
                  "src/data_loader/temporal_split.py", "src/data_loader/schema.py",
                  "src/evaluation/evaluator.py", "src/evaluation/metrics.py",
                  "src/evaluation/group_analyzer.py", "requirements.txt"]
    if content_config is not None:
        code_files.extend(["src/baselines/b3_category.py", "src/baselines/b4_semantic_plm.py",
                           "scripts/cache_plm_embeddings.py"])
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                  capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {
        "status": "running", "data_kind": data_kind, "seed": config["seed"],
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "baseline_config": config, "evaluation_config": eval_config,
        "selected_models": selected, "content_config": content_config, "embedding_cache": cache_manifest,
        "inputs": inputs, "git_base_revision": revision,
        "code_sha256": {name: fingerprint(repo / name)["sha256"] for name in code_files},
        "source_policy": "History excluded; cumulative observed impression clicks, not full Ghist",
        "cold_warm_policy": "cumulative observed clicks even for B2_24h",
        "history_threshold_source": "train 09-13 only; unchanged for validation and test",
        "content_history_policy": ("current impression History only; all entries, duplicates retained; "
                                   "assumed before impression, not proven before day-start; no label updates")
                                  if content_config is not None else None,
        "random_algorithm": "sha256([seed,source,impression_id,news_id]); top 53 bits / 2**53",
        "splits": {name: partition_stats(rows) for name, rows in
                   (("train", partitions.train), ("validation", partitions.validation),
                    ("train_final", partitions.train + partitions.validation), ("test", partitions.test))},
        "news_count": len(categories), "runs": [],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")

    save_manifest()
    combined = []
    print(json.dumps({"splits": manifest["splits"]}), flush=True)
    try:
        for split, evaluation_rows, source_rows, source_end in (
            ("validation", partitions.validation, partitions.train, VALIDATION_START),
            ("test", partitions.test, partitions.final_train(), TEST_START),
        ):
            index = ClickIndex(source_rows, source_end=source_end)
            if content_config is None:
                active_models = [RandomBaseline(config["seed"]), PopularityBaseline(index),
                                 PopularityBaseline(index, window_hours=24)]
            else:
                active_models = [CategoryBaseline(news) if name == "B3_category" else SemanticBaseline(news, store)
                                 for name in selected]
            for model in active_models:
                result = evaluate(prediction_rows(evaluation_rows, model, index, categories),
                                  train_impressions=train_for_groups, config=eval_config,
                                  run_metadata={"data_kind": data_kind, "model": model.name,
                                                "seed": str(config["seed"]), "manifest": str(manifest_path),
                                                "source_end": source_end.isoformat(),
                                                "history_policy": manifest["content_history_policy"] or "not_used",
                                                "embedding_cache_manifest_sha256":
                                                    cache_manifest["manifest_sha256"] if cache_manifest else "not_used"})
                table_path = output_dir / split / model.name
                prediction_path = predictions_dir / split / model.name
                table_path.mkdir(parents=True, exist_ok=False)
                prediction_path.mkdir(parents=True, exist_ok=False)
                write_csv(table_path / "overall.csv", result.overall)
                write_csv(table_path / "groups.csv", result.groups)
                write_csv(prediction_path / "per_impression.csv", result.per_impression)
                (prediction_path / "metadata.json").write_text(
                    json.dumps(result.metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
                thresholds = {k: v for k, v in result.metadata["history_thresholds"].items()
                              if k != "training_impression_keys"}
                details = {"split": split, "model": model.name, "seed": config["seed"],
                           "source_end": source_end.isoformat(), "click_source_impressions": index.impression_count,
                           "click_source_events": index.event_count, "overall": result.overall,
                           "history_thresholds": thresholds, "libraries": result.metadata["libraries"],
                           "predictions": str(prediction_path / "per_impression.csv"),
                           "content_diagnostics": dict(getattr(model, "diagnostics", {})),
                           "missing_category_candidates": sum(c is None for r in result.per_impression
                                                               for c in r["categories"])}
                manifest["runs"].append(details)
                combined.extend({"split": split, "model": model.name, "seed": config["seed"], **m}
                                for m in result.overall)
                save_manifest()
                print(json.dumps({"completed": f"{split}/{model.name}", "overall": result.overall}), flush=True)
                del result
                gc.collect()
        write_csv(output_dir / "metrics.csv", combined)
        manifest["status"] = "completed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save_manifest()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-behaviors", default="data/raw/train/behaviors.tsv")
    parser.add_argument("--dev-behaviors", default="data/raw/dev/behaviors.tsv")
    parser.add_argument("--news", nargs="+", default=["data/raw/train/news.tsv", "data/raw/dev/news.tsv"])
    parser.add_argument("--config", default="configs/baselines_config.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-kind", default="MIND-small")
    parser.add_argument("--content-config", help="Use approved PHUC-03 config instead of B1/B2 selection")
    parser.add_argument("--embedding-cache", help="Completed frozen PLM cache, required for B4")
    parser.add_argument("--models", nargs="+", choices=["B3_category", "B4_semantic_plm"],
                        help="Content-model subset; defaults to both when --content-config is supplied")
    args = parser.parse_args()
    run_baselines(train_path=args.train_behaviors, dev_path=args.dev_behaviors, news_paths=args.news,
                  output_dir=args.output_dir, predictions_dir=args.predictions_dir,
                  config_path=args.config, eval_config_path=args.eval_config, seed=args.seed,
                  data_kind=args.data_kind, content_config_path=args.content_config,
                  embedding_cache=args.embedding_cache, models=args.models)


if __name__ == "__main__":
    main()

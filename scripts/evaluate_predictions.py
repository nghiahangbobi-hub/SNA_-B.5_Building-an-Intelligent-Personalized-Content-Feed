"""Evaluate external model scores on complete clean-GTS MIND validation or test.

CSV requires source, impression_id, candidate_ids, scores (arrays encoded as JSON).
Raw data supplies labels and group metadata; model provenance remains a declaration.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_baselines import fingerprint, partition_stats, threshold_inputs, write_csv
from src.baselines.b2_popularity import ClickIndex
from src.data_loader.mind_reader import read_behaviors, read_categories
from src.data_loader.prediction_reader import external_predictions, read_prediction_scores
from src.data_loader.schema import require_count, require_text
from src.data_loader.temporal_split import VALIDATION_START, TEST_START, global_temporal_split
from src.evaluation.evaluator import evaluate, load_eval_config


def load_provenance(path: str | Path, *, model: str, seed: int) -> dict:
    require_text(model, "model")
    require_count(seed, "seed")
    declaration = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(declaration, dict):
        raise ValueError("model provenance must be a JSON object")
    if declaration.get("model") != model or declaration.get("seed") != seed:
        raise ValueError("model/seed disagree with provenance")
    require_count(declaration["seed"], "declared seed")
    for field in ("training_data", "code_revision"):
        require_text(declaration.get(field), field)
    if not isinstance(declaration.get("model_config"), dict):
        raise ValueError("model_config must be a JSON object")
    # Reject nonstandard JSON constants, including those in nested model settings.
    json.dumps(declaration, allow_nan=False)
    return declaration


def evaluate_external(
    *, predictions_path, model, seed, provenance_path, split, output_dir, predictions_dir,
    train_path="data/raw/train/behaviors.tsv", dev_path="data/raw/dev/behaviors.tsv",
    news_paths=("data/raw/train/news.tsv", "data/raw/dev/news.tsv"),
    eval_config_path="configs/eval_config.yaml", data_kind="MIND-small",
) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("external CLI supports clean GTS validation/test only")
    declaration = load_provenance(provenance_path, model=model, seed=seed)
    config = load_eval_config(eval_config_path)
    output_dir, predictions_dir = Path(output_dir).resolve(), Path(predictions_dir).resolve()
    if (output_dir == predictions_dir or output_dir in predictions_dir.parents
            or predictions_dir in output_dir.parents):
        raise ValueError("output directories must be separate and nonnested")
    if output_dir.exists() or predictions_dir.exists():
        raise FileExistsError("use new output directories; existing evidence is preserved")
    if not news_paths:
        raise ValueError("at least one news file is required")
    inputs = [fingerprint(Path(p)) for p in
              (train_path, dev_path, *news_paths, predictions_path, provenance_path, eval_config_path)]
    partitions = global_temporal_split(read_behaviors(train_path, source="MINDsmall_train"),
                                       read_behaviors(dev_path, source="MINDsmall_dev"))
    rows = getattr(partitions, split)
    scores = read_prediction_scores(predictions_path, rows)
    categories = read_categories(news_paths)
    source = partitions.train if split == "validation" else partitions.final_train()
    source_end = VALIDATION_START if split == "validation" else TEST_START
    index = ClickIndex(source, source_end=source_end)
    repo = Path(__file__).resolve().parents[1]
    code_files = [
        "scripts/evaluate_predictions.py", "src/data_loader/prediction_reader.py",
        "scripts/run_baselines.py", "src/data_loader/mind_reader.py",
        "src/data_loader/schema.py", "src/data_loader/temporal_split.py",
        "src/baselines/b2_popularity.py", "src/evaluation/evaluator.py",
        "src/evaluation/metrics.py", "src/evaluation/group_analyzer.py",
    ]
    manifest = {
        "status": "running", "ticket": "PHUC-05", "data_kind": data_kind, "model": model,
        "seed": seed, "split": split, "protocol": "gts_clean",
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "inputs": inputs, "evaluation_config": config, "model_declaration": declaration,
        "verification_boundary": "raw alignment and evaluator metadata verified; model training leakage not verified",
        "code_sha256": {p: fingerprint(repo / p)["sha256"] for p in code_files},
        "split_stats": partition_stats(rows), "source_stats": partition_stats(source),
        "source_end": source_end.isoformat(),
        "timestamp_policy": "positive label uses impression time, not actual click time; History excluded from counts",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=False)

    def save():
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        result = evaluate(
            external_predictions(rows, scores, index, categories),
            train_impressions=threshold_inputs(partitions.train, categories), config=config,
            run_metadata={"model": model, "seed": str(seed), "data_kind": data_kind,
                          "protocol": "gts_clean", "model_provenance": str(provenance_path),
                          "upstream_training_verification": "declaration_only",
                          "manifest": str(output_dir / "manifest.json")},
        )
        write_csv(output_dir / "overall.csv", result.overall)
        write_csv(output_dir / "groups.csv", result.groups)
        write_csv(predictions_dir / "per_impression.csv", result.per_impression)
        (predictions_dir / "metadata.json").write_text(
            json.dumps(result.metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        manifest.update(
            status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
            overall=result.overall, libraries=result.metadata["libraries"],
            history_thresholds={k: v for k, v in result.metadata["history_thresholds"].items()
                                if k != "training_impression_keys"},
            artifacts=[fingerprint(p) for p in (output_dir / "overall.csv", output_dir / "groups.csv",
                                                predictions_dir / "per_impression.csv",
                                                predictions_dir / "metadata.json")],
        )
        save()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-provenance", required=True, help="JSON declaration; see docs/HANDOFF.md")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--train-behaviors", default="data/raw/train/behaviors.tsv")
    parser.add_argument("--dev-behaviors", default="data/raw/dev/behaviors.tsv")
    parser.add_argument("--news", nargs="+", default=["data/raw/train/news.tsv", "data/raw/dev/news.tsv"])
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    parser.add_argument("--data-kind", choices=["MIND-small", "synthetic"], default="MIND-small")
    args = parser.parse_args()
    manifest = evaluate_external(
        predictions_path=args.predictions, model=args.model, seed=args.seed,
        provenance_path=args.model_provenance, split=args.split,
        output_dir=args.output_dir, predictions_dir=args.predictions_dir,
        train_path=args.train_behaviors, dev_path=args.dev_behaviors, news_paths=args.news,
        eval_config_path=args.eval_config, data_kind=args.data_kind,
    )
    print(json.dumps({"status": manifest["status"], "overall": manifest["overall"]}))


if __name__ == "__main__":
    main()

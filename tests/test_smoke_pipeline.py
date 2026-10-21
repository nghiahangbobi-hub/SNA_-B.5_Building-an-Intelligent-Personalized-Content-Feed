"""PHUC-05: two complete offline synthetic runs, including CLI handoff."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.run_baselines import run_baselines
from scripts.run_e1_leakage import run_e1
from src.baselines.b4_semantic_plm import CACHE_POLICY_KEYS, CONTENT_POLICY, content_sha256, file_sha256
from src.data_loader.mind_reader import read_news


def synthetic_inputs(path):
    path.mkdir()
    train, dev, news = [path / p for p in ("train.tsv", "dev.tsv", "news.tsv")]
    lines = {"train": [], "dev": []}
    for i in range(28):
        day = 9 + i % 7
        candidates = "A-1 B-0 C-1" if i % 3 else "A-0 B-1 C-0"
        history = "A B" if i % 2 else ""
        lines["dev" if day == 15 else "train"].append(
            f"{i}\tU{i % 3}\t11/{day}/2019 12:00:00 PM\t{history}\t{candidates}\n")
    train.write_text("".join(lines["train"]), encoding="utf-8")
    dev.write_text("".join(lines["dev"]), encoding="utf-8")
    news.write_text("A\tsports\tfootball\tA title\t\turl\t[]\t[]\n"
                    "B\ttravel\ttravel\tB title\t\turl\t[]\t[]\n"
                    "C\tsports\ttennis\tC title\t\turl\t[]\t[]\n", encoding="utf-8")
    cache = path / "synthetic_cache"
    cache.mkdir()
    metadata = read_news([news])
    np.save(cache / "embeddings.npy", np.eye(3, 384, dtype=np.float32), allow_pickle=False)
    (cache / "news_ids.json").write_text(json.dumps(["A", "B", "C"]), encoding="utf-8")
    manifest = {
        "format_version": 1, "status": "completed", "data_kind": "synthetic",
        "backend": "synthetic_fixture", "news_count": 3, "encoded_news": 3,
        "embedding_policy": {k: CONTENT_POLICY[k] for k in CACHE_POLICY_KEYS},
        "content_sha256": content_sha256(metadata),
        "files": {name: file_sha256(cache / name) for name in ("embeddings.npy", "news_ids.json")},
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return train, dev, news, cache


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_entire_pipeline_twice_including_external_cli_and_handoff(tmp_path):
    train, dev, news, cache = synthetic_inputs(tmp_path / "inputs")
    signatures = []
    for repeat in (1, 2):
        root = tmp_path / f"repeat{repeat}"
        root.mkdir()
        common = dict(train_path=train, dev_path=dev, news_paths=[news], data_kind="synthetic")
        baseline = run_baselines(**common, output_dir=root / "base",
                                 predictions_dir=root / "base_pred")
        content = run_baselines(**common, output_dir=root / "content",
                                predictions_dir=root / "content_pred",
                                content_config_path="configs/content_baselines_config.yaml", embedding_cache=cache)
        e1 = run_e1(**common, reference_dir=root / "base", output_dir=root / "e1",
                    predictions_dir=root / "e1_pred", splits_dir=root / "splits")
        assert all(m["status"] == "completed" for m in (baseline, content, e1))
        provenance = root / "model.json"
        provenance.write_text(json.dumps({
            "model": "B2_cumulative", "seed": 0, "training_data": "synthetic train+validation",
            "model_config": {"model": "B2_cumulative"}, "code_revision": "synthetic_fixture",
        }), encoding="utf-8")
        command = [
            sys.executable, "-m", "scripts.evaluate_predictions",
            "--predictions", str(root / "base_pred/test/B2_cumulative/per_impression.csv"),
            "--model", "B2_cumulative", "--seed", "0", "--model-provenance", str(provenance),
            "--split", "test", "--train-behaviors", str(train), "--dev-behaviors", str(dev),
            "--news", str(news), "--data-kind", "synthetic",
            "--output-dir", str(root / "external"), "--predictions-dir", str(root / "external_pred"),
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        for filename in ("overall.csv", "groups.csv"):
            assert read_csv(root / "external" / filename) == read_csv(root / "base/test/B2_cumulative" / filename)
        process = subprocess.run([
            sys.executable, "-m", "scripts.build_handoff", "--baseline-dirs",
            str(root / "base"), str(root / "content"), "--e1-dir", str(root / "e1"),
            "--output-dir", str(root / "handoff"),
        ], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert len(read_csv(root / "handoff/metrics.csv")) == 64
        assert all(r["data_kind"] == "synthetic" for r in read_csv(root / "handoff/metrics.csv"))
        assert (root / "handoff/environment.freeze.txt").stat().st_size > 0
        # Entire score/label/group arrays, metric files and ordered splits must repeat exactly.
        files = list(root.glob("*_pred/**/per_impression.csv")) + list(root.glob("splits/**/membership.csv"))
        files += [root / p / "metrics.csv" for p in ("base", "content", "e1")]
        assert len(list(root.glob("*_pred/**/per_impression.csv"))) == 17  # 6 + 4 + 6 + external
        signatures.append({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    assert signatures[0] == signatures[1]

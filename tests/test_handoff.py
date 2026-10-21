import csv
import json
from pathlib import Path

import pytest

from src.evaluation.handoff import BASELINE_MODELS, E1_PROTOCOLS, build_handoff, collect_results, fingerprint
from src.evaluation.metrics import METRIC_NAMES


@pytest.fixture
def sources(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("synthetic source", encoding="utf-8")
    baseline, e1 = tmp_path / "baseline", tmp_path / "e1"
    for directory, experiment in ((baseline, "baseline"), (e1, "E1")):
        directory.mkdir()
        runs, metrics = [], []
        combos = [(model, "gts_clean") for model in BASELINE_MODELS] if experiment == "baseline" else [
            ("B2_cumulative", protocol) for protocol in E1_PROTOCOLS]
        for model, protocol in combos:
            for split in ("validation", "test"):
                overall = [{"metric": m, "mean": 0.5, "n_impressions": 2, "n_valid": 2, "n_undefined": 0}
                           for m in METRIC_NAMES]
                runs.append({"model": model, "protocol": protocol, "split": split,
                             "overall": overall, "libraries": {}})
                metrics.extend({"model": model, "protocol": protocol, "split": split, "seed": 0, **r}
                               for r in overall)
        manifest = {
            "status": "completed", "data_kind": "synthetic", "seed": 0, "evaluation_config": {},
            "baseline_config": {}, "inputs": [fingerprint(raw)], "runs": runs, "splits": {},
            "git_base_revision": "fixture", "code_sha256": {}, "python": "fixture",
            "phuc02_comparison": {"passed": True},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    return baseline, e1


def test_handoff_grid_and_source_identity(sources, tmp_path):
    baseline, e1 = sources
    rows, provenance = collect_results([baseline], e1)
    assert len(rows) == 64
    assert len(provenance) == 2
    assert sum(r["intentional_leakage"] for r in rows) == 16
    assert all(r["data_kind"] == "synthetic" for r in rows)
    manifest = build_handoff(baseline_dirs=[baseline], e1_dir=e1, output_dir=tmp_path / "handoff")
    assert manifest["rows"] == 64
    with pytest.raises(FileExistsError):
        build_handoff(baseline_dirs=[baseline], e1_dir=e1, output_dir=tmp_path / "handoff")


@pytest.mark.parametrize("change", ["mean", "missing", "duplicate", "undefined", "nan"])
def test_bad_metric_artifacts_rejected(sources, change):
    baseline, e1 = sources
    path = baseline / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "undefined":
        rows[0]["n_undefined"] = "1"
    else:
        rows[0]["mean"] = "nan" if change == "nan" else "0.123"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        collect_results([baseline], e1)


@pytest.mark.parametrize("change", ["failed", "seed", "input", "evaluator", "reference", "duplicate_source"])
def test_incompatible_sources_rejected(sources, change):
    baseline, e1 = sources
    if change == "duplicate_source":
        with pytest.raises(ValueError, match="64"):
            collect_results([baseline, baseline], e1)
        return
    path = e1 / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    if change == "failed":
        m["status"] = "failed"
    elif change == "seed":
        m["seed"] = 1
    elif change == "input":
        Path(m["inputs"][0]["path"]).write_text("modified", encoding="utf-8")
    elif change == "evaluator":
        m["evaluation_config"] = {"different": True}
    else:
        m["phuc02_comparison"]["passed"] = False
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):
        collect_results([baseline], e1)

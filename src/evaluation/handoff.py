"""Assemble existing results with provenance, without rerunning or pooling models."""

import csv
import hashlib
from importlib.metadata import distributions
import json
import math
import platform
from pathlib import Path

from src.evaluation.metrics import METRIC_NAMES

BASELINE_MODELS = ("B1_random", "B2_cumulative", "B2_24h", "B3_category", "B4_semantic_plm")
E1_PROTOCOLS = ("gts_clean", "random", "gts_static")


def fingerprint(path: str | Path) -> dict:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"path": path.as_posix(), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def _metric(row: dict) -> dict:
    mean = None if row["mean"] in (None, "") else float(row["mean"])
    if mean is not None and (not math.isfinite(mean) or not 0 <= mean <= 1):
        raise ValueError("invalid metric mean in source artifact")
    result = {"metric": row["metric"], "mean": mean,
              **{k: int(row[k]) for k in ("n_impressions", "n_valid", "n_undefined")}}
    if (result["metric"] not in METRIC_NAMES or result["n_impressions"] < 1
            or min(result["n_valid"], result["n_undefined"]) < 0
            or result["n_valid"] + result["n_undefined"] != result["n_impressions"]
            or (mean is None) != (result["n_valid"] == 0)):
        raise ValueError("invalid metric name/denominators")
    return result


def collect_results(baseline_dirs: list[str | Path], e1_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Require the complete 64-row approved seed-0 baseline/E1 grid.

    Historical code hashes are retained as historical identities; they need not
    equal current code after later approved tickets extended a shared runner.
    """
    output, sources = [], []
    common_inputs, common_eval, common_kind = None, None, None
    for directory, experiment in [*((Path(p), "baseline") for p in baseline_dirs), (Path(e1_dir), "E1")]:
        manifest_path, metric_path = directory / "manifest.json", directory / "metrics.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "completed" or manifest["seed"] != 0:
            raise ValueError("handoff requires completed approved seed-0 source runs")
        identity = [item["sha256"] for item in manifest["inputs"]]
        if common_inputs is None:
            common_inputs, common_eval, common_kind = identity, manifest["evaluation_config"], manifest["data_kind"]
        if (identity != common_inputs or manifest["evaluation_config"] != common_eval
                or manifest["data_kind"] != common_kind):
            raise ValueError("source runs disagree on input data/evaluator/data kind")
        for item in manifest["inputs"]:
            if fingerprint(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("source input checksum mismatch")
        cache = manifest.get("embedding_cache")
        if cache:
            if fingerprint(Path(cache["path"]) / "manifest.json")["sha256"] != cache["manifest_sha256"]:
                raise ValueError("PLM cache manifest checksum mismatch")
            for name, digest in cache["files"].items():
                if fingerprint(Path(cache["path"]) / name)["sha256"] != digest:
                    raise ValueError("PLM cache file checksum mismatch")
        expected = {}
        for run in manifest["runs"]:
            protocol = run["protocol"] if experiment == "E1" else "gts_clean"
            for metric in run["overall"]:
                key = protocol, run["split"], run["model"], metric["metric"]
                if key in expected:
                    raise ValueError("duplicate metric in source manifest")
                expected[key] = _metric(metric)
        seen = set()
        with metric_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                protocol = row["protocol"] if experiment == "E1" else "gts_clean"
                key = protocol, row["split"], row["model"], row["metric"]
                parsed = _metric(row)
                if key in seen or expected.get(key) != parsed or int(row["seed"]) != 0:
                    raise ValueError("metrics.csv disagrees with source manifest or contains duplicates")
                seen.add(key)
                output.append({
                    "experiment": experiment, "protocol": protocol, "model": row["model"],
                    "split": row["split"], "seed": 0, "data_kind": common_kind,
                    "intentional_leakage": experiment == "E1" and protocol != "gts_clean",
                    **parsed, "source_metrics": metric_path.as_posix(),
                    "source_manifest": manifest_path.as_posix(),
                    "source_metrics_sha256": fingerprint(metric_path)["sha256"],
                })
        if seen != set(expected):
            raise ValueError("missing metrics from source CSV")
        if experiment == "E1":
            if not manifest["phuc02_comparison"]["passed"]:
                raise ValueError("E1 clean reference comparison did not pass")
            for spec in manifest["splits"].values():
                item = spec["membership"]
                if fingerprint(item["path"])["sha256"] != item["sha256"]:
                    raise ValueError("E1 split membership checksum mismatch")
        sources.append({
            "experiment": experiment, "manifest": fingerprint(manifest_path),
            "metrics": fingerprint(metric_path), "data_kind": manifest["data_kind"],
            "inputs": manifest["inputs"], "seed": manifest["seed"],
            "evaluation_config": manifest["evaluation_config"],
            "resolved_model_config": manifest.get("content_config") or manifest.get("resolved_config")
                                     or manifest["baseline_config"],
            "splits": manifest["splits"], "embedding_cache": cache,
            "git_base_revision": manifest["git_base_revision"], "historical_code_sha256": manifest["code_sha256"],
            "python": manifest["python"],
            "libraries": manifest["runs"][0]["libraries"],
        })
    expected_grid = {
        ("baseline", "gts_clean", model, split, metric)
        for model in BASELINE_MODELS for split in ("validation", "test") for metric in METRIC_NAMES
    } | {
        ("E1", protocol, "B2_cumulative", split, metric)
        for protocol in E1_PROTOCOLS for split in ("validation", "test") for metric in METRIC_NAMES
    }
    keys = [(r["experiment"], r["protocol"], r["model"], r["split"], r["metric"]) for r in output]
    if len(keys) != len(set(keys)) or set(keys) != expected_grid:
        raise ValueError("handoff requires exactly 64 unique baseline/E1 metric rows")
    return sorted(output, key=lambda r: (r["experiment"], r["protocol"], r["model"], r["split"], r["metric"])), sources


def environment_snapshot() -> dict:
    packages = sorted({(dist.metadata["Name"], dist.version) for dist in distributions()
                       if dist.metadata.get("Name")})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(packages),
            "note": "Observed environment, not a portable lockfile or an installation instruction."}


def build_handoff(*, baseline_dirs: list[str | Path], e1_dir: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("use a new handoff directory")
    metrics, sources = collect_results(baseline_dirs, e1_dir)
    env = environment_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    table = ["# B1–B4 and E1 results", "", "Seed 0; existing measurements, no pooling across protocols.",
             "", "| Experiment | Protocol | Model | Split | AUC | MRR | nDCG@5 | nDCG@10 |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    grouped = {}
    for row in metrics:
        grouped.setdefault(tuple(row[k] for k in ("experiment", "protocol", "model", "split")), {})[
            row["metric"]] = row["mean"]
    for key, values in grouped.items():
        values = ["null" if values[m] is None else f"{values[m]:.6f}" for m in METRIC_NAMES]
        table.append("| " + " | ".join([*key, *values]) + " |")
    table.extend(["", "E1 Random uses a different evaluation population. Static intentionally includes evaluation labels.",
                  "B3/B4 use current-impression History; individual click times are unavailable.",
                  "These are single-seed results, not significance estimates.", ""])
    (output / "results.md").write_text("\n".join(table), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (output / "environment.freeze.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in env["packages"].items()) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed", "ticket": "PHUC-05", "data_kind": metrics[0]["data_kind"], "seed": 0,
        "rows": len(metrics), "source_runs": sources,
        "interpretation": "Historical metric/source integrity checked; no new training or test-set tuning.",
        "artifacts": [fingerprint(output / name) for name in
                      ("metrics.csv", "results.md", "environment.json", "environment.freeze.txt")],
        "code": [fingerprint(Path(__file__)),
                 fingerprint(Path(__file__).resolve().parents[2] / "scripts/build_handoff.py")],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest

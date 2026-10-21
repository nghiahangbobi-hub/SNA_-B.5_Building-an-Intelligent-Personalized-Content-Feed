"""PHUC-04: reproducible splitting, explicit leakage, and PHUC-02 equivalence."""

from collections import Counter
import csv
from dataclasses import replace
from datetime import datetime, timedelta
import json
import random

import pytest
import yaml

from scripts.run_baselines import run_baselines
from scripts.run_e1_leakage import POLICY, load_config, prediction_rows, run_e1, source_audit
from src.baselines.b2_popularity import ClickIndex, PopularityBaseline
from src.baselines.e1_popularity import CLICK_TIME_POLICY, E1StaticPopularity
from src.data_loader.mind_reader import Behavior, read_behaviors
from src.data_loader.random_split import random_impression_split
from src.data_loader.temporal_split import TRAIN_START, TEST_START, day_cutoff
from src.evaluation.evaluator import evaluate
from src.evaluation.leakage_comparison import compare_phuc02, compare_protocols, metric_delta
from src.evaluation.metrics import METRIC_NAMES


def behavior(i, *, day=13, labels=(1, 0), split="train"):
    return Behavior("train", str(i), "U1", datetime(2019, 11, day, 12), 0,
                    ("A", "B"), labels, split, history_ids=())


def membership(partitions):
    return {split: [r.key for r in getattr(partitions, split)] for split in ("train", "validation", "test")}


@pytest.mark.parametrize("seed", [0, 7, 999])
def test_random_split_independent_of_read_order(seed):
    rows = [behavior(i, day=9 + i % 7) for i in range(103)]
    rows += [replace(rows[0], source="dev")]  # Same ID in another source is a separate impression.
    expected = random_impression_split(rows, seed=seed)
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    for reordered in (reversed(rows), shuffled, rows[::2] + rows[1::2]):
        assert membership(random_impression_split(reordered, seed=seed)) == membership(expected)
    assert [len(getattr(expected, s)) for s in ("train", "validation", "test")] == [83, 10, 11]
    originals = {r.key: r for r in rows}
    assigned = [r for split in ("train", "validation", "test") for r in getattr(expected, split)]
    assert len({r.key for r in assigned}) == len(rows)
    assert {r.key for r in assigned} == set(originals)
    for row in assigned:
        assert row.candidate_ids == originals[row.key].candidate_ids
        assert row.labels == originals[row.key].labels
    assert all(r.split == "train" for r in rows)  # Source objects were not mutated.


def test_split_uses_only_seed_and_impression_key():
    rows = [behavior(i) for i in range(100)]
    changed = [replace(r, labels=(0, 1), candidate_ids=("B", "A"), history_length=30,
                       history_ids=("X",) * 30, timestamp=TRAIN_START) for r in rows]
    assert membership(random_impression_split(rows, seed=0)) == membership(random_impression_split(changed, seed=0))
    assert membership(random_impression_split(rows, seed=0)) != membership(random_impression_split(rows, seed=1))


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_invalid_random_seed(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        random_impression_split([behavior(i) for i in range(20)], seed=seed)


def test_random_rejects_duplicate_small_and_out_of_period():
    rows = [behavior(i) for i in range(20)]
    for invalid, message in ((rows + rows[:1], "duplicate"), (rows[:3], "nonempty"),
                             ([replace(rows[0], timestamp=datetime(2019, 11, 16))] + rows[1:], "timestamps")):
        with pytest.raises(ValueError, match=message):
            random_impression_split(invalid, seed=0)


def test_clean_boundaries_and_intentionally_leaky_counts():
    old = replace(behavior(1), timestamp=TEST_START - timedelta(microseconds=1),
                  history_length=3, history_ids=("ghost",) * 3)
    current = replace(behavior(2, labels=(0, 1), split="test"), timestamp=TEST_START)
    future = behavior(3, day=15, labels=(0, 1), split="test")
    clean = PopularityBaseline(ClickIndex([old], source_end=TEST_START))
    static = E1StaticPopularity([old, current, future], protocol="gts_static")
    assert clean.score(current.context()) == (1, 0)
    assert static.score(current.context()) == (1, 2)
    assert static.counts["ghost"] == 0
    with pytest.raises(ValueError, match="strictly precede"):
        ClickIndex([replace(current, split="train")], source_end=TEST_START)
    audit = source_audit([old, current, future], [current])
    assert audit["source_evaluation_overlap_impressions"] == 1
    assert audit["event_evaluation_pairs_at_or_after_day_cutoff"] == 2
    assert audit["event_evaluation_pairs_after_impression_time"] == 1
    assert audit["timestamp_policy"] == CLICK_TIME_POLICY


def test_random_source_can_use_future_but_not_evaluation_labels():
    train = behavior(1, day=15, labels=(0, 1))
    evaluation = behavior(2, day=9, labels=(1, 0), split="test")
    model = E1StaticPopularity([train], protocol="random")
    assert model.score(evaluation.context()) == (0, 1)
    assert model.score(replace(evaluation, labels=(0, 1)).context()) == (0, 1)
    with pytest.raises(ValueError, match="train/train_final"):
        E1StaticPopularity([evaluation], protocol="random")
    with pytest.raises(ValueError, match="explicitly leaky"):
        E1StaticPopularity([train], protocol="gts_clean")


@pytest.mark.parametrize("labels", [(1,), (1, None), (True, 0), (2, 0)])
def test_invalid_e1_source_labels(labels):
    with pytest.raises(ValueError):
        E1StaticPopularity([behavior(1, labels=labels)], protocol="gts_static")


def test_duplicate_e1_source_rejected():
    row = behavior(1)
    with pytest.raises(ValueError, match="duplicate"):
        E1StaticPopularity([row, row], protocol="gts_static")


def test_evaluator_edge_policies_and_absent_leaky_prior_metadata():
    # Two unseen candidates tie at zero. MRR averages both positives' reciprocal ranks.
    row = replace(behavior(1, split="test"), candidate_ids=("X", "Y", "Z"), labels=(1, 0, 1))
    model = E1StaticPopularity([behavior(2)], protocol="gts_static")
    predictions = list(prediction_rows([row], model, {}))
    assert predictions[0].metadata_cutoff is None
    assert all(c.prior_click_count is None for c in predictions[0].candidates)
    result = evaluate(predictions)
    assert result.metadata["history_thresholds"] is None
    record = result.per_impression[0]
    assert record["candidate_ids"] == ["X", "Y", "Z"]
    assert record["auc"] == 0.5
    assert record["mrr"] == pytest.approx(2 / 3)
    assert record["groups"]["cold_warm"] == ("unknown",)
    assert record["groups"]["history_length"] == ("empty",)
    assert record["groups"]["category_presence"] == ("unknown",)
    no_positive = evaluate(prediction_rows([replace(row, labels=(0, 0, 0))], model, {}))
    assert all(r["mean"] is None and r["n_undefined"] == 1 for r in no_positive.overall)


@pytest.mark.parametrize(("baseline", "value", "delta", "percent", "reason"), [
    (0.5, 0.6, 0.1, 20, None), (0.5, 0.4, -0.1, -20, None),
    (0, 0.2, 0.2, None, "zero_baseline"), (0, 0, 0, None, "zero_baseline"),
    (None, 0.2, None, None, "undefined_metric"), (0.2, None, None, None, "undefined_metric"),
])
def test_delta_policies(baseline, value, delta, percent, reason):
    result = metric_delta(baseline, value)
    assert result["absolute_delta"] == (pytest.approx(delta) if delta is not None else None)
    assert result["relative_delta_pct"] == (pytest.approx(percent) if percent is not None else None)
    assert result["reason"] == reason


def test_comparison_requires_complete_unique_protocol_grid():
    with pytest.raises(ValueError, match="exactly four"):
        compare_protocols([])


@pytest.mark.parametrize("key,value", [
    ("seed", 1), ("seed", False), ("seed", 0.0), ("history_in_popularity", True),
    ("click_timestamp", "actual_click"), ("random_test_source", "all"),
])
def test_unapproved_config_rejected(tmp_path, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**POLICY, key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        load_config(path)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def fixture_run(tmp_path):
    train, dev, news = (tmp_path / name for name in ("train.tsv", "dev.tsv", "news.tsv"))
    sources = {"train": [], "dev": []}
    for i in range(28):
        day = 9 + i % 7
        labels = "A-1 B-0 C-0" if i % 3 else "A-0 B-1 C-1"
        sources["dev" if day == 15 else "train"].append(
            f"{i}\tU{i % 4}\t11/{day}/2019 12:00:00 PM\t{'ghost' if i % 2 else ''}\t{labels}\n")
    train.write_text("".join(sources["train"]), encoding="utf-8")
    dev.write_text("".join(sources["dev"]), encoding="utf-8")
    news.write_text("".join(f"{nid}\tsports\tsub\ttitle\tabstract\turl\t[]\t[]\n"
                            for nid in ("A", "B", "C")), encoding="utf-8")
    reference = tmp_path / "reference"
    run_baselines(train_path=train, dev_path=dev, news_paths=[news], output_dir=reference,
                  predictions_dir=tmp_path / "reference_predictions", data_kind="synthetic")
    return {
        "train_path": train, "dev_path": dev, "news_paths": [news], "reference_dir": reference,
        "output_dir": tmp_path / "e1", "predictions_dir": tmp_path / "predictions",
        "splits_dir": tmp_path / "splits", "data_kind": "synthetic",
    }


def test_end_to_end_scores_sources_splits_and_phuc02_equivalence(fixture_run):
    manifest = run_e1(**fixture_run)
    assert manifest["status"] == "completed"
    assert manifest["phuc02_comparison"]["passed"]
    assert manifest["phuc02_comparison"]["all_exact_equal"]
    assert len(manifest["runs"]) == 6
    assert len(read_csv(fixture_run["output_dir"] / "metrics.csv")) == 24
    deltas = read_csv(fixture_run["output_dir"] / "inflation.csv")
    assert len(deltas) == 16
    assert all(r["same_evaluation_set"] == ("True" if r["protocol"] == "gts_static" else "False")
               for r in deltas)
    raw = list(read_behaviors(fixture_run["train_path"], source="MINDsmall_train"))
    raw += list(read_behaviors(fixture_run["dev_path"], source="MINDsmall_dev"))
    by_key = {r.key: r for r in raw}
    splits = {p: read_csv(fixture_run["splits_dir"] / p / "membership.csv")
              for p in ("gts_clean", "random", "gts_static")}
    assert splits["gts_clean"] == splits["gts_static"]
    for protocol, members in splits.items():
        assert {(r["source"], r["impression_id"]) for r in members} == set(by_key)
        assert len(members) == len(raw)
        for split in ("validation", "test"):
            permitted = {"train"} if split == "validation" else {"train", "validation"}
            source = raw if protocol == "gts_static" else [
                by_key[r["source"], r["impression_id"]] for r in members if r["split"] in permitted]
            exported = read_csv(fixture_run["predictions_dir"] / protocol / split / "per_impression.csv")
            for row in exported:
                original = by_key[row["source"], row["impression_id"]]
                allowed = [s for s in source if protocol != "gts_clean"
                           or s.timestamp < day_cutoff(original.timestamp)]
                counts = Counter(n for s in allowed for n, label in zip(s.candidate_ids, s.labels) if label)
                assert json.loads(row["candidate_ids"]) == list(original.candidate_ids)
                assert json.loads(row["labels"]) == list(original.labels)
                assert json.loads(row["scores"]) == [counts[n] for n in original.candidate_ids]
                assert json.loads(row["prior_click_counts"]) == [None] * len(original.labels)
                assert row["metadata_cutoff"] == ""
    clean_audits = [r["source_audit"] for r in manifest["runs"] if r["protocol"] == "gts_clean"]
    assert all(a["event_evaluation_pairs_at_or_after_day_cutoff"] == 0 for a in clean_audits)
    with pytest.raises(FileExistsError, match="new E1"):
        run_e1(**fixture_run)


def test_phuc02_mismatch_blocks_completion_and_records_reason(fixture_run):
    path = fixture_run["reference_dir"] / "metrics.csv"
    rows = read_csv(path)
    for row in rows:
        if row["model"] == "B2_cumulative":
            row["mean"] = "0.123456789"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="differs from PHUC-02"):
        run_e1(**fixture_run)
    manifest = json.loads((fixture_run["output_dir"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["phuc02_comparison"]["passed"] is False
    assert {r["protocol"] for r in manifest["runs"]} == {"gts_clean"}
    assert "PHUC-02" in manifest["error"]
    assert not (fixture_run["output_dir"] / "inflation.csv").exists()


def test_phuc02_input_mismatch_rejected_before_run(fixture_run):
    with fixture_run["train_path"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="provenance mismatch"):
        run_e1(**fixture_run)
    assert not fixture_run["output_dir"].exists()


def test_phuc02_comparison_checks_counts_and_missing_metrics():
    actual = [{"protocol": "gts_clean", "split": s, "metric": m, "mean": 0.5,
               "n_impressions": 3, "n_valid": 3, "n_undefined": 0}
              for s in ("validation", "test") for m in METRIC_NAMES]
    reference = [{**r, "model": "B2_cumulative", "seed": 0} for r in actual]
    reference[0] = {**reference[0], "n_valid": 2, "n_undefined": 1}
    result = compare_phuc02(actual, reference, seed=0)
    assert sum(not r["passed"] for r in result) == 1
    with pytest.raises(ValueError, match="eight"):
        compare_phuc02(actual, reference[:-1], seed=0)

"""PHUC-02: known-event boundaries, label isolation, parsing, and end-to-end GTS."""

import csv
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from scripts.run_baselines import load_config, prediction_rows, run_baselines
from src.baselines.base import ScoringContext
from src.baselines.b1_random import RandomBaseline
from src.baselines.b2_popularity import ClickIndex, PopularityBaseline
from src.data_loader.mind_reader import Behavior, read_behaviors, read_categories
from src.data_loader.temporal_split import (
    TRAIN_START, VALIDATION_START, TEST_START, TEST_END, day_cutoff, global_temporal_split,
)


def behavior(identifier="1", timestamp=TEST_START - timedelta(hours=1), news=("N1",), labels=(1,),
             split="train", history=0, source="train"):
    return Behavior(source, identifier, "U1", timestamp, history, news, labels, split)


@pytest.fixture
def tsv_files(tmp_path):
    train = tmp_path / "train.tsv"
    dev = tmp_path / "dev.tsv"
    news = tmp_path / "news.tsv"
    train.write_text(
        "1\tU1\t11/09/2019 11:59:59 PM\tNghost Nghost\tN1-1 N2-0\n"
        "2\tU1\t11/13/2019 12:00:00 AM\t\tN1-1 N2-1 N3-0\n"
        "3\tU2\t11/13/2019 11:59:59 PM\tN9\tN2-1 N3-0\n"
        "4\tU3\t11/14/2019 12:00:00 AM\tN1\tN3-1 N2-0\n"
        "5\tU4\t11/14/2019 12:01:00 PM\tN1 N2\tN2-1 N1-0\n", encoding="utf-8")
    dev.write_text(
        "1\tU1\t11/15/2019 12:01:00 AM\tN1 Nghost\tN1-1 N2-0 N3-0 Nnew-0\n"
        "2\tU3\t11/15/2019 11:59:00 PM\tN2\tNnew-1 N1-0\n", encoding="utf-8")
    news.write_text("".join(f"{nid}\t{cat}\tsub\ttitle\tabstract\turl\t[]\t[]\n"
                            for nid, cat in [("N1", "sports"), ("N2", "finance"),
                                             ("N3", "sports"), ("Nnew", "")]), encoding="utf-8")
    return train, dev, news


def partitions(paths):
    train, dev, _ = paths
    return global_temporal_split(read_behaviors(train, source="MINDsmall_train"),
                                  read_behaviors(dev, source="MINDsmall_dev"))


def test_reader_preserves_order_and_empty_history(tsv_files):
    rows = list(read_behaviors(tsv_files[0], source="train"))
    assert rows[1].candidate_ids == ("N1", "N2", "N3")
    assert rows[1].labels == (1, 1, 0)
    assert rows[1].history_length == 0
    assert rows[1].timestamp == datetime(2019, 11, 13)
    assert rows[0].history_length == 2


@pytest.mark.parametrize("line", [
    "bad\n", "1\tU1\tbad date\t\tN1-1\n", "1\tU1\t11/13/2019 1:00:00 PM\t\tN1-2\n",
    "1\tU1\t11/13/2019 1:00:00 PM\t\tN1-1 N1-0\n", "1\tU1\t11/13/2019 1:00:00 PM\t\t\n",
    "1\tU1\t11/13/2019 1:00:00 PM\t\tN1\n",
])
def test_bad_behavior_fails_with_line_number(tmp_path, line):
    path = tmp_path / "bad.tsv"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.tsv:1:"):
        list(read_behaviors(path, source="train"))


def test_duplicate_input_rows_rejected(tsv_files):
    train = tsv_files[0]
    rows = train.read_text(encoding="utf-8").splitlines()
    train.write_text(rows[0] + "\n" + rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate impression"):
        list(read_behaviors(train, source="train"))


def test_category_merge_missing_and_conflict(tsv_files, tmp_path):
    categories = read_categories([tsv_files[2], tsv_files[2]])
    assert categories["Nnew"] is None
    assert categories["N1"] == "sports"
    conflict = tmp_path / "conflict.tsv"
    conflict.write_text("N1\tfinance\tsub\ttitle\tabstract\turl\t[]\t[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting category"):
        read_categories([tsv_files[2], conflict])


def test_gts_dates_and_final_train(tsv_files):
    p = partitions(tsv_files)
    assert [len(p.train), len(p.validation), len(p.test), len(p.final_train())] == [3, 2, 2, 5]
    assert all(r.timestamp < VALIDATION_START and r.split == "train" for r in p.train)
    assert all(VALIDATION_START <= r.timestamp < TEST_START for r in p.validation)
    assert all(TEST_START <= r.timestamp < TEST_END and r.source == "MINDsmall_dev" for r in p.test)
    assert all(r.split == "train_final" for r in p.final_train())
    assert all(r.split == "validation" for r in p.validation)  # final_train does not mutate


@pytest.mark.parametrize("bad_time", [TRAIN_START - timedelta(seconds=1), TEST_START])
def test_official_train_outside_window_fails(tsv_files, bad_time):
    with pytest.raises(ValueError, match="outside approved train"):
        global_temporal_split([behavior(timestamp=bad_time)],
                               read_behaviors(tsv_files[1], source="dev"))


def test_official_dev_wrong_day_fails(tsv_files):
    with pytest.raises(ValueError, match="outside approved dev"):
        global_temporal_split(read_behaviors(tsv_files[0], source="train"),
                               [behavior(source="dev", timestamp=TEST_END)])


def test_gts_requires_all_partitions(tsv_files):
    with pytest.raises(ValueError, match="nonempty"):
        global_temporal_split([behavior(timestamp=TRAIN_START)],
                               read_behaviors(tsv_files[1], source="dev"))


def test_random_seed_reproducible_order_independent():
    context = ScoringContext("train", "i1", TEST_START, ("N1", "N2", "N3"))
    first = RandomBaseline(0).score(context)
    assert first == RandomBaseline(0).score(context)
    assert first != RandomBaseline(1).score(context)
    assert all(0 <= score < 1 for score in first)
    shuffled = replace(context, candidate_ids=("N3", "N1", "N2"))
    assert RandomBaseline(0).score(shuffled) == (first[2], first[0], first[1])
    assert RandomBaseline(0).score(replace(context, impression_id="other")) != first


def test_random_stable_across_python_processes():
    code = ("from datetime import datetime; from src.baselines.base import ScoringContext; "
            "from src.baselines.b1_random import RandomBaseline; "
            "print(RandomBaseline(0).score(ScoringContext('s','i',datetime(2019,11,15),('N1','N2'))))")
    output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    local = RandomBaseline(0).score(ScoringContext("s", "i", TEST_START, ("N1", "N2")))
    assert output == str(local)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_bad_random_seed(seed):
    with pytest.raises(ValueError):
        RandomBaseline(seed)


def test_window_boundaries_exclude_equal_and_future():
    times = [TEST_START - timedelta(days=1, microseconds=1), TEST_START - timedelta(days=1),
             TEST_START - timedelta(microseconds=1), TEST_START, TEST_START + timedelta(hours=1)]
    index = ClickIndex([behavior(str(i), timestamp=t) for i, t in enumerate(times)], source_end=TEST_END)
    assert index.count("N1", TEST_START) == 3
    assert index.count("N1", TEST_START, window_hours=24) == 2
    assert index.count("unseen", TEST_START) == 0
    assert index.count("unseen", TEST_START, window_hours=24) == 0


def test_within_day_predictions_use_same_snapshot():
    index = ClickIndex([behavior(timestamp=TEST_START + timedelta(hours=1))], source_end=TEST_END)
    baseline = PopularityBaseline(index)
    early = ScoringContext("dev", "i", TEST_START, ("N1",))
    late = replace(early, timestamp=TEST_START + timedelta(hours=23, minutes=59))
    assert baseline.score(early) == baseline.score(late) == (0.0,)
    assert day_cutoff(late.timestamp) == TEST_START


@pytest.mark.parametrize("split", ["validation", "test", "unassigned"])
def test_evaluation_labels_cannot_enter_click_index(split):
    with pytest.raises(ValueError, match="never evaluation labels"):
        ClickIndex([behavior(split=split)], source_end=TEST_START)


def test_source_end_strictly_excludes_current_events():
    with pytest.raises(ValueError, match="strictly precede"):
        ClickIndex([behavior(timestamp=TEST_START)], source_end=TEST_START)


def test_duplicate_sources_fail_but_repeat_user_news_clicks_count():
    row = behavior()
    with pytest.raises(ValueError, match="duplicate"):
        ClickIndex([row, row], source_end=TEST_START)
    index = ClickIndex([row, replace(row, impression_id="2")], source_end=TEST_START)
    assert index.count("N1", TEST_START) == 2


def test_unsorted_source_matches_sorted():
    rows = [behavior("a", TEST_START - timedelta(hours=3)),
            behavior("b", TEST_START - timedelta(days=2))]
    a, b = (ClickIndex(order, source_end=TEST_START) for order in (rows, rows[::-1]))
    for cutoff in (TEST_START, TEST_START - timedelta(days=1)):
        assert a.count("N1", cutoff) == b.count("N1", cutoff)


def test_history_does_not_add_events(tsv_files):
    p = partitions(tsv_files)
    original = ClickIndex(p.train, source_end=VALIDATION_START)
    changed = ClickIndex([replace(r, history_length=99999) for r in p.train], source_end=VALIDATION_START)
    assert original.event_count == changed.event_count == 4
    assert original.count("Nghost", VALIDATION_START) == 0
    assert original.count("N1", VALIDATION_START) == changed.count("N1", VALIDATION_START) == 2


def test_known_validation_and_test_scores(tsv_files):
    p = partitions(tsv_files)
    validation = ClickIndex(p.train, source_end=VALIDATION_START)
    final = ClickIndex(p.final_train(), source_end=TEST_START)
    context = ScoringContext("dev", "1", TEST_START + timedelta(hours=10), ("N1", "N2", "N3", "Nnew"))
    assert PopularityBaseline(final).score(context) == (2.0, 3.0, 1.0, 0.0)
    assert PopularityBaseline(final, window_hours=24).score(context) == (0.0, 1.0, 1.0, 0.0)
    assert validation.count("N3", VALIDATION_START) == 0  # validation click not yet available


def test_evaluation_labels_do_not_change_scores_or_counts(tsv_files):
    p = partitions(tsv_files)
    index = ClickIndex(p.final_train(), source_end=TEST_START)
    categories = read_categories([tsv_files[2]])
    flipped = [replace(r, labels=tuple(1 - y for y in r.labels)) for r in p.test]
    for model in (RandomBaseline(), PopularityBaseline(index), PopularityBaseline(index, window_hours=24)):
        a = list(prediction_rows(p.test, model, index, categories))
        b = list(prediction_rows(flipped, model, index, categories))
        assert [[(c.score, c.prior_click_count) for c in r.candidates] for r in a] == [
            [(c.score, c.prior_click_count) for c in r.candidates] for r in b]


def test_24h_scores_use_cumulative_count_for_cold_metadata(tsv_files):
    p = partitions(tsv_files)
    index = ClickIndex(p.final_train(), source_end=TEST_START)
    prediction = next(prediction_rows(p.test, PopularityBaseline(index, window_hours=24), index, {}))
    n1 = prediction.candidates[0]
    assert n1.news_id == "N1" and n1.score == 0 and n1.prior_click_count == 2
    assert prediction.metadata_cutoff == TEST_START


def test_unseen_ties_keep_original_order(tsv_files):
    p = partitions(tsv_files)
    index = ClickIndex([], source_end=TEST_START)
    prediction = next(prediction_rows(p.test, PopularityBaseline(index), index, {}))
    assert tuple(c.news_id for c in prediction.candidates) == p.test[0].candidate_ids
    assert all(c.score == 0 for c in prediction.candidates)


@pytest.mark.parametrize("hours", [0, 1, -24, True])
def test_unapproved_window_fails(hours):
    with pytest.raises(ValueError):
        PopularityBaseline(ClickIndex([], source_end=TEST_START), window_hours=hours)


def test_config_changes_need_explicit_implementation(tmp_path):
    import yaml
    config = load_config("configs/baselines_config.yaml")
    config["history_in_popularity"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        load_config(path)


def test_end_to_end_synthetic_outputs(tsv_files, tmp_path):
    out, pred = tmp_path / "tables", tmp_path / "predictions"
    manifest = run_baselines(train_path=tsv_files[0], dev_path=tsv_files[1], news_paths=[tsv_files[2]],
                             output_dir=out, predictions_dir=pred, data_kind="synthetic")
    assert manifest["status"] == "completed" and len(manifest["runs"]) == 6
    assert manifest["splits"]["train"]["impressions"] == 3
    assert manifest["splits"]["train_final"]["clicks"] == 6
    assert manifest["splits"]["test"]["impressions"] == 2
    assert all(run["history_thresholds"]["n_train"] == 3 for run in manifest["runs"])
    assert [run["click_source_events"] for run in manifest["runs"]] == [4, 4, 4, 6, 6, 6]
    with (out / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    with (pred / "test/B2_cumulative/per_impression.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["scores"]) == [2.0, 3.0, 1.0, 0.0]
    assert row["metadata_cutoff"] == "2019-11-15T00:00:00"
    assert len(manifest["inputs"]) == 3 and all(len(f["sha256"]) == 64 for f in manifest["inputs"])
    with pytest.raises(FileExistsError):
        run_baselines(train_path=tsv_files[0], dev_path=tsv_files[1], news_paths=[tsv_files[2]],
                      output_dir=out, predictions_dir=pred, data_kind="synthetic")

import csv
from dataclasses import replace
from datetime import datetime
import json

import pytest

from scripts.evaluate_predictions import load_provenance
from src.baselines.b2_popularity import ClickIndex
from src.data_loader.mind_reader import Behavior
from src.data_loader.prediction_reader import external_predictions, read_prediction_scores
from src.evaluation.evaluator import evaluate


def truth():
    return [Behavior("MINDsmall_dev", str(i), "U", datetime(2019, 11, 15), 0,
                     ("A", "B", "C"), (1, 0, 1), "test") for i in (1, 2)]


def write_predictions(path, records):
    fields = ["source", "impression_id", "candidate_ids", "scores"]
    if any("labels" in r for r in records):
        fields.append("labels")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def good_records():
    return [{"source": r.source, "impression_id": r.impression_id,
             "candidate_ids": json.dumps(r.candidate_ids), "scores": "[0,0,0]"} for r in truth()]


def test_scores_row_order_is_irrelevant_and_metadata_is_rebuilt(tmp_path):
    path = tmp_path / "scores.csv"
    write_predictions(path, list(reversed(good_records())))
    scores = read_prediction_scores(path, truth())
    source = [replace(truth()[0], source="train", timestamp=datetime(2019, 11, 14), split="train")]
    index = ClickIndex(source, source_end=datetime(2019, 11, 15))
    rows = list(external_predictions(truth(), scores, index, {"A": "sports", "B": "finance"}))
    assert rows[0].candidates[0].prior_click_count == 1
    assert rows[0].candidates[1].prior_click_count == 0
    assert rows[0].metadata_cutoff == datetime(2019, 11, 15)
    result = evaluate(rows)
    assert result.per_impression[0]["mrr"] == pytest.approx(2 / 3)
    assert result.per_impression[0]["groups"]["cold_warm"] == ("mixed",)
    assert result.per_impression[0]["groups"]["history_length"] == ("empty",)


@pytest.mark.parametrize("field,value", [
    ("source", "wrong"), ("impression_id", "unknown"),
    ("candidate_ids", '["B","A","C"]'), ("candidate_ids", '["A","B"]'),
    ("candidate_ids", '["A","B","C","X"]'), ("candidate_ids", '["A","A","C"]'),
    ("scores", "[1,2]"), ("scores", "[NaN,0,0]"), ("scores", "[Infinity,0,0]"),
    ("scores", "[1e999,0,0]"), ("scores", "[true,0,0]"),
    ("scores", '["1",0,0]'), ("scores", "null"), ("scores", "[1,"),
])
def test_invalid_scores_rejected(tmp_path, field, value):
    rows = good_records()
    rows[0][field] = value
    path = tmp_path / "bad.csv"
    write_predictions(path, rows)
    with pytest.raises(ValueError, match=r"bad.csv:2:"):
        read_prediction_scores(path, truth())


@pytest.mark.parametrize("which", ["missing", "duplicate", "extra", "wrong_labels", "bool_labels"])
def test_coverage_and_labels(tmp_path, which):
    records = good_records()
    if which == "missing":
        records.pop()
    elif which == "duplicate":
        records[1] = records[0].copy()
    elif which == "extra":
        records.append({**records[0], "impression_id": "3"})
    else:
        for r in records:
            r["labels"] = "[1,0,1]"
        records[0]["labels"] = "[0,1,0]" if which == "wrong_labels" else "[true,0,1]"
    path = tmp_path / "bad.csv"
    write_predictions(path, records)
    with pytest.raises(ValueError):
        read_prediction_scores(path, truth())


@pytest.mark.parametrize("text", [
    "source,impression_id,candidate_ids\n",
    "source,impression_id,candidate_ids,scores,scores\n",
    "source,impression_id,candidate_ids,scores\nMINDsmall_dev,1\n",
    'source,impression_id,candidate_ids,scores\nMINDsmall_dev,1,"[]","[]",extra\n',
])
def test_bad_csv_shape(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_prediction_scores(path, truth())


@pytest.mark.parametrize("change", [{"seed": True}, {"seed": 1}, {"model": "B6"}, {"model_config": []},
                                   {"training_data": ""}, {"code_revision": ""}])
def test_provenance_is_required_and_consistent(tmp_path, change):
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"model": "B5", "seed": 0, "model_config": {},
                               "training_data": "train", "code_revision": "abc", **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_provenance(path, model="B5", seed=0)

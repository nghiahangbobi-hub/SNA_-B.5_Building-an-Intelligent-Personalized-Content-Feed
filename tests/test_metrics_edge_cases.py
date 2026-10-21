"""Deterministic PHUC-01 acceptance checks; collectable by pytest or unittest.

Synthetic fixtures only. Optional dependency checks are explicitly skipped when
the approved third-party packages are not installed, never counted as passing.
"""

import csv
import importlib.util
import json
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import log2
from pathlib import Path

from src.data_loader.schema import CandidatePrediction, ImpressionPrediction
from src.evaluation.evaluator import DEFAULT_CONFIG, evaluate, load_eval_config, write_tables
from src.evaluation.group_analyzer import HistoryThresholds, cohort_memberships
from src.evaluation.metrics import auc_score, compute_metrics, mrr_score, ndcg_score

TIME = datetime(2019, 11, 15, 12)


def impression(
    identifier="i1", labels=(1, 0, 1), scores=(0.9, 0.5, 0.1),
    categories=("sports", "finance", "sports"), counts=(0, 2, 0), history=4,
    split="test", timestamp=TIME,
):
    candidates = tuple(CandidatePrediction(f"N{i}", label, score, categories[i], counts[i])
                       for i, (label, score) in enumerate(zip(labels, scores)))
    return ImpressionPrediction(identifier, "U1", timestamp, candidates, split, "synthetic",
                                history, timestamp - timedelta(hours=12), "synthetic_counts")


def training_rows(lengths=(0, 1, 3, 5, 7, None)):
    return [impression(f"train-{i}", history=length, split="train",
                       timestamp=TIME - timedelta(days=2)) for i, length in enumerate(lengths)]


def synthetic_report():
    """Reusable evidence fixture, not a baseline or MIND experiment."""
    rows = [impression("mixed"),
            impression("cold", labels=(0, 0, 0), counts=(0, 0, 0), history=0),
            impression("warm", labels=(1, 1, 1), counts=(1, 2, 3), history=10),
            impression("unknown", counts=(None, None, None),
                       categories=(None, "finance", "finance"), history=None)]
    return evaluate(rows, train_impressions=training_rows(),
                    run_metadata={"data_kind": "synthetic", "fixture_version": "phuc01-v1",
                                  "seed": "0 (randomized oracle checks only)",
                                  "purpose": "acceptance fixture; not MIND results"})


class MetricTests(unittest.TestCase):
    def test_hand_calculated_multiple_positives(self):
        metrics = compute_metrics([1, 0, 1], [0.9, 0.5, 0.1])
        self.assertEqual(metrics["auc"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 2 / 3)
        expected = 1.5 / (1 + 1 / log2(3))
        self.assertAlmostEqual(metrics["ndcg@5"], expected)
        self.assertAlmostEqual(metrics["ndcg@10"], expected)

    def test_perfect_and_reversed_ranking(self):
        self.assertEqual(compute_metrics([1, 0], [2, 1]),
                         {"auc": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0})
        result = compute_metrics([1, 0], [1, 2])
        self.assertEqual(result["auc"], 0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["ndcg@5"], 1 / log2(3))

    def test_cutoffs_five_and_ten(self):
        labels = [0] * 5 + [1] + [0] * 4 + [1]
        scores = list(range(11, 0, -1))
        self.assertEqual(ndcg_score(labels, scores, 5), 0)
        self.assertAlmostEqual(ndcg_score(labels, scores, 10),
                               (1 / log2(7)) / (1 + 1 / log2(3)))

    def test_stable_ties_do_not_use_labels(self):
        self.assertEqual(auc_score([0, 1], [0, 0]), 0.5)
        self.assertEqual(mrr_score([0, 1], [0, 0]), 0.5)
        self.assertEqual(mrr_score([1, 0], [0, 0]), 1)
        self.assertAlmostEqual(ndcg_score([0, 1], [0, 0]), 1 / log2(3))

    def test_all_negative_is_undefined(self):
        self.assertTrue(all(v is None for v in compute_metrics([0, 0], [1, 0]).values()))

    def test_all_positive(self):
        result = compute_metrics([1, 1], [1, 0])
        self.assertIsNone(result["auc"])
        self.assertEqual(result["mrr"], 0.75)
        self.assertEqual(result["ndcg@5"], 1)

    def test_single_candidate(self):
        result = compute_metrics([1], [0.5])
        self.assertIsNone(result["auc"])
        self.assertEqual(result["mrr"], 1)
        self.assertEqual(result["ndcg@10"], 1)

    def test_invalid_inputs_fail(self):
        cases = [([], []), ([1, 0], [1]), ([2], [0]), ([-1], [0]),
                 ([None], [0]), ([True], [0]), ([1.0], [0]), ([1], [None]),
                 ([1], [float("nan")]), ([1], [float("inf")]), ([1], ["1"])]
        for labels, scores in cases:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                compute_metrics(labels, scores)

    def test_invalid_k(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                ndcg_score([1], [1], k)

    def test_auc_against_independent_pairwise_oracle(self):
        rng = random.Random(0)
        for _ in range(100):
            labels = [1, 0] + [rng.randrange(2) for _ in range(12)]
            scores = [rng.randrange(4) for _ in labels]
            positive = [s for y, s in zip(labels, scores) if y == 1]
            negative = [s for y, s in zip(labels, scores) if y == 0]
            expected = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
            expected /= len(positive) * len(negative)
            self.assertAlmostEqual(auc_score(labels, scores), expected)

    def test_strictly_increasing_score_transform(self):
        labels, scores = [0, 1, 1, 0], [-2, 0, 3, 1]
        self.assertEqual(compute_metrics(labels, scores),
                         compute_metrics(labels, [s * 3 + 7 for s in scores]))

    @unittest.skipUnless(importlib.util.find_spec("numpy") and
                         importlib.util.find_spec("sklearn"), "NumPy/scikit-learn unavailable")
    def test_reference_numpy_and_sklearn(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        labels = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0.2, 0.9, 0.7, 0.3, 0.1, 0.6])
        ranked = labels[np.argsort(scores)[::-1]]  # no ties: reference order is unambiguous
        expected_mrr = np.sum(ranked / np.arange(1, 7)) / np.sum(labels)
        self.assertAlmostEqual(mrr_score(labels, scores), expected_mrr)
        self.assertAlmostEqual(auc_score(labels, scores), roc_auc_score(labels, scores))
        self.assertAlmostEqual(auc_score([1, 0, 1], [0, 0, 1]),
                               roc_auc_score([1, 0, 1], [0, 0, 1]))
        for k in (5, 10):
            actual = np.sum((2 ** ranked[:k] - 1) / np.log2(np.arange(len(ranked[:k])) + 2))
            ideal = np.sort(labels)[::-1][:k]
            best = np.sum((2 ** ideal - 1) / np.log2(np.arange(len(ideal)) + 2))
            self.assertAlmostEqual(ndcg_score(labels, scores, k), actual / best)
        # Model outputs often use NumPy scalars; schema normalizes these for JSON.
        candidate = CandidatePrediction("numpy", np.int64(1), np.float64(0.7),
                                        prior_click_count=np.int64(0))
        row = replace(impression(), candidates=(candidate,), history_length=np.int64(2))
        json.dumps(evaluate([row]).per_impression, allow_nan=False)


class SchemaTests(unittest.TestCase):
    def test_candidates_copied_to_immutable_tuple(self):
        row = impression()
        items = list(row.candidates)
        copied = replace(row, candidates=items)
        items.clear()
        self.assertEqual(len(copied.candidates), 3)

    def test_numeric_fields_normalized_for_artifacts(self):
        candidate = CandidatePrediction("N1", 1, 2, prior_click_count=0)
        self.assertIs(type(candidate.score), float)
        self.assertIs(type(candidate.label), int)
        self.assertIs(type(candidate.prior_click_count), int)

    def test_invalid_candidate_metadata(self):
        for kwargs in ({"news_id": ""}, {"category": " "}, {"prior_click_count": -1},
                       {"prior_click_count": True}, {"score": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(impression().candidates[0], **kwargs)

    def test_empty_or_duplicate_candidates(self):
        for candidates in ((), (impression().candidates[0],) * 2, ("N1",)):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                replace(impression(), candidates=candidates)

    def test_invalid_impression_fields(self):
        for kwargs in ({"impression_id": ""}, {"user_id": ""}, {"source": ""},
                       {"split": "dev"}, {"history_length": -1},
                       {"history_length": True}, {"timestamp": "2019-11-15"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(impression(), **kwargs)

    def test_future_metadata_cutoff_rejected(self):
        with self.assertRaisesRegex(ValueError, "after"):
            replace(impression(), metadata_cutoff=TIME + timedelta(seconds=1))

    def test_cutoff_equal_impression_allowed_for_strict_past_counts(self):
        self.assertEqual(replace(impression(), metadata_cutoff=TIME).metadata_cutoff, TIME)

    def test_counts_require_cutoff_and_provenance(self):
        with self.assertRaises(ValueError):
            replace(impression(), metadata_cutoff=None, metadata_source=None)
        with self.assertRaises(ValueError):
            replace(impression(), metadata_source=None)

    def test_unknown_counts_need_no_cutoff(self):
        row = replace(impression(counts=(None,) * 3), metadata_cutoff=None, metadata_source=None)
        self.assertEqual(cohort_memberships(row, None)["cold_warm"], ("unknown",))

    def test_mixed_naive_aware_rejected(self):
        with self.assertRaisesRegex(ValueError, "time basis"):
            replace(impression(), metadata_cutoff=TIME.replace(tzinfo=timezone.utc))


class GroupTests(unittest.TestCase):
    def test_train_quantiles_exclude_empty_and_missing(self):
        thresholds = HistoryThresholds.fit(training_rows())
        self.assertEqual((thresholds.q25, thresholds.q75), (2.5, 5.5))
        self.assertEqual((thresholds.n_train, thresholds.n_positive_history), (6, 4))
        self.assertEqual([thresholds.group(n) for n in (None, 0, 1, 4, 7)],
                         ["unknown", "empty", "short", "medium", "long"])

    def test_no_positive_train_history(self):
        thresholds = HistoryThresholds.fit(training_rows((0, None)))
        self.assertIsNone(thresholds.q25)
        self.assertEqual(thresholds.group(5), "unknown")
        self.assertEqual(thresholds.group(0), "empty")

    def test_single_or_tied_train_histories(self):
        for lengths in ((3,), (3, 3, 3)):
            thresholds = HistoryThresholds.fit(training_rows(lengths))
            self.assertEqual((thresholds.q25, thresholds.q75), (3, 3))
            self.assertEqual(thresholds.group(3), "short")
            self.assertEqual(thresholds.group(4), "long")

    def test_threshold_boundaries(self):
        thresholds = HistoryThresholds.fit(training_rows((1, 5, 9, 13, 17)))
        self.assertEqual([thresholds.group(n) for n in (5, 6, 13, 14)],
                         ["short", "medium", "medium", "long"])

    def test_no_validation_test_or_train_final_fit(self):
        for split in ("test", "validation", "train_final"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train-only"):
                HistoryThresholds.fit([impression(split=split)])
        with self.assertRaises(ValueError):
            HistoryThresholds.fit([])

    def test_duplicate_train_rows_rejected(self):
        row = training_rows()[0]
        with self.assertRaises(ValueError):
            HistoryThresholds.fit([row, row])

    def test_all_cold_warm_cohorts(self):
        for counts, expected in (((0, 0, 0), "cold_only"), ((1, 2, 3), "warm_only"),
                                 ((0, 1, 0), "mixed"), ((0, None, 1), "unknown")):
            with self.subTest(counts=counts):
                self.assertEqual(cohort_memberships(impression(counts=counts), None)["cold_warm"],
                                 (expected,))

    def test_categories_overlap_deduplicate_and_distinguish_unknown(self):
        groups = cohort_memberships(impression(categories=("sports", "sports", None)), None)
        self.assertEqual(groups["category_presence"], ("category:sports", "unknown"))
        self.assertEqual(cohort_memberships(impression(categories=("unknown", None, "unknown")),
                                           None)["category_presence"],
                         ("category:unknown", "unknown"))


class EvaluatorTests(unittest.TestCase):
    def test_macro_average_not_pooled_auc(self):
        a = impression("a", labels=(1, 0), scores=(0.2, 0.1),
                       counts=(0, 0), categories=("sports", "sports"))
        b = impression("b", labels=(1, 0, 0, 0), scores=(0.5, 0.9, 0.8, 0.7),
                       counts=(0,) * 4, categories=("sports",) * 4)
        result = evaluate([a, b])
        self.assertEqual(result.overall[0]["mean"], 0.5)
        self.assertNotEqual(result.overall[0]["mean"], auc_score([1, 0, 1, 0, 0, 0],
                                                               [0.2, 0.1, 0.5, 0.9, 0.8, 0.7]))

    def test_per_metric_denominators(self):
        result = synthetic_report()
        metrics = {row["metric"]: row for row in result.overall}
        self.assertEqual((metrics["auc"]["n_valid"], metrics["auc"]["n_undefined"]), (2, 2))
        self.assertEqual((metrics["mrr"]["n_valid"], metrics["mrr"]["n_undefined"]), (3, 1))
        self.assertEqual(metrics["auc"]["mean"], 0.5)

    def test_category_group_keeps_complete_ranking(self):
        row = impression()
        result = evaluate([row])
        for group in result.groups:
            if group["dimension"] == "category_presence" and group["metric"] == "mrr":
                self.assertAlmostEqual(group["mean"], 2 / 3)
                self.assertEqual(group["n_impressions"], 1)
        self.assertEqual(result.per_impression[0]["candidate_ids"], ["N0", "N1", "N2"])

    def test_unknown_metadata_preserves_overall(self):
        row = impression(counts=(None,) * 3, categories=(None,) * 3, history=None)
        result = evaluate([row])
        self.assertEqual(result.overall[0]["n_valid"], 1)
        self.assertTrue(all(group["group"] == "unknown" for group in result.groups))

    def test_missing_train_never_fits_evaluation_histories(self):
        result = evaluate([impression(history=1000)])
        self.assertIsNone(result.metadata["history_thresholds"])
        self.assertEqual(result.per_impression[0]["groups"]["history_length"], ("unknown",))

    def test_test_lengths_do_not_change_train_thresholds(self):
        first = evaluate([impression(history=1)], train_impressions=training_rows())
        second = evaluate([impression(history=100000)], train_impressions=training_rows())
        self.assertEqual(first.metadata["history_thresholds"], second.metadata["history_thresholds"])

    def test_future_or_equal_train_time_rejected(self):
        for time in (TIME, TIME + timedelta(seconds=1)):
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "precede"):
                evaluate([impression()], train_impressions=[impression("train", split="train", timestamp=time)])

    def test_train_evaluation_key_overlap_rejected(self):
        row = replace(training_rows()[0], impression_id="i1")
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate([impression()], train_impressions=[row])

    def test_duplicate_ids_empty_and_mixed_split_rejected(self):
        row = impression()
        for rows in ([], [row, row], [row, replace(row, impression_id="v", split="validation")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate(rows)

    def test_same_id_different_sources_allowed(self):
        row = impression()
        self.assertEqual(len(evaluate([row, replace(row, source="other")]).per_impression), 2)

    def test_metadata_and_provenance_retained(self):
        result = synthetic_report()
        self.assertEqual(result.metadata["run"]["data_kind"], "synthetic")
        self.assertEqual(result.metadata["config"], DEFAULT_CONFIG)
        self.assertEqual(len(result.metadata["history_thresholds"]["training_impression_keys"]), 6)
        self.assertEqual(result.per_impression[0]["metadata_source"], "synthetic_counts")
        self.assertEqual(result.per_impression[0]["metadata_cutoff"], "2019-11-15T00:00:00")

    def test_config_drift_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([impression()], config={**DEFAULT_CONFIG, "ranking_ties": "label"})
        with self.assertRaises(ValueError):
            evaluate([impression()], config={})

    def test_tables_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            paths = write_tables(synthetic_report(), Path(root) / "run")
            with paths["report"].open(encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertIsNone(data["per_impression"][1]["mrr"])
            self.assertEqual(data["metadata"]["run"]["data_kind"], "synthetic")
            with paths["overall"].open(encoding="utf-8", newline="") as handle:
                overall = list(csv.DictReader(handle))
            self.assertEqual(len(overall), 4)
            with paths["per_impression"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["mrr"], "")
            self.assertEqual(json.loads(rows[0]["candidate_ids"]), ["N0", "N1", "N2"])
            with self.assertRaises(FileExistsError):
                write_tables(synthetic_report(), Path(root) / "run")

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML unavailable")
    def test_yaml_matches_approved_config(self):
        self.assertEqual(load_eval_config(), DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()

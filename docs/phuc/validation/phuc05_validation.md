# PHUC-05 validation evidence — 2026-10-20

Status: **Verified for the approved seed-0 handoff scope; AC1–AC3 Passed.**
This document summarizes observed checks. The pytest XML and JSON audit files
provide machine-readable evidence; this markdown is not a raw terminal transcript.

## Approved work and unchanged policies

The user approved prediction adapter/CLI, a 64-row result bundle, full regression,
two deterministic synthetic smoke runs, 20 validation + 20 test samples per
baseline with the existing real PLM cache, and full B2 cumulative re-evaluation.
OD-07 was resolved for seed-0 delivery; multi-seed/statistical work remains Future.
The old 19/10/six-hour and 27/10 plan dates remain historical, not silently revised.

No dependency installation, model download, change to PHUC-01–04 baseline/metric/
protocol source/config, commit, push or team message occurred.

External input supplies scores only. Raw MIND supplies labels, candidate order,
user/time/history/category and clean daily-cutoff group metadata. The adapter
does not certify an external model's training pipeline. Positive labels use the
impression timestamp as a proxy; actual click times are unavailable. History is
not used in click counters. Existing metric/null/tie/group policies are unchanged.

## Checks

| Check | Result and evidence |
|---|---|
| Full pytest suite | **Passed: 199 tests + 42 subtests**, 0 failed/skipped, 22.41 s. Includes the prior 156 tests and 43 PHUC-05 tests. |
| Synthetic smoke twice | **Passed**: B1, both B2 variants, B3, synthetic B4 cache, all 3 E1 protocols, external CLI and handoff CLI. Seventeen per-impression score outputs per repetition, ordered splits and metric CSVs are byte-identical across repetitions. |
| Strict input checks | **Passed**: missing/extra/duplicate impressions/candidates, reordered candidates, nonfinite/bool/string/missing scores, malformed CSV/JSON, inconsistent optional labels and invalid provenance are rejected. |
| Synthetic CLI parity | **Passed**: external CLI reproduces original B2 overall and group rows; handoff contains exactly 64 synthetic-labeled rows. |
| Handoff rejection checks | **Passed**: changed/missing/duplicate/invalid metrics, failed/other-seed/incompatible sources, changed raw inputs and failed E1 reference comparisons are rejected. |
| Real score replay | **Passed**: 200 model-impression occurrences, 8,615 candidate scores. All five model configurations match saved scores exactly, including B4; max absolute error 0. Repeated scoring is exactly stable. |
| Real B4 cache | **Passed**: dataset/content, revision/policy, mapping/vector checksums, dimensions, unit norms and backend validated by existing loader. No re-encoding. |
| B2 validation external CLI | **Passed**, exit 0, 30,270 impressions / 1,222,429 candidates. Four overall rows and 88 group rows exactly match PHUC-02, including file hashes. |
| B2 test external CLI | **Passed**, exit 0, 73,152 impressions / 2,740,998 candidates. Four overall rows and 88 group rows exactly match PHUC-02, including file hashes. |
| Group metadata reproduction | **Passed**: Q25=8, Q75=42, fitted from 126,695 train impressions (124,229 with positive history), max train time 13/11/2019 23:59:21. |
| Consolidated bundle | **Passed**, CLI exit 0, 64 rows: 40 baseline + 24 E1. Input/cache/split identity and CSV/manifest consistency checked; output/code hashes match. |
| CLI help | **Passed**, both CLIs exit 0 and document required flags. |
| Previous source/protocol preservation | **Passed**, no Git diff in prior baseline/evaluator/data split sources, configs, baseline/E1/cache runners or requirements. |
| Large-artifact isolation | **Passed**, new predictions and existing embeddings remain Git-ignored. |
| Whitespace and local documentation links | **Passed**, final repository/document checks. |
| Multi-seed, graph training, significance tests, new PLM encoding | **Not Run**, outside approved PHUC-05 scope. Existing real cache/encoder acceptance remains in PHUC-03 evidence. |

No failed or skipped check occurred. The expected-error tests listed above are
passing tests, not failed experimental runs. JUnit reports 241 test-case entries
because it includes 42 subtests; do not present that as 241 independent tests.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --junitxml outputs/logs/phuc05_pytest.xml
.\.venv\Scripts\python.exe -m scripts.evaluate_predictions --help
.\.venv\Scripts\python.exe -m scripts.build_handoff --help
.\.venv\Scripts\python.exe -m scripts.build_handoff --output-dir outputs/tables/phuc05_handoff_seed0_2026-10-20
.\.venv\Scripts\python.exe -m scripts.evaluate_predictions --predictions outputs/predictions/phuc02_mindsmall_seed0_2026-10-19/validation/B2_cumulative/per_impression.csv --model B2_cumulative --seed 0 --model-provenance outputs/logs/phuc05_b2_provenance.json --split validation --output-dir outputs/tables/phuc05_b2_recheck_validation_2026-10-20 --predictions-dir outputs/predictions/phuc05_b2_recheck_validation_2026-10-20
.\.venv\Scripts\python.exe -m scripts.evaluate_predictions --predictions outputs/predictions/phuc02_mindsmall_seed0_2026-10-19/test/B2_cumulative/per_impression.csv --model B2_cumulative --seed 0 --model-provenance outputs/logs/phuc05_b2_provenance.json --split test --output-dir outputs/tables/phuc05_b2_recheck_test_2026-10-20 --predictions-dir outputs/predictions/phuc05_b2_recheck_test_2026-10-20
git diff --check
```

These are observed command destinations. Use fresh output directories to rerun;
the scripts deliberately refuse to overwrite evidence.

The real sample replay and final handoff audit were inline Python commands.
Replay loaded unchanged baseline classes, source TSVs and the validated real cache,
scored the first 20 impressions per split twice, and joined to saved prediction
rows by source+ID. Candidate IDs and labels were checked. B1/B2/B3 required exact
equality; B4 allowed absolute tolerance 1e-6, but actual errors were all zero.
The final audit compared complete overall/group CSV rows and hashes, checked CLI
and bundle code/artifact hashes, and checked the Git diff for protected sources.

## Real replay coverage

| Model | Validation impressions / scores | Test impressions / scores | Max absolute error |
|---|---:|---:|---:|
| B1 Random | 20 / 1,059 | 20 / 664 | 0 |
| B2 cumulative | 20 / 1,059 | 20 / 664 | 0 |
| B2 24h | 20 / 1,059 | 20 / 664 | 0 |
| B3 Category | 20 / 1,059 | 20 / 664 | 0 |
| B4 Semantic PLM | 20 / 1,059 | 20 / 664 | 0 |

This is 40 unique impressions evaluated by five model configurations, not 200
unique raw impressions. The replay does not introduce new benchmark means.

Full B2 cumulative re-evaluation:

| Split | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Validation | 0.5162956361101749 | 0.21501765844013485 | 0.21680688078767524 | 0.2778410639545543 |
| Test | 0.531751528369352 | 0.23820138108047803 | 0.24604210472429233 | 0.3098167064622724 |

All eight means, their denominators and all 176 group rows exactly match PHUC-02.
Both splits have no undefined metrics. This is scoring fixed saved predictions,
not retraining or tuning against test labels.

## Environment and PLM identity

Windows; Python 3.13.5; pytest 9.1.1; PyYAML 6.0.3; NumPy 2.5.4;
scikit-learn 1.9.1. Cache encoder versions: torch 2.14.1,
sentence-transformers 5.7.0, transformers 5.20.1.

Model: `sentence-transformers/all-MiniLM-L6-v2`;
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Cache manifest SHA-256:
`308dbe26502b523a94d27d327aa48f51a1cc300e50283e3819614c081410f1d9`.
The bundle records all installed package versions as observed data, not as a
portable dependency lock. Historical manifests retain original code identities.

## Artifacts and acceptance

- [Handoff guide](../../docs/HANDOFF.md)
- [Readable results](../tables/phuc05_handoff_seed0_2026-10-20/results.md)
- [64-row CSV](../tables/phuc05_handoff_seed0_2026-10-20/metrics.csv)
- [Bundle manifest](../tables/phuc05_handoff_seed0_2026-10-20/manifest.json)
- [Environment snapshot](../tables/phuc05_handoff_seed0_2026-10-20/environment.json)
- [Pytest XML](phuc05_pytest.xml)
- [Real score replay](phuc05_real_score_replay.json)
- [B2/bundle audit](phuc05_handoff_audit.json)
- [B2 producer declaration](phuc05_b2_provenance.json)
- [Validation CLI manifest](../tables/phuc05_b2_recheck_validation_2026-10-20/manifest.json)
- [Test CLI manifest](../tables/phuc05_b2_recheck_test_2026-10-20/manifest.json)

**AC1 Passed:** full regression and two complete smoke repetitions; evidence for
PHUC-01–04 linked from HANDOFF with earlier blocker history preserved.

**AC2 Passed:** observed environment/config/seed/input/split/PLM identity recorded;
synthetic exact repetition, real score reproduction and complete B2 CLI parity
provide evidence beyond stored parameters.

**AC3 Passed:** all 64 source-backed baseline/E1 metric rows, strict external
prediction contract and runnable CLI examples; declared provenance is separated
from checked adapter data. All statistical/History/timestamp/Random population
limits are documented.

No unresolved technical blocker remains for approved PHUC-01–05 seed-0 scope.
Future multi-seed/graph work is not marked complete. User review and any commit/
push remain separate actions requiring explicit instruction.

## Created/modified files

- Added `src/data_loader/prediction_reader.py`, `src/evaluation/handoff.py`,
  `scripts/build_handoff.py`, `tests/test_prediction_reader.py`,
  `tests/test_handoff.py`, `docs/HANDOFF.md`.
- Implemented existing placeholders `scripts/evaluate_predictions.py` and
  `tests/test_smoke_pipeline.py`.
- Updated README and WORK_STATUS; added this validation log, pytest/replay/audit/
  declaration files and three PHUC-05 result directories.
- Raw data, existing model/cache artifacts, requirements and PHUC-01–04 source/
  configs remain unchanged.

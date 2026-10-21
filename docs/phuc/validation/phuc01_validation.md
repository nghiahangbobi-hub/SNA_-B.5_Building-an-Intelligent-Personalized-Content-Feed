# PHUC-01 validation evidence — 2026-10-19

Environment: Windows, Python 3.13.5, repository-local `.venv`.
Scope: synthetic acceptance fixtures only; no MIND or ML experiment was run.
This is a recorded summary of observed command output, not an automatically captured transcript.

## Commands and actual outcomes

| Command / check | Outcome |
|---|---|
| `python -m venv .venv` | Passed, exit 0. |
| `.venv\Scripts\python.exe -m pip install -r requirements.txt` | Failed, exit 1. Network socket access denied: WinError 10013. Dependencies were not installed. |
| `python -m pip cache list` and `.venv\Scripts\python.exe -m pip cache info` | No cached wheels; HTTP cache 0 bytes / 0 files. No offline installation source available. |
| `.venv\Scripts\python.exe -m unittest discover -s tests -p test_metrics_edge_cases.py -v` — first run | Exit 0: 43 discovered, 41 Passed, 0 Failed, 2 Skipped. |
| Same unittest command — after scalar-normalization change | Exit 0: 44 discovered, **42 Passed, 0 Failed, 2 Skipped**, 0 errors. |
| `.venv\Scripts\python.exe -m pytest tests/test_metrics_edge_cases.py -v` | Could not start: `No module named pytest`. Pytest suite execution: **Not Run**. |
| `git diff --check` | Passed, exit 0; Git emitted LF/CRLF normalization warnings only. |

Skipped tests (Not Run, not Passed):

- `MetricTests.test_reference_numpy_and_sklearn`: NumPy/scikit-learn unavailable. Includes reference parity and NumPy-scalar export checks.
- `EvaluatorTests.test_yaml_matches_approved_config`: PyYAML unavailable.

## Artifacts and acceptance coverage

Synthetic fixture: `tests/test_metrics_edge_cases.py::synthetic_report`.
Six train impressions fit history thresholds; four evaluation impressions exercise
mixed/cold/warm/unknown cohorts, including zero-positive and all-positive labels.
Randomized AUC pairwise-oracle check uses seed 0; no seed is needed for deterministic scoring.

- `outputs/tables/phuc01_synthetic_2026-10-19/per_impression.csv`
- `outputs/tables/phuc01_synthetic_2026-10-19/overall.csv`
- `outputs/tables/phuc01_synthetic_2026-10-19/groups.csv`
- `outputs/tables/phuc01_synthetic_2026-10-19/report.json`

Generation invoked `write_tables(synthetic_report(), 'outputs/tables/phuc01_synthetic_2026-10-19')`.
The persisted JSON was compared to a fresh in-memory result after the final source change: equal.
CSV/JSON round-trip and refusal to overwrite prior run directories are covered by a passing test.

| Criterion | Evidence / remaining work |
|---|---|
| AC1 | Hand-calculated ranking metrics, multiple positives, k boundaries and macro aggregation pass. AUC also matches an independent pairwise oracle on 100 seed-0 samples. Third-party reference parity remains Not Run. |
| AC2 | Strict schema, missing/invalid labels and scores, ties, undefined denominators, duplicate keys and temporal metadata boundaries pass under unittest. Pytest execution remains Not Run. |
| AC3 | Full-candidate cohorts, train-only quantiles, source/cutoff provenance and overall/group/per-impression artifacts pass. YAML loading remains Not Run. Tables are synthetic, not MIND results. |

Status: **Implemented**, not Verified. Complete dependency installation in an authorized
network-enabled environment, then run pytest with no skipped reference/config checks.
The metadata producer still must guarantee that counted clicks precede its declared cutoff;
this evaluator does not inspect the underlying click events. No commit or push was performed.

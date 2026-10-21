# PHUC-02 validation evidence — 2026-10-19

Status: **Verified for the approved PHUC-02 protocol**, seed 0.
This document summarizes observed commands/results; it is not a raw terminal transcript.

## Approved scope and source restrictions

- Official train: `data/raw/train/{behaviors,news}.tsv`; official dev: `data/raw/dev/{behaviors,news}.tsv`.
- Train 09–13/11/2019; validation 14/11; final training source 09–14/11; test = official dev 15/11.
- Daily cutoff at 00:00. Cumulative events are strictly before cutoff; 24h events are in `[cutoff - 24h, cutoff)`.
- Only labeled positive candidates in permitted training impressions contribute events. History is excluded from popularity, never assigned invented timestamps.
- B1 uses keyed SHA-256 pseudorandom scores. B2 zero-count ties keep original candidate order.
- Cold/warm uses cumulative observed-impression counts for every baseline. This is NOT full Ghist popularity or publication-time cold start.
- No hyperparameter tuning, E1, graph models, or multi-seed significance claims.

## Commands and actual outcomes

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_temporal_leakage.py tests/test_metrics_edge_cases.py -q
.\.venv\Scripts\python.exe -m scripts.run_baselines --output-dir outputs/tables/phuc02_mindsmall_seed0_2026-10-19 --predictions-dir outputs/predictions/phuc02_mindsmall_seed0_2026-10-19
git diff --check
```

| Check | Actual result |
|---|---|
| Pytest | **82 passed, 42 subtests passed**, 0 failed, 0 skipped, in 2.82s. Includes 38 PHUC-02 cases and 44 PHUC-01 regression cases. |
| Pytest warning | One `PytestCacheWarning`: WinError 5 writing `.pytest_cache/v/cache`. Tests still passed; subsequent commands can use `-p no:cacheprovider`. No permissions were changed. |
| Real-data runner | Exit 0, manifest `completed`, all 6 model/split runs exported. UTC start `2026-10-19T10:20:35.082407+00:00`, finish `2026-10-19T10:22:25.891074+00:00`. |
| Whitespace | `git diff --check` exit 0; LF/CRLF warnings are normalization notices, not whitespace errors. |
| Independent score audit | Passed for first 20 impressions of each B2 variant/split: **80 impressions, 3,446 candidate scores/counts**. Fresh counters built directly from official train TSV match exported scores and cumulative metadata; cutoffs also match. |
| Provenance audit | All 11 code/requirements SHA-256 hashes match manifest. All 6 runs have expected impression counts, no undefined metrics, and no missing candidate categories. |
| Git artifact isolation | Detailed prediction path confirmed ignored by `git check-ignore`; raw data remains ignored. No commit/push. |

Environment observed: Windows, Python 3.13.5, pytest 9.1.1, PyYAML 6.0.3,
NumPy 2.5.4, scikit-learn 1.9.1. No new dependencies were installed for PHUC-02.

## Verified data coverage

| Partition | Impressions | Clicks | Candidates |
|---|---:|---:|---:|
| Train 09–13/11 | 126,695 | 189,519 | 4,621,015 |
| Validation 14/11 | 30,270 | 46,825 | 1,222,429 |
| Train final 09–14/11 | 156,965 | 236,344 | 5,843,444 |
| Test/dev 15/11 | 73,152 | 111,383 | 2,740,998 |

News metadata union: 65,238 IDs. Candidate category missing count: 0 in all runs.
History quantiles fitted only on train 09–13: Q25 = 8, Q75 = 42; reused unchanged for validation/test.
No impression was discarded for undefined metrics: every metric uses 30,270 validation or 73,152 test impressions.

## Observed metrics (rounded here only)

| Split | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Validation | B1 Random | 0.500969 | 0.202572 | 0.203639 | 0.262599 |
| Validation | B2 cumulative | 0.516296 | 0.215018 | 0.216807 | 0.277841 |
| Validation | B2 24h | 0.518704 | 0.219096 | 0.222133 | 0.282755 |
| Test | B1 Random | 0.499228 | 0.217864 | 0.222431 | 0.285491 |
| Test | B2 cumulative | 0.531752 | 0.238201 | 0.246042 | 0.309817 |
| Test | B2 24h | 0.535480 | 0.239505 | 0.247976 | 0.312459 |

B2 24h is higher on these measured splits at this configuration. These observations
do not establish statistical significance or temporal-leakage effects.

## Artifacts and acceptance criteria

- [Combined metrics](../tables/phuc02_mindsmall_seed0_2026-10-19/metrics.csv).
- [Manifest](../tables/phuc02_mindsmall_seed0_2026-10-19/manifest.json): resolved configs, input hashes, code hashes, Git base revision, versions, source cutoffs and per-run summaries.
- `outputs/tables/phuc02_mindsmall_seed0_2026-10-19/{validation,test}/{B1_random,B2_cumulative,B2_24h}/`: overall/group tables.
- `outputs/predictions/phuc02_mindsmall_seed0_2026-10-19/`: six full prediction CSVs and six metadata JSONs, 705,712,072 bytes total; intentionally not tracked by Git.

| Acceptance criterion | Evidence |
|---|---|
| AC1: reproducible B1, correct cumulative/24h counts | Same-seed/cross-process/order tests; known-event fixtures; independent real-data score audit. |
| AC2: no future/current-cutoff clicks or invented History timestamps | Boundary tests, evaluation-label isolation, duplicate-source rejection, History exclusion and within-day snapshot tests all passed. |
| AC3: end-to-end real-data metrics and provenance for all three baselines | Completed six-run manifest, full validation/test coverage, four metrics, group/per-impression output, input/code hashes and seed 0. |

All PHUC-02 criteria are met for the user-approved variant. PHUC-03–05 remain outside
this implementation. Keep the History exclusion limitation explicit when comparing to Proposal baselines.

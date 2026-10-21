# PHUC-04 validation evidence — 2026-10-20

Status: **Verified for the approved PHUC-04 protocol**, seed 0; AC1–AC3 Passed.
This is a summary of observed commands/results, not a raw terminal transcript.

## Approved scope and timestamp semantics

- User approved E1 B2 cumulative, seed 0, and explicitly required order-independent
  random splitting, impression-time labeling, and comparison against PHUC-02.
- Each positive candidate label contributes one event **at the impression
  timestamp**. This is a proxy: actual click time is not supplied by MIND.
  History is never counted and is never assigned invented timestamps.
- Clean GTS: validation 14/11 uses 09–13/11; test 15/11 uses 09–14/11.
  All source events strictly precede the evaluation day's 00:00 cutoff.
- Random: whole impressions from 09–15/11 sorted by SHA-256 of compact UTF-8 JSON
  `[seed, source, impression_id]`, with source/ID tie-break. Train/validation sizes
  are floor(8N/10), floor(N/10); test gets the remainder. Validation uses random
  train; test uses train+validation. Counts intentionally ignore timestamps.
- Static GTS: same evaluation impressions as clean GTS; count all 09–15/11,
  **including evaluation labels**. Random/Static artifacts and counts are isolated
  from the existing B2 implementation and all PHUC-01–03 artifacts.
- All protocols use the existing evaluator/config, full candidate lists, stable
  candidate-order ranking ties, MIND multi-positive MRR, macro means and null
  undefined metrics. Missing popularity = 0.
- E1 compares overall metrics only: no history thresholds fitted, no prior-click
  counts or cutoff metadata supplied, no claims of group-wise E1 analysis.
- Delta = protocol minus clean GTS; relative percent = 100 × delta / clean GTS.
  Preserve negative values; zero denominator yields null relative delta with a
  reason; missing metric yields null deltas.
- No new dependency, model download, tuning, graph run, commit or push.

## Commands and observed checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_e1_leakage.py tests/test_temporal_leakage.py tests/test_metrics_edge_cases.py tests/test_content_baselines.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m scripts.run_e1_leakage --help
.\.venv\Scripts\python.exe -m scripts.run_e1_leakage --output-dir outputs/tables/phuc04_e1_mindsmall_seed0_2026-10-20 --predictions-dir outputs/predictions/phuc04_e1_mindsmall_seed0_2026-10-20 --splits-dir data/splits/phuc04_e1_mindsmall_seed0_2026-10-20
git diff --check
```

| Check | Observed result |
|---|---|
| E1 + regression tests | **Passed: 156 tests, 42 subtests**, 0 failed/skipped, 34.97 s. Includes 33 E1 tests and 123 existing tests. |
| Order independence | **Passed**: original/reversed/shuffled/interleaved orders yield identical ordered membership for seeds 0, 7, 999. Same impression ID in different sources remains distinct. Label/history/candidate changes do not alter splitting. |
| Count policies | **Passed**: daily cutoff boundary, future random sources, static self-label inclusion, History exclusion, unknown candidates, duplicate/invalid labels, complete candidates, multi-positive/tie/null policies. |
| Comparison safeguards | **Passed**: negative delta/zero denominator/null metric; synthetic reference mismatch stops after clean GTS and records failed status/reason; wrong reference provenance rejected before outputs. These expected-error tests are passing tests, not failed experimental runs. |
| PHUC-02 comparison on real MIND | **Passed: 8/8 means exactly equal**, all valid/undefined/total denominators equal. Both validation and test, no rounding, absolute difference 0. |
| CLI help | **Passed**, exit 0; required output/prediction/split arguments documented. |
| Real-data run | **Passed**, exit 0, manifest `completed`, all 6 protocol/split runs. UTC start `2026-10-20T08:31:22.279078+00:00`, finish `2026-10-20T08:36:45.948411+00:00`. |
| Final artifact/score audit | **Passed**, exit 0: 252,868 evaluation-impression occurrences / 9,645,975 candidate scores across all 6 runs. Raw alignment, independent source Counters, all aggregate means and 46 hashes match. |
| Whitespace | **Passed**, `git diff --check` exit 0. Git LF/CRLF normalization warnings are not whitespace errors. |
| Large-artifact isolation | **Passed** for representative prediction and split-membership paths via `git check-ignore`. |
| Multi-seed/graph/statistical tests | **Not Run** — outside approved PHUC-04 scope. |

No failing check has been observed. Tests were not skipped for missing libraries.

Independent audit evidence: [phuc04_artifact_audit.json](phuc04_artifact_audit.json).
The audit was run as an inline Python command through the repository `.venv`;
it parsed TSV/CSV directly and did not import production E1 helpers. It rebuilt
clean/random/static Counters from permitted source partitions, checked all
exported scores, matched hashes of user/time/history-length/candidate-ID/label
sequences to raw rows, recomputed macro means from per-impression outputs, and
verified split union/disjointness. It independently reconstructed hash ordering
from reversed full-dataset keys, verified identical GTS/Static membership, and
confirmed 20 PHUC-01–03 source/config/test files have no Git diff.
252,868 is the sum of evaluation occurrences across runs, not unique input rows.

Environment: Windows, Python 3.13.5, pytest 9.1.1, PyYAML 6.0.3, NumPy 2.5.4,
scikit-learn 1.9.1. No PLM/embedding model is used by E1, so a PLM revision is
not applicable. Versions are also recorded per run in the manifest.

## Data coverage

Total: 230,117 impressions; 347,727 positive-label events; 8,584,442 candidates.

| Protocol partition | Impressions | Positive-label events | Candidates |
|---|---:|---:|---:|
| GTS train 09–13 | 126,695 | 189,519 | 4,621,015 |
| GTS validation 14 | 30,270 | 46,825 | 1,222,429 |
| GTS final train 09–14 | 156,965 | 236,344 | 5,843,444 |
| GTS test 15 | 73,152 | 111,383 | 2,740,998 |
| Random train | 184,093 | 278,080 | 6,865,321 |
| Random validation | 23,011 | 34,725 | 855,547 |
| Random final train | 207,104 | 312,805 | 7,720,868 |
| Random test | 23,013 | 34,922 | 863,574 |

Static uses the same GTS partitions; its count source always contains all 230,117
impressions / 347,727 positives. Seed = 0; candidate lists are never sampled.

## Mandatory PHUC-02 equivalence

Reference: `outputs/tables/phuc02_mindsmall_seed0_2026-10-19/{metrics.csv,manifest.json}`.
Input file SHA-256 hashes, data kind, seed, completed status and evaluation config
were checked before running. All eight B2 cumulative aggregates match exactly.
Tolerance allowed by the checker: absolute 1e-12, relative 0; actual difference 0.

| Split | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Validation: PHUC-02 = E1 clean | 0.5162956361101749 | 0.21501765844013485 | 0.21680688078767524 | 0.2778410639545543 |
| Test: PHUC-02 = E1 clean | 0.531751528369352 | 0.23820138108047803 | 0.24604210472429233 | 0.3098167064622724 |

[Detailed comparison](../tables/phuc04_e1_mindsmall_seed0_2026-10-20/phuc02_comparison.csv).
No mismatch requires explanation or protocol change.

## Experiment results and acceptance

Observed means below are rounded only for display. CSVs retain full precision.
All metrics have `n_valid = n_impressions`, `n_undefined = 0` across all six runs.

| Split | Protocol | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Validation | GTS clean | 0.516296 | 0.215018 | 0.216807 | 0.277841 |
| Validation | Random | 0.602270 | 0.298095 | 0.318631 | 0.372393 |
| Validation | GTS static | 0.586619 | 0.249164 | 0.260949 | 0.319224 |
| Test | GTS clean | 0.531752 | 0.238201 | 0.246042 | 0.309817 |
| Test | Random | 0.600933 | 0.298722 | 0.320846 | 0.374188 |
| Test | GTS static | 0.599364 | 0.289196 | 0.311321 | 0.372301 |

Test deltas relative to GTS clean (absolute metric units, then relative percent):

| Protocol | Metric | Absolute delta | Relative delta |
|---|---|---:|---:|
| Random | AUC | +0.069181 | +13.0100% |
| Random | MRR | +0.060521 | +25.4073% |
| Random | nDCG@5 | +0.074804 | +30.4028% |
| Random | nDCG@10 | +0.064371 | +20.7771% |
| GTS static | AUC | +0.067612 | +12.7150% |
| GTS static | MRR | +0.050995 | +21.4083% |
| GTS static | nDCG@5 | +0.065278 | +26.5314% |
| GTS static | nDCG@10 | +0.062485 | +20.1683% |

Full [metrics](../tables/phuc04_e1_mindsmall_seed0_2026-10-20/metrics.csv),
[validation/test deltas](../tables/phuc04_e1_mindsmall_seed0_2026-10-20/inflation.csv),
and [run manifest](../tables/phuc04_e1_mindsmall_seed0_2026-10-20/manifest.json).

### Interpretation and limits

For this fixed B2 configuration, using the whole-period count source, including
evaluation labels, increases all four test metrics on **the same 73,152 impressions**.
Static AUC rises by 0.067612 (12.7150% relative) and nDCG@10 by 0.062485
(20.1683% relative). These are observed offline increases under the intentional
leakage protocol, not evidence of improved predictive quality.

Random also yields higher metrics, but its 23,013 test impressions span all seven
days and differ from GTS's 73,152 impressions on 15/11. Its delta mixes changed
test composition, changed count-source composition/size, and temporal leakage.
It cannot be interpreted as a pure leakage effect or as evidence that Random is
more leaky than Static. No significance test, multi-seed conclusion, online claim,
actual-click-time assertion, or graph-model generalization is made.

### Observed leakage provenance

| Protocol | Split | Source/evaluation overlap | Evaluation impressions exposed to source events at/after day cutoff |
|---|---|---:|---:|
| GTS clean | Validation | 0 | 0 |
| GTS clean | Test | 0 | 0 |
| Random | Validation | 0 | 23,011 |
| Random | Test | 0 | 23,013 |
| GTS static | Validation | 30,270 | 30,270 |
| GTS static | Test | 73,152 | 73,152 |

These counts describe events available in the source, not necessarily events
matching a particular scored candidate. Relative to the impression timestamp
itself (rather than day cutoff), all Random evaluation impressions have future
source events; 73,151 of 73,152 Static test impressions do. Static also includes
the evaluation impression's own positive labels. Future/self-label leakage is
intentional and labeled in manifest, per-impression metadata and summary CSVs.

### Acceptance criteria

- **AC1 Passed:** 24 aggregate metric rows cover all three protocols on validation
  and test; complete candidates and split identity/size/provenance verified on real
  data. All eight clean aggregates and their denominators exactly match PHUC-02.
- **AC2 Passed:** 16 delta rows with exact unrounded calculations; tests cover null,
  zero baseline and negative delta. Leaky source policies are explicit and artifacts
  are isolated. Random membership is independent of read order in unit tests and
  the full-dataset audit. Every exported score matches an independent count source.
- **AC3 Passed:** measured increases and their limits are recorded above. Static
  compares the same test impressions; Random includes population/source changes.
  No multi-seed, significance, online or graph claim is made.

No unresolved blocker remains for approved PHUC-04. PHUC-05, commits and pushes
remain pending separate user confirmation.

## Implementation files

- `src/data_loader/random_split.py`: deterministic whole-impression assignment.
- `src/baselines/e1_popularity.py`: isolated leaky count source with explicit protocol.
- `src/evaluation/leakage_comparison.py`: delta policies and PHUC-02 comparison.
- `scripts/run_e1_leakage.py`, `configs/e1_protocols.yaml`: orchestration/provenance.
- `tests/test_e1_leakage.py`: deterministic fixtures and end-to-end reference/score tests.
- `README.md`, `docs/WORK_STATUS.md`, this log: protocol, usage and acceptance evidence.

Existing PHUC-01–03 source/config/tests are unchanged. The outdated scaffold
statement in AGENTS.md was left intact; this log and WORK_STATUS describe observed
implementation state. README's proposal links were corrected to SNA-Proposal-v2.md.

# PHUC-03 Validation — 2026-10-19

Recorded summary of observed commands/results, not a complete terminal transcript.
Scope: approved B3/B4 implementation; no E1, no commit/push, no evaluator changes.
Current status: **Verified** after the continuation recorded below. Earlier failures and
partial acceptance are preserved as historical evidence; B3 was not rerun.

## Approved policy

- Current-impression History only, all entries including repeats; no label updates or
  future-impression aggregation. History is assumed before that impression, not proven
  before day-start. No fabricated timestamps; B2 counts remain unchanged.
- B3: empirical subcategory frequency. Category fallback only if candidate subcategory
  or the entire usable history-subcategory profile is absent. Unseen valid subcategory = 0.
- B4: frozen unit-normalized news embeddings, float64 history mean, cosine similarity.
  Empty/unusable history, missing candidate content, or user norm <= 1e-12 => score 0.
  Zero fallback can rank above negative cosine; no popularity blend.
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, verified from the
  [official model API](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2)
  during preparation. Config pins this revision rather than mutable `main`.
  Text: stripped title + `. ` + stripped abstract (omit empty parts), 256 word pieces,
  384 dimensions, CPU, batch 64, seed 0. Actual encoder execution was **Not Run** at the
  first handoff; it is verified in the continuation below.
- GTS, impression-level AUC/MIND multiple-positive MRR/nDCG@5/@10, stable candidate-order
  ties, and train-only group thresholds remain PHUC-01/02 policies.

## Implementation

- Modified `src/data_loader/mind_reader.py`, `src/baselines/base.py`: per-impression
  history IDs and full news content; B1/B2 positional APIs remain compatible.
- Implemented `src/baselines/b3_category.py`, `src/baselines/b4_semantic_plm.py`,
  `scripts/cache_plm_embeddings.py`; extended `scripts/run_baselines.py`.
- Added `configs/content_baselines_config.yaml`, `tests/test_content_baselines.py`;
  updated `requirements.txt`, `README.md`, `docs/WORK_STATUS.md`.
- Cache validates mapping, content hash, model/revision/preprocessing, file checksums,
  float32 shape, finite unit vectors, completion state and synthetic/real identity.
  Existing cache/output directories are never overwritten.

## Initial checks (historical; before real-cache continuation)

| Check | Result | Evidence |
|---|---|---|
| PHUC-01/02 regression after integration | Passed | 82 tests, 42 subtests, 2.54s |
| Combined suite including PHUC-03 | Passed | 123 tests, 42 subtests, 3.02s; no Failed/Skipped |
| Cache writer with mock encoder | Passed | Included in 41 new tests; checks pinned revision, frozen/eval mode, normalization, failure manifest, no overwrite; **not real encoder validation** |
| B3 actual MIND-small | Passed | Two split runs completed, process exit 0; artifacts below |
| Independent raw-TSV score/order audit | Passed after correcting audit command | 50 impressions per split, 100 total; 4,016 candidate scores/order matched |
| Code provenance/metric denominators | Passed | All 14 code hashes matched current files; n_valid = n_impressions for all eight metric rows |
| `git diff --check` | Passed | No whitespace errors; Git warned about LF/CRLF conversion |
| Large artifact ignore rules | Passed | `git check-ignore` matched prediction/cache paths |
| PLM dependency install | Failed | pip reported no matching distribution; no successful installation |
| Direct PyPI connectivity | Failed | `urllib.request.urlopen` raised `URLError` wrapping socket `PermissionError: [WinError 10013]` |
| Real cache command | Failed at dependency preflight | `ModuleNotFoundError: No module named 'torch'`, wrapped with install-requirements guidance; no cache directory created |
| Real MiniLM inference/B4 metrics | Not Run | Missing dependencies and real cache; no fabricated embedding or B4 result |

Combined test command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_content_baselines.py tests/test_temporal_leakage.py tests/test_metrics_edge_cases.py -q -p no:cacheprovider
```

Tests use known vectors/synthetic TSV and temporary mock caches, not downloaded PLM
weights. 42 subtests are not 42 additional independent collected tests.
The first ad hoc audit used the wrong CSV column `news_ids` and failed with `KeyError`.
After checking the header, the command used `candidate_ids` and passed; no source,
prediction, or metric artifact was changed to make the audit pass.

## Actual B3 run

```powershell
.\.venv\Scripts\python.exe -m scripts.run_baselines --content-config configs/content_baselines_config.yaml --models B3_category --output-dir outputs/tables/phuc03_b3_mindsmall_seed0_2026-10-19 --predictions-dir outputs/predictions/phuc03_b3_mindsmall_seed0_2026-10-19
```

Manifest UTC timestamps: 12:18:29.261319 to 12:19:08.060942 (evaluation/artifact stage,
excluding initial parsing). News union: 65,238. Seed 0; no tuning or repeated test run.

| Split | Impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Validation 14/11 | 30,270 | 0.586645207880361 | 0.262718713844252 | 0.2784275017315943 | 0.33717686042829337 |
| Test/dev 15/11 | 73,152 | 0.6046322994805222 | 0.29368867078449784 | 0.31955177868709744 | 0.37869487135356694 |

| Diagnostic | Validation | Test |
|---|---:|---:|
| Empty-history / no-profile impressions | 772 | 2,214 |
| History entries, including repeats | 1,001,673 | 2,362,514 |
| Candidate entries | 1,222,429 | 2,740,998 |
| Zero-fallback candidates due to no profile/metadata | 30,575 | 82,907 |
| Missing history/candidate metadata | 0 | 0 |
| Missing category/subcategory metadata | 0 | 0 |
| Category-fallback candidates | 0 | 0 |

Counter keys absent from a run's `content_diagnostics` mean zero occurrences. Diagnostic
counts overlap; zero-fallback is not the count of every naturally zero B3 score.

Artifacts:

- [metrics.csv](../tables/phuc03_b3_mindsmall_seed0_2026-10-19/metrics.csv)
- [manifest.json](../tables/phuc03_b3_mindsmall_seed0_2026-10-19/manifest.json), including
  config, input hashes, code hashes, split statistics, library versions, diagnostics.
- `outputs/tables/phuc03_b3_mindsmall_seed0_2026-10-19/{validation,test}/B3_category/`:
  `overall.csv`, `groups.csv`.
- `outputs/predictions/phuc03_b3_mindsmall_seed0_2026-10-19/`: full per-impression CSV
  and metadata JSON per split (Git-ignored).

Initially observed environment: Windows, Python 3.13.5; NumPy 2.5.4, scikit-learn 1.9.1,
PyYAML 6.0.3, pytest 9.1.1. No observed torch/sentence-transformers installed version.

## Historical blocker and partial acceptance (before continuation)

Two pip attempts (`sentence-transformers>=5,<6`, `torch>=2.6,<3`, including explicit
PyPI index) failed. The network probe establishes blocked socket access, not Python
incompatibility. We did not change network permissions or work around restrictions.

- AC1: **Passed** for numerical scores, fallback/zero norm, alignment and synthetic integration.
- AC2: **Partial** — cache machinery tested; real PLM cache/revision execution pending.
  History isolation is tested; individual click-time assumptions cannot be proven from MIND.
- AC3: **Partial** — real B3 metrics/diagnostics available; real B4 results missing.

Next: install requirements in the user's network-enabled VS Code terminal; generate
the pinned cache, then run **B4 only** using README commands and new output directories.
Validate actual cache/manifests, record real B4 metrics and rerun tests in that environment
before marking PHUC-03 Verified. Do not rerun B3 test to tune settings or move to PHUC-04.

Research limits: one seed; no significance claim; raw History has no click timestamps
and news has no publication timestamps. A frozen external PLM is not proof that its
pretraining data/weights were available in November 2019.

## Continuation — real cache and B4 verified, 2026-10-19

The user installed dependencies and generated the cache outside the restricted session.
Codex verified those artifacts and ran **B4 only**; no B3 rerun, source-code/config change,
dependency installation, full-cache regeneration, commit, or push in this continuation.
OD-09 is resolved for PHUC-03; this does not establish changed sandbox network permissions.

### Cache and model verification — Passed

- Cache: `data/cache/phuc03_minilm_2026-10-19/`, backend `sentence_transformers`,
  status `completed`, dataset `MIND-small`. Model/revision match the approved policy above.
- Shape `(65238, 384)`, float32; every news ID has content and one embedding, no duplicates,
  missing text, or nonfinite values. Norm range: 0.9999998807907104–1.0000001192092896.
- Mapping/vector checksums, complete news-text digest, both raw news input hashes and all
  four cache-generation code hashes matched. Cache-manifest SHA256:
  `308dbe26502b523a94d27d327aa48f51a1cc300e50283e3819614c081410f1d9`.
- Cache library versions match the installed environment: NumPy 2.5.4, torch 2.14.1,
  sentence-transformers 5.7.0, transformers 5.20.1; Python 3.13.5.
- Real encoder loaded the pinned revision with `local_files_only=True`, HF/Transformers
  offline mode, eval mode, gradients disabled, CPU and deterministic algorithms. Eight
  news texts were re-encoded in memory, not written back to cache: `N1`, `N17337`,
  `N24676`, `N32013`, `N39353`, `N46692`, `N5403`, `N61369`.
  `assert_allclose(atol=1e-5, rtol=1e-5)` Passed; maximum absolute error
  `7.450580596923828e-08` (probe batch size 8 versus original cache batch 64).
- Nonblocking `FutureWarning`: `get_sentence_embedding_dimension` renamed to
  `get_embedding_dimension` in the installed library. Current call still succeeded;
  source was not changed during validation.

### B4 execution — Passed

```powershell
.\.venv\Scripts\python.exe -m scripts.run_baselines --content-config configs/content_baselines_config.yaml --models B4_semantic_plm --embedding-cache data/cache/phuc03_minilm_2026-10-19 --output-dir outputs/tables/phuc03_b4_mindsmall_seed0_2026-10-19 --predictions-dir outputs/predictions/phuc03_b4_mindsmall_seed0_2026-10-19
```

Exit 0, manifest `completed`, selected models exactly `[B4_semantic_plm]`, two runs.
Manifest UTC timestamps: 16:34:27.482844 to 16:35:35.378149 (excludes initial parsing).
All 14 run code hashes and all input hashes matched current files; embedded cache
manifest fingerprint matched the actual cache. Seed 0, no tuning or repeated test run.

| Split | Impressions | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Validation 14/11 | 30,270 | 0.6394181909829567 | 0.2949256855258288 | 0.3167859642607704 | 0.37641080843396424 |
| Test/dev 15/11 | 73,152 | 0.634950927064799 | 0.30791306712545236 | 0.3365764476131443 | 0.3955805043037474 |

All metric rows have `n_valid = n_impressions`, `n_undefined = 0`. Empty-history
impressions: validation 772, test 2,214. Zero-fallback candidates: 30,575 / 82,907.
All 1,001,673 / 2,362,514 history entries were usable. Missing history/candidate
metadata or text and zero-norm nonempty user vectors: zero occurrences. Group thresholds
are Q25=8 and Q75=42 from 126,695 train impressions only, max time 2019-11-13T23:59:21;
unchanged between validation and test.

Artifacts:

- [B4 metrics](../tables/phuc03_b4_mindsmall_seed0_2026-10-19/metrics.csv)
- [B4 manifest](../tables/phuc03_b4_mindsmall_seed0_2026-10-19/manifest.json), including
  cache provenance/library versions, per-split diagnostics and group thresholds.
- `outputs/tables/phuc03_b4_mindsmall_seed0_2026-10-19/{validation,test}/B4_semantic_plm/`:
  `overall.csv`, `groups.csv`.
- `outputs/predictions/phuc03_b4_mindsmall_seed0_2026-10-19/`: full prediction CSV and
  metadata JSON (Git-ignored). Existing B3 artifacts were preserved.

### Regression and temporal audit — Passed

- Repeated the combined test command above in the newly installed environment:
  **123 passed, 42 subtests passed in 3.24s**, no Failed/Skipped.
- Independently read raw TSV History/candidates and computed mean-vector cosine, with
  explicit candidate-vector normalization, for the first 50 impressions of each split.
  **100 impressions / 4,016 candidate scores** matched (`atol=rtol=1e-6`), maximum absolute
  error `5.282275261642155e-08`; candidate order and labels matched raw data exactly.
- On those real rows, flipping evaluation labels left scores exactly unchanged.
  Scoring an artificial later-impression context with different History before rescoring
  the original also left scores unchanged. These in-memory probes are not extra
  benchmark runs and their predictions were not added to reported metrics.
- Independently counted positive candidates from raw **official train only** with
  timestamp strictly before 14/11 or 15/11. Every sampled prior-click count and metadata
  cutoff matched; no dev labels or untimestamped History entered those counts.
- Unit tests cover equal/future cutoff rejection, 24-hour boundary semantics, history
  isolation, full candidate preservation, cache corruption, fallback and zero norms.
- `git diff --check`: Passed (LF/CRLF warnings only). Large cache/prediction paths remain
  Git-ignored. No source/config changes were needed for the successful checks.

### Final acceptance

| Criterion | Result | Evidence |
|---|---|---|
| AC1 | Passed | Known-vector/formula and fallback tests, full regression, independent real-score audit |
| AC2 | Passed under approved assumptions | Real pinned-model cache with mapping/checksums/versions; real-encoder parity; History/label isolation and strict metadata click cutoffs |
| AC3 | Passed | Existing real B3 plus new real B4 validation/test metrics, group tables, diagnostics, config/seed/cache provenance |

PHUC-03 is **Verified for the approved seed-0 protocol**. This is not proof of individual
History click times or historical availability of pretrained weights/text, and is not
a multi-seed/significance result. No unresolved acceptance blocker remains. Next action:
user review/confirmation before planning PHUC-04; no automatic ticket transition.

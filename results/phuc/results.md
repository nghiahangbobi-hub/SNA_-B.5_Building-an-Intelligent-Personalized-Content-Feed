# B1–B4 and E1 results

Seed 0; existing measurements, no pooling across protocols.

| Experiment | Protocol | Model | Split | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---:|---:|---:|---:|
| E1 | gts_clean | B2_cumulative | test | 0.531752 | 0.238201 | 0.246042 | 0.309817 |
| E1 | gts_clean | B2_cumulative | validation | 0.516296 | 0.215018 | 0.216807 | 0.277841 |
| E1 | gts_static | B2_cumulative | test | 0.599364 | 0.289196 | 0.311321 | 0.372301 |
| E1 | gts_static | B2_cumulative | validation | 0.586619 | 0.249164 | 0.260949 | 0.319224 |
| E1 | random | B2_cumulative | test | 0.600933 | 0.298722 | 0.320846 | 0.374188 |
| E1 | random | B2_cumulative | validation | 0.602270 | 0.298095 | 0.318631 | 0.372393 |
| baseline | gts_clean | B1_random | test | 0.499228 | 0.217864 | 0.222431 | 0.285491 |
| baseline | gts_clean | B1_random | validation | 0.500969 | 0.202572 | 0.203639 | 0.262599 |
| baseline | gts_clean | B2_24h | test | 0.535480 | 0.239505 | 0.247976 | 0.312459 |
| baseline | gts_clean | B2_24h | validation | 0.518704 | 0.219096 | 0.222133 | 0.282755 |
| baseline | gts_clean | B2_cumulative | test | 0.531752 | 0.238201 | 0.246042 | 0.309817 |
| baseline | gts_clean | B2_cumulative | validation | 0.516296 | 0.215018 | 0.216807 | 0.277841 |
| baseline | gts_clean | B3_category | test | 0.604632 | 0.293689 | 0.319552 | 0.378695 |
| baseline | gts_clean | B3_category | validation | 0.586645 | 0.262719 | 0.278428 | 0.337177 |
| baseline | gts_clean | B4_semantic_plm | test | 0.634951 | 0.307913 | 0.336576 | 0.395581 |
| baseline | gts_clean | B4_semantic_plm | validation | 0.639418 | 0.294926 | 0.316786 | 0.376411 |

E1 Random uses a different evaluation population. Static intentionally includes evaluation labels.
B3/B4 use current-impression History; individual click times are unavailable.
These are single-seed results, not significance estimates.

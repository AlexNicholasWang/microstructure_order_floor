# Provisional OFI factor study: July 29–31, 2025

**Research context:** UCLA Mathematics / MSF, decomposition of equity order flow for return forecasting. This document records a finished *exploratory computational experiment*, not a validated forecast.

## Data construction

- Universe: 80 equities (see `../../universe_candidates.csv`).
- Trading days: July 29, 30, and 31, 2025. Regular-session 09:30–16:00 Eastern.
- Source: message-only, compressed, LOBSTER-style per-symbol event archives available on UCLA `lobster2`, not snapshots.
- Aggregation: 78 five-minute buckets per trading day; raw tensor `3 × 78 × 80`, or a `234 × 80` stacked panel.
- Provisional Level 1 quotes: reconstruct outstanding displayed orders from message types; compute eventwise best-quote OFI. Opening-book coverage and quote correctness are **not independently verified**.
- Volume denominator: sum of share quantities of executions of event types 4 and 5, excluding cross trades. This is **not validated consolidated exchange volume**.
- Depth denominator: elapsed-time-weighted mean displayed best bid + best ask share quantity in each bucket.
- Historical baseline: trailing, same-clock-bucket raw OFI averaged across strictly earlier available days, exponentially downweighted using a five-day half-life. The first day has no earlier-day baseline.

## Processing and completeness

| Check | Result |
| --- | ---: |
| Archive runs reported `processed_provisional` | 240 / 240 |
| Raw OFI matrix size | 234 × 80 |
| Potential raw OFI entries | 18,720 |
| Missing raw OFI entries | 0 |
| Missing volume-normalized entries | 3 |
| Missing depth-normalized entries | 0 |

A processing-success status does **not** imply reconstructed quotes are correct. Missing volume denominators cause the corresponding normalized OFI entries to be absent rather than silently filled. The downstream factor analysis excludes rows with any nonfinite ticker entry.

## Factor study

**Train:** July 29 and 30. **Held out:** July 31. Each ticker is standardized using only training statistics. A weighted SVD uses a five-day half-life on the training buckets. The plotted SVD spectrum and explained variance are **training-sample** estimates; the visual component heatmaps are for the held-out day.

| Statistic | Normalized by Level 1 depth | Normalized by executed volume |
| --- | ---: | ---: |
| Complete training buckets | 156 | 153 |
| Complete held-out buckets | 78 | 78 |
| Heuristic SVD elbow | 15 | 12 |
| SVD rank reaching 80% training variance | 31 | 33 |
| Training variance explained at chosen rank | 80.61% | 80.45% |
| Robust PCA training nonzero sparse fraction | 71.60% | 70.31% |
| Robust PCA iterations | 22 | 22 |

The two normalizations give similar broad SVD spectra. **31–33 components out of 80** are needed to reach the 80% training threshold, so the observed panel is not highly low-rank under this criterion. The estimated elbow is a heuristic, not a formal identified factor count.

The training Robust PCA objective is approximately `X=L+S+E`. Its reported relative reconstruction error is below `1e-5`, which is an *algorithmic reconstruction diagnostic*, not predictive accuracy. Roughly 70% of entries are nonzero in the training sparse part, raising doubts about a genuinely sparse economic interpretation. Held-out sparse components use a separate threshold heuristic and may appear nearly blank; **do not equate the held-out sparse picture with the training sparse fraction**.

Residual OFI lag / accumulated-past-flow correlations were exported, but **not independently assessed for statistical significance or price prediction**. No forward-return forecast, return dataset, or out-of-sample trading performance has been established.

## Generated files on the authorized host

`team_results/` contains stacked five-minute matrices, `.npy` 3D tensors, a reconstruction QC table, `manifest.json`, and a strictly historical weighted raw-OFI baseline.

`team_results/factors_depth/` and `team_results/factors_volume/` each contain:

- `svd_rank_selection.png` and `svd_rank_selection.csv`
- `svd_components.png` (held-out observed, SVD low rank, residual)
- `rpca_components.png` (held-out observed, robust low rank, sparse, remainder)
- `factor_summary.json`, `svd_loadings.csv`
- `heldout_standardized.csv`, `svd_lowrank.csv`, `svd_residual.csv`
- `rpca_lowrank.csv`, `rpca_sparse.csv`, `rpca_error.csv`
- `heldout_residual_persistence.csv`

The repo includes **only the aggregate JSON summaries** and source needed to regenerate these files; do not upload the market-data-derived outputs or plots publicly without verifying permission to redistribute.

## Outstanding scientific work

1. Independently verify reconstructed best bid, best ask and displayed depth against an actual order-book/snapshot reference, including the opening state and unusual events.
2. Audit QC warnings, zero-volume denominators, quote coverage, ticker/date alignment, and potential corporate-action effects.
3. Tune Robust PCA with meaningful sparsity/reconstruction tradeoffs and compare against a consistent SVD baseline.
4. Evaluate factor stability on additional trading dates, with chronologically separated training and held-out data.
5. Analyze residual OFI persistence with uncertainty estimates and **then** test pre-registered future-midquote / residual-return targets with leakage controls.

## Code entry points

- `scripts/lobster2/reconstruct_ofi.py`: provisional one-minute Level 1 reconstruction (message-only).
- `scripts/lobster2/build_ofi_matrix.py`: one-day, one-minute multi-ticker assembly; specify `--script scripts/lobster2/reconstruct_ofi.py`.
- `scripts/lobster2/ofi_team_pipeline.py`: three-day/five-minute construction, normalizations, weighted historical mean, SVD and exploratory Robust PCA.

Older `scripts/run_pipeline.py` and `scripts/build_level1_ofi_matrix.py` are a **separate paired-message/orderbook baseline**, retained for reproducibility.

## Follow-up tools added November 5, 2026

The repo now includes scripts/lobster2/audit_research.py and scripts/lobster2/refine_rpca.py. Both run from cached team_results panels, without reopening the 240 stock-day archives.

~~~bash
python scripts/lobster2/audit_research.py --input team_results --output team_results/audit
OPENBLAS_NUM_THREADS=1 python scripts/lobster2/refine_rpca.py --input team_results --metric ofi_per_depth --train-days 2
OPENBLAS_NUM_THREADS=1 python scripts/lobster2/refine_rpca.py --input team_results --metric ofi_per_volume --train-days 2
~~~

The audit checks reconstruction QC warnings, stock-day/timestamp alignment, normalization formula consistency, held-out decomposition identities, and descriptive residual OFI persistence. The sensitivity study varies regularization *on training dates only* and compares the low-rank/sparsity tradeoff against same-rank SVD on a held-out day.

These checks are not independent quote validation or return forecasting. Detailed audit outputs derived from licensed market data remain private pending approval.

## November 5 cached-panel audit and RPCA sensitivity results

The subsequent audit on lobster2 reported `PASS_WITH_PROVISIONAL_QUOTES`, with no normalization formula, timestamp alignment or additive decomposition consistency errors. **This is not independent book/quote validation.**

| Held-out measure (July 31, 2025) | Depth normalized | Volume normalized |
| --- | ---: | ---: |
| SVD rank used | 31 | 33 |
| Held-out standardized squared-energy captured by SVD | 55.09% | 56.05% |
| Training Robust PCA nonzero sparse fraction | 71.60% | 70.31% |
| Held-out thresholded sparse nonzero fraction | 2.84% | 0.63% |
| Median lag-1 OFI residual correlation across 80 stocks | 0.0600 | 0.0532 |
| Median 3-bucket past-sum-to-next residual correlation | 0.0566 | 0.0682 |

Robust PCA tuning on the two training dates gave the following regularization tradeoff:

| Lambda multiplier | Depth: sparse % / rank(L) | Volume: sparse % / rank(L) |
| ---: | ---: | ---: |
| 0.5 | 99.85% / 3 | 99.89% / 3 |
| 1.0 | 71.60% / 52 | 70.31% / 54 |
| 1.5 | 24.83% / 80 | 24.26% / 80 |
| 2.0 | 8.23% / 80 | 7.88% / 80 |
| 3.0 | 1.31% / 80 | 1.30% / 80 |

A full-rank low-rank component trivially reconstructs the held-out 80-dimensional panel, so 100% projection capture at those regularization settings **must not** be reported as predictive success. Further model/threshold design is required. A three-day study is too short for reliable stability inference; the longer period has been selected but the 20-day build had not yet been completed at the time of this note.

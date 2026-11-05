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

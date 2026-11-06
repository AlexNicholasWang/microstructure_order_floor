# Equity order-flow decomposition: 19-session empirical report

**UCLA Mathematics / MSF group — November 6, 2026**

**Status:** The preliminary end-to-end experiment is complete. Independent quote validation, statistical significance, and executable trading analysis remain outstanding.

> **Main finding:** SVD-residualized order-flow imbalance (OFI) improved pooled out-of-sample (OOS) R² against raw standardized OFI in **5 of 6** metric × horizon comparisons. **All OOS R² values were nevertheless negative**, and prediction correlations were small. We have **not** demonstrated a dependable trading signal.

## Research question and design

Does removing common cross-equity flow factors from observed OFI improve forecasts of subsequent midpoint returns?

- **Universe and source:** 80 equities, licensed LOBSTER-style message-only archives on authorized UCLA lobster2 server; no independent snapshots used.
- **Dates:** July 7–31, 2025, 19 full regular sessions, 78 five-minute buckets per session.
- **Feature tensor:** 19 × 78 × 80, or 1,482 × 80 stacked (118,560 potential ticker/bucket values).
- **Calendar correction:** July 3, 2025 was a 13:00 Eastern early-close session, so it was excluded in full rather than treating a 78-bucket calendar assumption as valid.
- **Train/test split:** 14 earlier sessions (July 7–24) train; 5 strictly later sessions (July 25, 28, 29, 30, 31) test.
- **Targets:** Reconstructed end-of-bucket midpoint log returns from end of current 5-minute OFI bucket to 1, 3, 6 buckets ahead (5/15/30 minutes). No cross-day labels.
- **Models:** Per-stock ridge regression on current plus intraday trailing-three-bucket OFI; standardized on train only. Compare raw standardized OFI vs OFI residual after removing train-only weighted-SVD common components. Run separately for Level 1 depth and executed-volume normalization.
- **Benchmark:** Per-stock mean **training** return; pooled OOS R² uses pooled test squared errors relative to this benchmark.

**Price and flow observations are provisionally reconstructed from message events, not verified against independently sourced bid/ask snapshots.**

## Forecasting results

All entries are *pooled test OOS R²*; higher is better. Δ is residual minus raw.

| Normalization | Horizon | Raw OFI | SVD residual | Δ | Stocks improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Depth | 5 min | -0.002511 | -0.002161 | +0.000350 | 43/80 |
| Depth | 15 min | -0.008491 | -0.004921 | +0.003570 | 48/80 |
| Depth | 30 min | -0.007329 | -0.006071 | +0.001259 | 42/80 |
| Volume | 5 min | -0.002336 | -0.002359 | -0.000022 | 41/80 |
| Volume | 15 min | -0.004392 | -0.001951 | +0.002441 | 49/80 |
| Volume | 30 min | -0.005184 | -0.003361 | +0.001822 | 40/80 |

The relative 15-minute advantage occurs for both normalizations. However, the count of stocks benefiting is only modestly above half, and all pooled forecast R² values remain negative. **Relative improvement should not be described as positive predictive power.**

Median *per-stock* OOS R² gains for depth are +0.001378 (5 min), +0.003676 (15 min), +0.000475 (30 min); for volume they are +0.000141, +0.002478, +0.000028.

**Prediction correlations were near zero.** Statistical significance is unassessed; correlated stocks, overlapping forward returns, and only five held-out dates preclude naive independent-observation inference.

## Factor and reconstruction diagnostics

| Measure | Depth-normalized | Volume-normalized |
| --- | ---: | ---: |
| Complete training buckets | 1,092 | 1,091 |
| Complete test buckets | 390 | 387 |
| Training SVD rank needed for approximately 80% variance | 42 | 43 |
| Training variance captured | 80.49% | 80.06% |
| Held-out SVD projected OFI energy | 65.80% | 72.28% |
| Median held-out lag-one residual OFI autocorrelation | 0.1230 | 0.0658 |
| Default training RPCA sparse nonzero fraction | 86.40% | 85.89% |
| Held-out thresholded RPCA sparse nonzero fraction | 0.78% | 0.87% |

The 80% variance threshold selects relatively many factors (42–43 out of 80). The held-out projection captures less energy than the training fit. Lag-one autocorrelation concerns **OFI persistence**, not price-return prediction.

### RPCA regularization sensitivity

| Penalty multiplier | Depth sparse nonzeros / rank(L) | Volume sparse nonzeros / rank(L) |
| ---: | ---: | ---: |
| 0.5 | 99.91% / 3 | 99.93% / 3 |
| 1.0 | 86.40% / 55 | 85.89% / 59 |
| 1.5 | 27.05% / 80 | 26.35% / 80 |
| 2.0 | 9.79% / 80 | 8.93% / 80 |
| 3.0 | 2.31% / 80 | 1.80% / 80 |

The RPCA decomposition is **very sensitive to the regularization penalty**. At full rank 80, 100% projection-energy capture is mathematically trivial. RPCA outputs were used for exploratory decomposition **only**; they were **not** used to produce the return-prediction table.

### Quality control

- Internal audit: **PASS_WITH_PROVISIONAL_QUOTES**, no normalization or matrix reconstruction consistency errors.
- Missing OFI values: raw **0**, depth-normalized **0**, volume-normalized **4** (zero-volume denominators).
- Reconstruction warnings: bad lines **0**, invalid quotes **0**, nonmonotonic **0**.
- Midpoint panel: **1,482 × 80**, **0** missing.

**Internal consistency is not independent book validation.**

## Private artifacts on lobster2

The detailed files are stored under the git-ignored directory team_results_regular/ on lobster2. Share only with collaborators authorized to receive derivatives of the licensed dataset.

| File/path | Intended team use |
| --- | --- |
| factors_depth/svd_components.png and factors_volume/svd_components.png | Observed, low-rank and residual OFI visualizations |
| factors_depth/svd_rank_selection.png and factors_volume/svd_rank_selection.png | Rank-selection curves |
| factors_depth/rpca_components.png and factors_volume/rpca_components.png | Exploratory RPCA component visualizations |
| rpca_tuning/*_sensitivity.png | Regularization-sensitivity figures |
| factors_*/factor_summary.json | Training factor fit settings and diagnostics |
| forward_tests/*/summary_oos_results.csv | Pooled 5/15/30-minute OOS results |
| forward_tests/*/per_stock_oos_results.csv | Per-stock OOS scores, internal only |
| audit/audit.json and audit/audit.md | Detailed data-consistency audit |
| prices/midquote_end_5min_stacked.csv | Reconstructed price panel; **do not upload publicly** |

This repo also contains a **nonconfidential aggregate** CSV and JSON to accompany this summary.

## Reproduce the final evaluations on the authorized server

~~~bash
cd ~/ofi-research
source ~/ofi-env/bin/activate
OPENBLAS_NUM_THREADS=1 python scripts/lobster2/audit_research.py --input team_results_regular --output team_results_regular/audit
for metric in ofi_per_depth ofi_per_volume; do
  OPENBLAS_NUM_THREADS=1 python scripts/lobster2/predict_residual_returns.py --input team_results_regular --prices team_results_regular/prices --metric "$metric" --train-days 14 --output "team_results_regular/forward_tests/$metric"
  OPENBLAS_NUM_THREADS=1 python scripts/lobster2/refine_rpca.py --input team_results_regular --metric "$metric" --train-days 14 --output team_results_regular/rpca_tuning
done
~~~

## Next scientific steps

1. Independently compare rebuilt quotes, displayed depths, and end-of-bucket midpoints to external order-book reference snapshots; verify opening-book reconstruction.
2. Extend the date range and run multiple rolling/expanding chronological splits.
3. Quantify uncertainty with suitable day/block clustering or bootstrapping accounting for overlapping horizons and cross-stock dependence.
4. Add explicit market, sector, ETF and other controls, compare residual **returns**, test more meaningful benchmarks, and evaluate RPCA-based forecasts separately only if supported by robust tuning.
5. Report spread, trading costs, executable timing, and sensitivity to reasonable model choices before discussing trading relevance.

**Team-ready one-liner:** We reconstructed 19 regular sessions of five-minute Level 1 OFI and midquotes for 80 equities and completed SVD/RPCA analyses plus out-of-sample 5/15/30-minute return tests; SVD residualization lowered relative prediction error in five of six settings, but none of the fitted models beat the historical-mean benchmark, and quote validation remains pending.

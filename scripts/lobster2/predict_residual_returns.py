#!/usr/bin/env python3
"""Preliminary chronological future-midquote-return test using residual OFI.

Features: five-minute OFI standardized on training days only, residualized
against training-only SVD factors, plus an intraday trailing 3-bucket average.
Targets: reconstructed midquote log return from this bucket END to h future
bucket ends. No cross-session labels, no lookahead or random train/test split.

IMPORTANT: returns are reconstructed from the same message-only data and are
NOT independently validated against external quotes. Forecast evaluation is
exploratory; this does not imply tradeable profits or causal predictive power.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def forward_targets(mid: np.ndarray, daylabels: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    y = np.full_like(mid, np.nan)
    for day in np.unique(daylabels):
        idx = np.flatnonzero(daylabels == day)
        if len(idx) <= horizon:
            continue
        start, end = mid[idx[:-horizon]], mid[idx[horizon:]]
        mask = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0)
        out = np.full_like(start, np.nan)
        np.log(np.divide(end, start, out=np.ones_like(start), where=mask), out=out, where=mask)
        out[~mask] = np.nan
        y[idx[:-horizon]] = out
    return y


def trailing_three(values: np.ndarray, daylabels: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan)
    for day in np.unique(daylabels):
        ix = np.flatnonzero(daylabels == day)
        if len(ix) >= 3:
            trio = (values[ix[:-2]] + values[ix[1:-1]] + values[ix[2:]]) / 3.0
            output[ix[2:]] = trio
    return output


def fit_ridge(features: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if features.ndim != 2 or len(y) != len(features):
        raise ValueError("Feature and target sizes differ")
    A = np.column_stack((np.ones(len(features)), features))
    penalty = np.diag([0] + [alpha] * features.shape[1])
    return np.linalg.solve(A.T @ A + penalty, A.T @ y)


def prepare_factor_model(X: np.ndarray, days: np.ndarray, training_days: list[str],
                         half_life_days: float, threshold: float, fixed_rank: int | None):
    good_rows = np.isfinite(X).all(axis=1)
    train_mask = np.isin(days, training_days) & good_rows
    if train_mask.sum() < 20:
        raise ValueError("Insufficient full-width OFI training rows for SVD")
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0, ddof=1)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid per-stock training normalization")
    standardized = np.full_like(X, np.nan)
    standardized[good_rows] = (X[good_rows] - mean) / std
    weights = np.ones(int(train_mask.sum()))
    if half_life_days > 0:
        last = np.max(pd.to_datetime(days[train_mask]))
        age = np.asarray((last - pd.to_datetime(days[train_mask])).days, dtype=float)
        weights = np.power(0.5, age / half_life_days)
    _, s, vt = np.linalg.svd(standardized[train_mask] * np.sqrt(weights[:, None]),
                             full_matrices=False)
    frac = s ** 2 / np.sum(s ** 2)
    k_80 = int(np.searchsorted(np.cumsum(frac), threshold) + 1)
    rank = fixed_rank if fixed_rank is not None else k_80
    if not 1 <= rank <= len(vt):
        raise ValueError(f"Requested rank {rank} outside 1..{len(vt)}")
    basis = vt[:rank]
    residual = np.full_like(standardized, np.nan)
    z = standardized[good_rows]
    residual[good_rows] = z - (z @ basis.T) @ basis
    return standardized, residual, {"rank": int(rank), "rank_80pct": k_80,
                                    "train_full_width_rows": int(train_mask.sum()),
                                    "explained_training_variance": float(np.cumsum(frac)[rank - 1])}


def evaluate_stock(target: np.ndarray, feature: np.ndarray, trailing: np.ndarray,
                   train_mask: np.ndarray, test_mask: np.ndarray,
                   alpha: float, min_train: int, min_test: int) -> dict | None:
    valid = np.isfinite(target) & np.isfinite(feature) & np.isfinite(trailing)
    tr = train_mask & valid
    te = test_mask & valid
    if tr.sum() < min_train or te.sum() < min_test:
        return None
    train_x = np.column_stack((feature[tr], trailing[tr]))
    test_x = np.column_stack((feature[te], trailing[te]))
    beta = fit_ridge(train_x, target[tr], alpha)
    actual = target[te]
    predicted = beta[0] + test_x @ beta[1:]
    benchmark = np.mean(target[tr])
    baseline_error = float(np.sum((actual - benchmark) ** 2))
    model_error = float(np.sum((actual - predicted) ** 2))
    test_corr = (float(np.corrcoef(actual, predicted)[0, 1])
                 if len(actual) >= 3 and np.std(actual) > 0 and np.std(predicted) > 0
                 else None)
    return {"train_obs": int(tr.sum()), "test_obs": int(te.sum()),
            "test_oos_r2": float(1 - model_error / baseline_error)
            if baseline_error > 0 else None,
            "test_prediction_correlation": test_corr,
            "test_directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "test_model_squared_error": model_error,
            "test_baseline_squared_error": baseline_error,
            "intercept": float(beta[0]), "beta_current": float(beta[1]),
            "beta_trailing3": float(beta[2])}


def run(args):
    panel_dir = Path(args.input)
    quote_dir = Path(args.prices)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.read_csv(panel_dir / f"{args.metric}_5min_stacked.csv")
    quote_frame = pd.read_csv(quote_dir / "midquote_end_5min_stacked.csv")
    if not feature_frame.columns.equals(quote_frame.columns):
        raise ValueError("Ticker order differs between OFI and reconstructed price panels")
    if not feature_frame["minute"].equals(quote_frame["minute"]):
        raise ValueError("OFI / price timestamps are not aligned")
    dates = feature_frame.minute.str.slice(0, 10).to_numpy()
    unique = sorted(np.unique(dates))
    if not 2 <= args.train_days < len(unique):
        raise ValueError("Must reserve at least one later complete day as holdout")
    training_days = unique[:args.train_days]
    test_days = unique[args.train_days:]
    train_mask = np.isin(dates, training_days)
    test_mask = np.isin(dates, test_days)
    names = feature_frame.columns[1:].tolist()
    X = feature_frame[names].to_numpy(dtype=float)
    mid = quote_frame[names].to_numpy(dtype=float)
    if (np.isfinite(mid) & (mid <= 0)).any():
        raise ValueError("Nonpositive midquote, cannot compute log returns")
    z, residual, svd = prepare_factor_model(
        X, dates, training_days, args.half_life_days,
        args.variance_target, args.rank)
    report = {"metric": args.metric, "train_days": training_days,
              "test_days": test_days, "svd": svd, "ridge_alpha": args.ridge_alpha,
              "horizons_in_5min_buckets": args.horizons,
              "data_status": "PROVISIONAL_RECONSTRUCTED_MIDQUOTES",
              "caution": "No independent quote validation or transaction-cost assessment; "
                         "predictive performance is an out-of-sample statistical exercise."}
    rows = []
    for h in args.horizons:
        target = forward_targets(mid, dates, h)
        for feature_name, panel in (("raw_standardized", z), ("svd_residual", residual)):
            trail = trailing_three(panel, dates)
            for ti, ticker in enumerate(names):
                scores = evaluate_stock(target[:, ti], panel[:, ti], trail[:, ti],
                                        train_mask, test_mask,
                                        args.ridge_alpha, args.min_train,
                                        args.min_test)
                if scores is not None:
                    rows.append({"ticker": ticker, "horizon_buckets": h,
                                 "model": feature_name, **scores})
    details = pd.DataFrame(rows)
    if details.empty:
        raise RuntimeError("No tickers passed minimum sample counts; check reconstructed price coverage")
    details.to_csv(out / "per_stock_oos_results.csv", index=False)
    group = []
    for (h, model), g in details.groupby(["horizon_buckets", "model"]):
        finite = pd.to_numeric(g["test_oos_r2"], errors="coerce").dropna()
        total_baseline = float(g.test_baseline_squared_error.sum())
        total_model = float(g.test_model_squared_error.sum())
        group.append({
            "horizon_buckets": int(h), "minutes_ahead": int(5 * h), "model": model,
            "tickers_evaluated": int(g.ticker.nunique()),
            "test_total_predictions": int(g.test_obs.sum()),
            "median_per_stock_oos_r2": float(finite.median()) if len(finite) else None,
            "positive_oos_r2_fraction": float((finite > 0).mean()) if len(finite) else None,
            "pooled_test_oos_r2": float(1 - total_model / total_baseline)
            if total_baseline > 0 else None,
            "median_per_stock_prediction_corr": float(g.test_prediction_correlation.median()),
        })
    pd.DataFrame(group).to_csv(out / "summary_oos_results.csv", index=False)
    report["summary"] = group
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(group).to_string(index=False))
    print("Saved provisional future-return evaluation under:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="team_results_long")
    p.add_argument("--prices", default="team_results_long/prices")
    p.add_argument("--output", default="team_results_long/forward_tests")
    p.add_argument("--metric", choices=("ofi_per_depth", "ofi_per_volume"), default="ofi_per_depth")
    p.add_argument("--train-days", type=int, default=15)
    p.add_argument("--rank", type=int)
    p.add_argument("--variance-target", type=float, default=0.8)
    p.add_argument("--half-life-days", type=float, default=5)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6])
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument("--min-test", type=int, default=30)
    a = p.parse_args()
    if any(h < 1 for h in a.horizons) or a.ridge_alpha < 0:
        p.error("All horizons must be positive, ridge-alpha nonnegative")
    run(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit existing 3-day OFI outputs without decompressing any market-data archives.

Run on the authorized lab host:
  python scripts/lobster2/audit_research.py --input team_results
Outputs are derived from licensed data and should remain private by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("ofi_raw", "ofi_per_volume", "ofi_per_depth", "executed_volume", "mean_depth")


def read_panel(folder: Path, metric: str) -> pd.DataFrame:
    path = folder / f"{metric}_5min_stacked.csv"
    frame = pd.read_csv(path)
    if frame.columns[0] != "minute":
        raise ValueError(f"{path} lacks the required first 'minute' column")
    return frame


def check_ratio(raw: np.ndarray, denom: np.ndarray, observed: np.ndarray) -> dict:
    finite = np.isfinite(raw) & np.isfinite(denom) & (denom > 0)
    expected = np.full(raw.shape, np.nan)
    np.divide(raw, denom, out=expected, where=finite)
    mismatch = np.isfinite(expected) != np.isfinite(observed)
    both = np.isfinite(expected) & np.isfinite(observed)
    mismatch[both] |= ~np.isclose(expected[both], observed[both], rtol=1e-5, atol=1e-9)
    return {
        "positive_denominators": int(finite.sum()),
        "zero_denominators": int(np.count_nonzero(np.isfinite(denom) & (denom == 0))),
        "ratio_mismatches": int(mismatch.sum()),
    }


def persistence_summary(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    if not {"ticker", "lookback_buckets", "lag_corr", "past_sum_to_next_corr"}.issubset(frame):
        raise ValueError(f"Unexpected residual persistence columns: {path}")
    summaries = []
    for lag, g in frame.groupby("lookback_buckets"):
        row = {"lookback_buckets": int(lag), "tickers": int(g.ticker.nunique())}
        for column in ("lag_corr", "past_sum_to_next_corr"):
            v = pd.to_numeric(g[column], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            row[column + "_median"] = float(np.median(v)) if len(v) else None
            row[column + "_mean"] = float(np.mean(v)) if len(v) else None
            row[column + "_positive_fraction"] = float(np.mean(v > 0)) if len(v) else None
            row[column + "_nonmissing_tickers"] = len(v)
        summaries.append(row)
    return summaries


def audit(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    names = list(manifest["tickers"])
    days = list(manifest["days"])
    frames = {m: read_panel(folder, m) for m in METRICS}
    template = frames["ofi_raw"]
    stamps = pd.to_datetime(template["minute"], errors="raise")
    errors = []
    expected_rows = 78 * len(days)
    if len(template) != expected_rows:
        errors.append(f"Expected {expected_rows} five-minute rows; got {len(template)}")
    if len(names) != 80:
        errors.append(f"Expected 80 tickers; got {len(names)}")
    if template.columns[1:].tolist() != names:
        errors.append("Columns do not exactly match ticker order in manifest")
    if stamps.duplicated().any():
        errors.append("Duplicate timestamps detected")
    for day in days:
        observed = stamps.dt.strftime("%Y%m%d").eq(day)
        wanted = pd.date_range(pd.to_datetime(day) + pd.Timedelta(hours=9, minutes=30),
                               periods=78, freq="5min")
        if not stamps[observed].reset_index(drop=True).equals(pd.Series(wanted)):
            errors.append(f"Expected five-minute regular-session timestamps not found on {day}")
    for metric, frame in frames.items():
        if not frame.columns.equals(template.columns):
            errors.append(f"{metric} does not match the OFI ticker-column schema")
        if not frame.minute.equals(template.minute):
            errors.append(f"{metric} timestamps are not aligned")

    matrices = {m: f.iloc[:, 1:].to_numpy(dtype=float) for m, f in frames.items()}
    missing = {m: int(np.count_nonzero(~np.isfinite(a))) for m, a in matrices.items()}
    volume_check = check_ratio(matrices["ofi_raw"], matrices["executed_volume"],
                               matrices["ofi_per_volume"])
    depth_check = check_ratio(matrices["ofi_raw"], matrices["mean_depth"],
                              matrices["ofi_per_depth"])
    if volume_check["ratio_mismatches"]:
        errors.append(f"Volume normalization mismatches: {volume_check['ratio_mismatches']}")
    if depth_check["ratio_mismatches"]:
        errors.append(f"Depth normalization mismatches: {depth_check['ratio_mismatches']}")

    qc = pd.read_csv(folder / "reconstruction_qc.csv")
    status_counts = {str(k): int(v) for k, v in qc.status.value_counts().items()}
    if len(qc) != len(days) * len(names):
        errors.append(f"QC has {len(qc)} rows, not {len(days) * len(names)}")
    if status_counts.get("processed_provisional", 0) != len(days) * len(names):
        errors.append("Not every stock-day was processed successfully")
    warning_counts = {}
    for col in ("bad_lines", "nonmonotonic", "invalid_quotes", "zero_volume_buckets"):
        if col in qc:
            warning_counts[col] = int(pd.to_numeric(qc[col], errors="coerce").fillna(0).sum())
    if "book_warnings" in qc:
        for entry in qc.book_warnings.dropna():
            try:
                values = json.loads(entry)
            except (TypeError, ValueError):
                errors.append("Unreadable JSON in QC book_warnings")
                continue
            for key, value in values.items():
                warning_counts[key] = warning_counts.get(key, 0) + int(value)

    factor_results = {}
    for label in ("depth", "volume"):
        d = folder / f"factors_{label}"
        if not d.exists():
            continue
        summary = json.loads((d / "factor_summary.json").read_text())
        standardized = pd.read_csv(d / "heldout_standardized.csv").iloc[:, 1:].to_numpy(dtype=float)
        svd_low = pd.read_csv(d / "svd_lowrank.csv").iloc[:, 1:].to_numpy(dtype=float)
        svd_res = pd.read_csv(d / "svd_residual.csv").iloc[:, 1:].to_numpy(dtype=float)
        robust_parts = [
            pd.read_csv(d / f"rpca_{name}.csv").iloc[:, 1:].to_numpy(dtype=float)
            for name in ("lowrank", "sparse", "error")
        ]
        reconstruction_error = float(np.max(np.abs(standardized - sum(robust_parts))))
        svd_identity_error = float(np.max(np.abs(standardized - svd_low - svd_res)))
        denom = float(np.sum(standardized**2))
        fraction = float(1 - np.sum(svd_res**2) / denom) if denom else None
        sparse = robust_parts[1]
        factor_results[label] = {
            "rank_used": int(summary["rank_used"]),
            "rank_80pct": int(summary["rank_80pct"]),
            "train_sparse_fraction": float(summary["rpca_sparse_fraction"]),
            "heldout_sparse_nonzero_fraction": float(np.mean(sparse != 0)),
            "heldout_svd_energy_fraction": fraction,
            "svd_identity_max_absolute_error": svd_identity_error,
            "rpca_identity_max_absolute_error": reconstruction_error,
            "residual_persistence": persistence_summary(d / "heldout_residual_persistence.csv"),
        }
        if reconstruction_error > 1e-8 or svd_identity_error > 1e-8:
            errors.append(f"{label}: additive decomposition fails numerical identity check")

    return {
        "status": "PASS_WITH_PROVISIONAL_QUOTES" if not errors else "CHECK_REQUIRED",
        "days": days,
        "ticker_count": len(names),
        "shape": [len(days), 78, len(names)],
        "panel_missing_values": missing,
        "volume_ratio_check": volume_check,
        "depth_ratio_check": depth_check,
        "stock_day_status": status_counts,
        "reconstruction_warning_counts": warning_counts,
        "factors": factor_results,
        "errors": errors,
        "limitations": [
            "This checks consistency, not correctness against independent quote snapshots.",
            "Three days and a single held-out day cannot establish stable factors.",
            "Residual correlations describe OFI persistence, not future-return prediction.",
            "No statistical significance is asserted; cross-stock dependencies and serial correlation remain.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("team_results"))
    p.add_argument("--output", type=Path, default=Path("team_results/audit"))
    args = p.parse_args()
    report = audit(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# OFI three-day data and residual audit",
        "",
        f"**Automated consistency result:** {report['status']}",
        f"**Data:** {report['ticker_count']} tickers, {len(report['days'])} days",
        "",
        "## Missing values",
        *[f"- {k}: {v}" for k, v in report["panel_missing_values"].items()],
        "",
        "## Normalization consistency",
        f"- Volume ratio mismatches: {report['volume_ratio_check']['ratio_mismatches']}",
        f"- Depth ratio mismatches: {report['depth_ratio_check']['ratio_mismatches']}",
        "",
        "## Reconstruction warning counts",
        *[f"- {k}: {v}" for k, v in sorted(report["reconstruction_warning_counts"].items())],
        "",
        "## Held-out factor and residual diagnostics",
    ]
    for metric, info in report["factors"].items():
        lines += [
            f"### {metric.title()} normalization",
            f"- Chosen SVD rank: {info['rank_used']}",
            f"- Held-out fraction of standardized OFI squared norm captured by SVD projection: "
            f"{info['heldout_svd_energy_fraction']:.3f}",
            f"- Training sparse nonzero fraction: {info['train_sparse_fraction']:.3f}",
            f"- Held-out sparse nonzero fraction: {info['heldout_sparse_nonzero_fraction']:.3f}",
        ]
        for entry in info["residual_persistence"]:
            lines.append(f"- Lag {entry['lookback_buckets']}: median residual autocorrelation "
                         f"{entry['lag_corr_median']}; median past-sum-to-next correlation "
                         f"{entry['past_sum_to_next_corr_median']}")
    lines += ["", "## Consistency errors", *([f"- {e}" for e in report["errors"]] or ["- None"]),
              "", "## Interpretation limits", *[f"- {x}" for x in report["limitations"]]]
    (args.output / "audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "factors": report["factors"]}, indent=2))
    print("Saved private diagnostic reports under", args.output)


if __name__ == "__main__":
    main()

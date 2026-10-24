from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microstructure import (  # noqa: E402
    align_integrated_ofi,
    align_returns,
    bucket_features,
    common_ofi_factor,
    cross_impact_matrix,
    discover_days,
    extract_7z,
    integrated_ofi,
    read_lobster_day,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OFI factor model and cross-impact matrices from LOBSTER data.")
    p.add_argument("archives", nargs="+", help="LOBSTER .7z archives or extracted directories")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--output", default=str(ROOT / "outputs"))
    p.add_argument("--cache", default=str(ROOT / "data" / "extracted"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    cache = Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    all_days = []
    for item in args.archives:
        path = Path(item).expanduser().resolve()
        if path.suffix.lower() == ".7z":
            extracted = extract_7z(path, cache / path.stem)
            all_days.extend(discover_days(extracted))
        elif path.is_dir():
            all_days.extend(discover_days(path))
        else:
            raise ValueError(f"Expected a .7z archive or directory, got: {path}")

    if not all_days:
        raise RuntimeError("No LOBSTER message/orderbook pairs were found.")

    per_ticker_days: dict[str, list[pd.DataFrame]] = {}
    for day in all_days:
        raw = read_lobster_day(day, levels=args.levels)
        buckets = bucket_features(raw, levels=args.levels, bucket_seconds=args.bucket_seconds)
        buckets["ticker"] = day.ticker
        buckets["date"] = day.date
        per_ticker_days.setdefault(day.ticker, []).append(buckets)

    ticker_frames: dict[str, pd.DataFrame] = {}
    weight_rows = []
    summary_rows = []
    for ticker, days in per_ticker_days.items():
        combined = pd.concat(days, ignore_index=True).sort_values("timestamp")
        integ = integrated_ofi(combined, levels=args.levels)
        ticker_frames[ticker] = integ.frame
        for feature, weight in integ.weights.items():
            weight_rows.append({"ticker": ticker, "feature": feature, "pc1_weight": weight})
        summary_rows.append(
            {
                "ticker": ticker,
                "buckets": len(integ.frame),
                "pc1_explained_variance": integ.explained_variance_ratio,
            }
        )
        integ.frame.to_csv(output / f"{ticker}_bucket_features.csv", index=False)

    pd.DataFrame(weight_rows).to_csv(output / "within_stock_pc1_weights.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "within_stock_pca_summary.csv", index=False)

    wide = align_integrated_ofi(ticker_frames)
    if wide.shape[1] < 2 or len(wide) < 3:
        print("Processed single-stock features, but not enough synchronized assets for factor matrices.")
        return

    wide.to_csv(output / "integrated_ofi_panel.csv")
    wide.corr().to_csv(output / "integrated_ofi_correlation.csv")
    wide.cov().to_csv(output / "integrated_ofi_covariance.csv")

    common = common_ofi_factor(wide)
    common.factor.to_csv(output / "common_ofi_factor.csv", header=True)
    common.loadings.to_csv(output / "factor_loadings.csv")
    common.residuals.to_csv(output / "idiosyncratic_ofi.csv")
    common.residuals.corr().to_csv(output / "idiosyncratic_ofi_correlation.csv")
    common.residuals.cov().to_csv(output / "idiosyncratic_ofi_covariance.csv")

    returns = align_returns(ticker_frames, wide.index)
    shared = returns.index.intersection(common.factor.index)
    if len(shared) >= 3:
        beta = cross_impact_matrix(
            returns.loc[shared],
            common.factor.loc[shared],
            common.residuals.loc[shared],
            use_lasso=True,
        )
        beta.to_csv(output / "cross_impact_coefficient_matrix.csv")

    (output / "run_summary.txt").write_text(
        "\n".join(
            [
                f"tickers={','.join(sorted(ticker_frames))}",
                f"levels={args.levels}",
                f"bucket_seconds={args.bucket_seconds}",
                f"synchronized_buckets={len(wide)}",
                f"common_factor_explained_variance={common.explained_variance_ratio:.6f}",
            ]
        )
        + "\n"
    )

    print(f"Done. Wrote outputs to {output}")


if __name__ == "__main__":
    main()

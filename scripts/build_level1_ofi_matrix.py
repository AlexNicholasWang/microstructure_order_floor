"""Construct Level 1 OFI time x asset matrix from LOBSTER archives.

Rows are aligned intraday 10-second timestamps. Columns are ticker symbols.
OFI is summed event-level bid order flow minus ask order flow, in shares.
No artificial values are inserted for missing tickers or timestamps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from microstructure import discover_days, extract_7z, read_lobster_day, bucket_features


def build_matrix(days, bucket_seconds=10, join="inner"):
    """Construct n x p panel from matching LOBSTER message/orderbook pairs."""
    by_ticker = {}
    for day in days:
        # Process each day separately so no overnight OFI differences are introduced.
        events = read_lobster_day(day, levels=1)
        buckets = bucket_features(events, levels=1, bucket_seconds=bucket_seconds)
        series = buckets.set_index("timestamp")["ofi_raw_1"]
        by_ticker.setdefault(day.ticker, []).append(series)

    if not by_ticker:
        raise ValueError("No matched LOBSTER files found")

    columns = {}
    for ticker, pieces in sorted(by_ticker.items()):
        series = pd.concat(pieces).sort_index()
        if series.index.has_duplicates:
            raise ValueError(f"Repeated timestamps for {ticker}; remove overlapping input archives")
        columns[ticker] = series

    matrix = pd.concat(columns, axis=1, join=join).sort_index()
    if join == "inner":
        matrix = matrix.dropna()
    matrix.index.name = "timestamp"
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="LOBSTER .7z archives or extracted CSV directories")
    p.add_argument("--expected-assets", type=int, default=80)
    p.add_argument("--tickers-file", type=Path, help="One ticker per line; all 80 names, if known")
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--join", choices=["inner", "outer"], default="inner")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "level1")
    p.add_argument("--cache", type=Path, default=ROOT / "data" / "extracted")
    args = p.parse_args()

    if args.bucket_seconds < 1 or 86400 % args.bucket_seconds:
        p.error("bucket-seconds must be a positive divisor of 86400")

    days = []
    for source in args.inputs:
        source = Path(source)
        if source.suffix.lower() == ".7z":
            root = extract_7z(source, args.cache / source.stem)
        elif source.is_dir():
            root = source
        else:
            p.error(f"Expected .7z or directory: {source}")
        days.extend(discover_days(root))

    matrix = build_matrix(days, bucket_seconds=args.bucket_seconds, join=args.join)
    observed = list(matrix.columns)
    missing = []
    if args.tickers_file:
        requested = [
            line.strip().upper()
            for line in args.tickers_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(set(requested)) != len(requested):
            p.error("tickers-file contains duplicate names")
        if len(requested) != args.expected_assets:
            p.error(f"Expected {args.expected_assets} ticker names, received {len(requested)}")
        unexpected = sorted(set(observed) - set(requested))
        if unexpected:
            p.error(f"Input files include tickers not in list: {unexpected}")
        missing = sorted(set(requested) - set(observed))
        matrix = matrix.reindex(columns=requested)

    args.output.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output / "level1_ofi_matrix.csv")
    pd.DataFrame({
        "ticker": matrix.columns,
        "nonmissing_buckets": matrix.notna().sum().to_numpy(),
    }).to_csv(args.output / "coverage.csv", index=False)
    (args.output / "summary.txt").write_text(
        f"shape={matrix.shape[0]} x {matrix.shape[1]}\n"
        f"expected_assets={args.expected_assets}\n"
        f"observed_tickers={','.join(observed)}\n"
        f"missing_tickers={','.join(missing)}\n"
        f"bucket_seconds={args.bucket_seconds}\n"
        f"join={args.join}\n"
        "values=raw Level 1 OFI in shares; bid event flow minus ask event flow\n"
        "missing values are NaN, never made up\n"
    )
    print(f"Saved Level 1 OFI matrix: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"Observed assets: {len(observed)} / expected {args.expected_assets}")
    if missing:
        print(f"Missing {len(missing)} named assets; left as NaN in output")


if __name__ == "__main__":
    main()

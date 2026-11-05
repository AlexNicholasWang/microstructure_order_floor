#!/usr/bin/env python3
"""Compare reconstructed end-of-bucket Level 1 quotes to independent reference quotes.

Reference CSV must contain (minute, ticker, bid, ask, bid_size, ask_size),
where minute labels the 5-minute bucket START and quote values are at bucket END.
Reference must come from independent exchange/order-book snapshots, NOT this
message-only reconstructor. Price scale may require explicit adjustment.

Run on an authorized host; never commit underlying reference/derived price data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("bid", "ask", "bid_size", "ask_size")
KEYS = ("minute", "ticker")


def validate(reconstructed: pd.DataFrame, reference: pd.DataFrame,
             price_multiplier: float, max_price_diff: float,
             max_size_diff: float) -> tuple[dict, pd.DataFrame]:
    for label, frame in (("reconstructed", reconstructed), ("reference", reference)):
        absent = set(KEYS + FIELDS) - set(frame.columns)
        if absent:
            raise ValueError(f"{label} missing columns: {sorted(absent)}")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"{label} has duplicate (minute,ticker) keys")
    lhs = reconstructed[list(KEYS + FIELDS)].copy()
    rhs = reference[list(KEYS + FIELDS)].copy()
    for field in FIELDS:
        lhs[field] = pd.to_numeric(lhs[field], errors="coerce")
        rhs[field] = pd.to_numeric(rhs[field], errors="coerce")
    rhs[["bid", "ask"]] *= price_multiplier
    joined = lhs.merge(rhs, on=list(KEYS), how="inner", suffixes=("_reconstructed", "_reference"))
    if joined.empty:
        raise ValueError("No identical timestamp/ticker keys between reference and reconstruction")
    valid = np.ones(len(joined), dtype=bool)
    diffs = {}
    for field in FIELDS:
        x = joined[field + "_reconstructed"].to_numpy(dtype=float)
        y = joined[field + "_reference"].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        valid &= good
        tolerance = max_price_diff if field in ("bid", "ask") else max_size_diff
        differences = np.abs(x - y)
        diffs[field] = {"compared": int(good.sum()),
                        "match_fraction": float(np.mean(differences[good] <= tolerance))
                        if good.any() else None,
                        "median_absolute_difference": float(np.median(differences[good]))
                        if good.any() else None}
        joined[field + "_abs_diff"] = differences
    all_match = valid.copy()
    for field in FIELDS:
        tolerance = max_price_diff if field in ("bid", "ask") else max_size_diff
        all_match &= joined[field + "_abs_diff"].to_numpy() <= tolerance
    joined["all_fields_match"] = all_match
    stats = {"reference_rows": len(reference), "reconstructed_rows": len(reconstructed),
             "matching_keys": len(joined), "fully_valid_compared": int(valid.sum()),
             "all_four_fields_match_fraction_among_valid":
                 float(np.mean(all_match[valid])) if valid.any() else None,
             "fields": diffs,
             "status": "REFERENCE_COMPARISON_COMPLETED",
             "warning": "Snapshot coverage and source independence must be documented; "
                        "matches do not validate other dates or event times."}
    return stats, joined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reconstructed", default="team_results_long/prices/end_of_bucket_quotes.csv")
    p.add_argument("--reference", required=True)
    p.add_argument("--output", default="team_results_long/quote_validation")
    p.add_argument("--reference-price-multiplier", type=float, default=1.0)
    p.add_argument("--price-tolerance", type=float, default=0.0)
    p.add_argument("--size-tolerance", type=float, default=0.0)
    args = p.parse_args()
    if args.reference_price_multiplier <= 0 or args.price_tolerance < 0 or args.size_tolerance < 0:
        p.error("Multiplier must be positive; tolerances must be nonnegative")
    stats, rows = validate(pd.read_csv(args.reconstructed), pd.read_csv(args.reference),
                           args.reference_price_multiplier, args.price_tolerance,
                           args.size_tolerance)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    rows.to_csv(out / "matched_quote_differences.csv", index=False)
    print(json.dumps(stats, indent=2))
    print("Saved private comparison:", out)


if __name__ == "__main__":
    main()

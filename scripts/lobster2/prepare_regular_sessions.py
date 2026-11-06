#!/usr/bin/env python3
"""Make a regular-session-only five-minute OFI panel from a cached panel.

By default excludes 2025-07-03, an early-close (13:00 ET) NYSE session,
because the original reconstruction always assumed 09:30-16:00 ET.

No LOBSTER reprocessing or licensed-data upload is needed. Every derivative
is written to --output; the input directory is left unchanged.

Usage:
  python scripts/lobster2/prepare_regular_sessions.py \
    --input team_results_long --output team_results_regular --exclude 20250703
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ofi_team_pipeline import METRICS, NBUCKET


def prepare(input_dir: Path, output_dir: Path, exclusions: list[str],
            half_life: float = 5.0) -> dict:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    manifest = json.loads((input_dir / "manifest.json").read_text())
    days = list(manifest["days"])
    tickers = list(manifest["tickers"])
    excluded = set(exclusions)
    unknown = excluded - set(days)
    if unknown:
        raise ValueError(f"Exclusion days are not in manifest: {sorted(unknown)}")
    kept = [day for day in days if day not in excluded]
    if len(kept) < 3:
        raise ValueError("Must retain at least three complete trading days")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output paths must differ")
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = {}
    matrices: dict[str, np.ndarray] = {}
    original = None
    for name in METRICS:
        df = pd.read_csv(input_dir / f"{name}_5min_stacked.csv")
        if df.columns.tolist() != ["minute"] + tickers:
            raise ValueError(f"{name}: ticker columns differ from manifest")
        if original is None:
            original = df["minute"]
        elif not df["minute"].equals(original):
            raise ValueError(f"{name}: timestamp sequence differs")
        dates = df.minute.astype(str).str[:10].str.replace("-", "", regex=False)
        counts = dates.value_counts()
        for day in days:
            if counts.get(day, 0) != NBUCKET:
                raise ValueError(f"{name}: expected {NBUCKET} buckets for {day}")
        if len(df) != len(days) * NBUCKET:
            raise ValueError(f"{name}: unexpected extra rows")
        filtered = df.loc[dates.isin(kept)].reset_index(drop=True)
        if filtered.empty:
            raise ValueError(f"{name}: nothing left after exclusions")
        values = filtered[tickers].to_numpy(dtype=float)
        matrices[name] = values.reshape(len(kept), NBUCKET, len(tickers))
        filtered.to_csv(output_dir / f"{name}_5min_stacked.csv", index=False)
        np.save(output_dir / f"{name}_tensor.npy", matrices[name])
        shapes[name] = list(matrices[name].shape)

    qc = pd.read_csv(input_dir / "reconstruction_qc.csv")
    if not {"day", "ticker", "status"}.issubset(qc.columns):
        raise ValueError("QC table lacks day, ticker or status")
    qc_dates = qc.day.astype(str).str.replace("-", "", regex=False)
    qc_filtered = qc.loc[qc_dates.isin(kept)].copy()
    qc_filtered.to_csv(output_dir / "reconstruction_qc.csv", index=False)
    if len(qc_filtered) != len(kept) * len(tickers):
        raise ValueError("Unexpected number of retained stock-day QC records")

    hist = np.full_like(matrices["ofi_raw"], np.nan)
    for d in range(1, len(kept)):
        old = matrices["ofi_raw"][:d]
        # Same prior-trading-session weighting convention as ofi_team_pipeline.py.
        age = np.arange(d, 0, -1)
        w = 2 ** (-age / half_life)
        valid = np.isfinite(old)
        num = (np.where(valid, old, 0.0) * w[:, None, None]).sum(axis=0)
        den = (valid * w[:, None, None]).sum(axis=0)
        hist[d] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    history = pd.DataFrame(hist.reshape(-1, len(tickers)), columns=tickers)
    filtered_times = pd.read_csv(output_dir / "ofi_raw_5min_stacked.csv",
                                 usecols=["minute"]).minute
    history.insert(0, "minute", filtered_times)
    history.to_csv(output_dir / "historical_weighted_raw_ofi.csv", index=False)

    result = dict(manifest)
    result.update({
        "days": kept,
        "excluded_days": sorted(excluded),
        "source_panel": str(input_dir),
        "half_life_trading_sessions": half_life,
        "regular_sessions_only": True,
        "notice": "2025-07-03 removed in full because the original 78-bucket "
                  "message-only reconstruction did not model its 13:00 ET early close. "
                  "Independent quote validation remains outstanding.",
    })
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Created {len(kept)}-day regular-session OFI panel at {output_dir}")
    print(f"Excluded dates: {sorted(excluded)}")
    print(f"Tensor shape: {matrices['ofi_raw'].shape}")
    return {"kept_days": kept, "excluded_days": sorted(excluded), "shapes": shapes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("team_results_long"))
    parser.add_argument("--output", type=Path, default=Path("team_results_regular"))
    parser.add_argument("--exclude", nargs="+", default=["20250703"])
    parser.add_argument("--half-life-trading-sessions", type=float, default=5.0)
    args = parser.parse_args()
    prepare(args.input, args.output, args.exclude, args.half_life_trading_sessions)


if __name__ == "__main__":
    main()

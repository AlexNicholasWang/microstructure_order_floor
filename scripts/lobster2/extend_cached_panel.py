#!/usr/bin/env python3
"""Extend cached 5-minute OFI panels to additional complete trading days.

Reuses processed stock-days from --base and incremental .npz cache. It never
overwrites the existing --base directory. Raw/derived outputs stay on lobster2.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ofi_team_pipeline import METRICS, NBUCKET, OPEN, one_archive


def available_days(root: Path, tickers: list[str], until: str) -> list[str]:
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or len(path.name) != 8 or not path.name.isdigit():
            continue
        if path.name > until:
            continue
        if all((path / (ticker + ".7z")).is_file() for ticker in tickers):
            out.append(path.name)
    return out


def get_prior(folder: Path, days: list[str], tickers: list[str]) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["tickers"] != tickers:
        raise ValueError("Base-panel ticker list/order differs from requested universe")
    base_days = list(manifest["days"])
    prior = {"days": base_days}
    for name in METRICS:
        f = pd.read_csv(folder / f"{name}_5min_stacked.csv")
        if f.columns[1:].tolist() != tickers:
            raise ValueError(f"Base-panel ticker mismatch for {name}")
        prior[name] = f[tickers].to_numpy(dtype=float).reshape(len(base_days), NBUCKET, len(tickers))
    prior["qc"] = pd.read_csv(folder / "reconstruction_qc.csv")
    return prior


def build(args):
    base = Path(args.base)
    root = Path(args.data_root)
    dest = Path(args.output)
    prior_manifest = json.loads((base / "manifest.json").read_text())
    tickers = prior_manifest["tickers"]
    previous_days = prior_manifest["days"]
    candidates = available_days(root, tickers, args.until)
    chosen = candidates[-args.max_days:]
    if not set(previous_days).issubset(chosen):
        raise ValueError("Selected days do not include all cached base days. "
                         "Increase --max-days or adjust --until to include the original panel.")
    if len(chosen) < 4:
        raise ValueError(f"Only {len(chosen)} complete days available; request more.")
    prior = get_prior(base, chosen, tickers)
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir = dest / ".stock_day_cache"
    cache_dir.mkdir(exist_ok=True)
    print(f"Selected {len(chosen)} complete dates, {len(tickers)} tickers; "
          f"reusing {len(previous_days)} previously processed trading days.", flush=True)
    print("Dates:", " ".join(chosen), flush=True)
    if args.dry_run:
        return

    arrays = {name: np.full((len(chosen), NBUCKET, len(tickers)), np.nan)
              for name in METRICS}
    qcrows = []
    prior_qc = prior["qc"]
    for di, day in enumerate(chosen):
        for ti, ticker in enumerate(tickers):
            matches = prior_qc[(prior_qc.day.astype(str) == day)
                               & (prior_qc.ticker.astype(str) == ticker)]
            if day in previous_days:
                old_index = previous_days.index(day)
                for name in METRICS:
                    arrays[name][di, :, ti] = prior[name][old_index, :, ti]
                if not matches.empty:
                    qcrows.append(matches.iloc[0].to_dict())
                else:
                    qcrows.append({"day": day, "ticker": ticker,
                                   "status": "reused_base_missing_qc"})
                continue
            cached = cache_dir / f"{day}_{ticker}.npz"
            if cached.exists():
                try:
                    with np.load(cached) as obj:
                        data = {name: obj[name] for name in METRICS}
                        stat = json.loads(str(obj["qc_json"].item()))
                    if not all(np.shape(data[name]) == (NBUCKET,) for name in METRICS):
                        raise ValueError("Invalid cache shape")
                except Exception as exc:
                    print(f"Cache invalid for {day} {ticker}: {exc}. Recomputing.", flush=True)
                    cached.unlink()
                else:
                    for name in METRICS:
                        arrays[name][di, :, ti] = data[name]
                    qcrows.append(stat)
                    continue
            print(f"[{di + 1}/{len(chosen)} day, {ti + 1}/{len(tickers)} stock] "
                  f"{day} {ticker}", flush=True)
            try:
                data, stats = one_archive(root / day / f"{ticker}.7z")
                stat = {"day": day, "ticker": ticker, "status": "processed_provisional",
                        **{k: json.dumps(v) if isinstance(v, dict) else v
                           for k, v in stats.items()}}
                for name in METRICS:
                    arrays[name][di, :, ti] = data[name]
                # Atomic cache save so interrupting the process does not corrupt it.
                tmp = cached.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **data, qc_json=json.dumps(stat))
                tmp.replace(cached)
                qcrows.append(stat)
            except Exception as exc:
                print(f"ERROR: {day} {ticker}: {exc}", flush=True)
                qcrows.append({"day": day, "ticker": ticker,
                               "status": "failed", "error": str(exc)})
    clocks = []
    for day in chosen:
        date = datetime.strptime(day, "%Y%m%d")
        clocks.extend([(date + timedelta(seconds=OPEN + NBUCKET * 0 + i * 300))
                       .strftime("%Y-%m-%d %H:%M") for i in range(NBUCKET)])
    for name, array in arrays.items():
        wide = pd.DataFrame(array.reshape(-1, len(tickers)), columns=tickers)
        wide.insert(0, "minute", clocks)
        wide.to_csv(dest / f"{name}_5min_stacked.csv", index=False)
        np.save(dest / f"{name}_tensor.npy", array)

    history = np.full_like(arrays["ofi_raw"], np.nan)
    for di in range(1, len(chosen)):
        old = arrays["ofi_raw"][:di]
        ages = np.arange(di, 0, -1)
        w = 2 ** (-ages / args.half_life_days)
        good = np.isfinite(old)
        top = (np.where(good, old, 0) * w[:, None, None]).sum(axis=0)
        bottom = (good * w[:, None, None]).sum(axis=0)
        history[di] = np.divide(top, bottom, out=np.full_like(top, np.nan),
                                where=bottom > 0)
    hist = pd.DataFrame(history.reshape(-1, len(tickers)), columns=tickers)
    hist.insert(0, "minute", clocks)
    hist.to_csv(dest / "historical_weighted_raw_ofi.csv", index=False)
    pd.DataFrame(qcrows).to_csv(dest / "reconstruction_qc.csv", index=False)
    manifest = dict(prior_manifest)
    manifest.update({"days": chosen, "tickers": tickers,
                     "source_base": str(base),
                     "incremental_cache": ".stock_day_cache",
                     "caution": "provisional message-only book; no independent snapshot verification"})
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Extended panels and tensors saved:", dest,
          "shape:", arrays["ofi_raw"].shape, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="team_results")
    p.add_argument("--output", default="team_results_long")
    p.add_argument("--data-root", default="/lobster/2025")
    p.add_argument("--until", default="20250731")
    p.add_argument("--max-days", type=int, default=20)
    p.add_argument("--half-life-days", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.max_days < 4 or a.half_life_days <= 0:
        p.error("--max-days >= 4 and --half-life-days > 0 are required")
    build(a)


if __name__ == "__main__":
    main()

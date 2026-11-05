#!/usr/bin/env python3
"""Cache provisional five-minute end-of-bucket best-quote prices for forward returns.

This uses message-only reconstruction and cannot independently validate it.
Each completed ticker-day is cached to make interrupted runs resumable.
Keep resulting prices and quotes on the authorized lab host.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruct_ofi import Book, OPEN
from ofi_team_pipeline import NBUCKET

FIELDS = ("mid", "bid", "ask", "bid_size", "ask_size")


def archive_quotes(archive: Path) -> tuple[dict, dict]:
    book = Book()
    values = {key: np.full(NBUCKET, np.nan) for key in FIELDS}
    stats = {"message_rows": 0, "parse_failures": 0, "nonmonotonic": 0}
    prev_time = -math.inf
    next_index = 0

    def capture(index):
        quote = book.quote()
        if quote is None:
            return
        (bid, bsize), (ask, asize) = quote
        values["mid"][index] = (bid + ask) / 2.0
        values["bid"][index] = bid
        values["ask"][index] = ask
        values["bid_size"][index] = bsize
        values["ask_size"][index] = asize

    process = subprocess.Popen(["7z", "x", "-so", str(archive)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, bufsize=1)
    try:
        for row in csv.reader(process.stdout):
            stats["message_rows"] += 1
            try:
                t = float(row[0])
                event, oid, qty, price, side = map(int, row[1:6])
            except (ValueError, IndexError):
                stats["parse_failures"] += 1
                continue
            if not math.isfinite(t):
                stats["parse_failures"] += 1
                continue
            if t < prev_time:
                stats["nonmonotonic"] += 1
            prev_time = max(prev_time, t)
            # Quote at boundary t is book just before any event at time t;
            # OFI buckets and prices align without using future events.
            while next_index < NBUCKET and t >= OPEN + (next_index + 1) * 300:
                capture(next_index)
                next_index += 1
            if next_index < NBUCKET:
                book.update(event, oid, qty, price, side)
        while next_index < NBUCKET:
            capture(next_index)
            next_index += 1
    finally:
        process.stdout.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"7z returned {code} for {archive}")
    stats["valid_price_buckets"] = int(np.isfinite(values["mid"]).sum())
    stats["book_warnings"] = dict(book.stats)
    return values, stats


def extract(args):
    panel_dir = Path(args.input)
    manifest = json.loads((panel_dir / "manifest.json").read_text())
    days, tickers = manifest["days"], manifest["tickers"]
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".stock_day_cache"
    cache.mkdir(exist_ok=True)
    n_days, n_stocks = len(days), len(tickers)
    q = {name: np.full((n_days, NBUCKET, n_stocks), np.nan) for name in FIELDS}
    qc = []
    for di, day in enumerate(days):
        for ti, ticker in enumerate(tickers):
            path = cache / f"{day}_{ticker}.npz"
            if path.exists():
                try:
                    with np.load(path) as obj:
                        vals = {k: obj[k] for k in FIELDS}
                        stat = json.loads(str(obj["qc_json"].item()))
                    if any(np.shape(x) != (NBUCKET,) for x in vals.values()):
                        raise ValueError("Cached quote arrays have invalid shape")
                except Exception as exc:
                    print(f"Ignoring corrupt cache {path}: {exc}", flush=True)
                    path.unlink()
                else:
                    for k in FIELDS:
                        q[k][di, :, ti] = vals[k]
                    qc.append(stat)
                    continue
            print(f"[{di + 1}/{n_days} day, {ti + 1}/{n_stocks} stock] "
                  f"{day} {ticker} price reconstruction", flush=True)
            archive = root / day / (ticker + ".7z")
            try:
                if not archive.is_file():
                    raise FileNotFoundError(archive)
                vals, stats = archive_quotes(archive)
                stat = {"day": day, "ticker": ticker, "status": "provisional",
                        **{k: json.dumps(v) if isinstance(v, dict) else v
                           for k, v in stats.items()}}
                for k in FIELDS:
                    q[k][di, :, ti] = vals[k]
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **vals, qc_json=json.dumps(stat))
                temp.replace(path)
                qc.append(stat)
            except Exception as exc:
                print("FAILED:", day, ticker, exc, flush=True)
                qc.append({"day": day, "ticker": ticker, "status": "failed",
                           "error": str(exc)})

    timestamps = pd.read_csv(panel_dir / "ofi_raw_5min_stacked.csv", usecols=["minute"]).minute
    if len(timestamps) != n_days * NBUCKET:
        raise ValueError("Source OFI timestamp count disagrees with the price tensor")
    wide = pd.DataFrame(q["mid"].reshape(-1, n_stocks), columns=tickers)
    wide.insert(0, "minute", timestamps)
    wide.to_csv(out / "midquote_end_5min_stacked.csv", index=False)
    rows = []
    for di, day in enumerate(days):
        for bucket in range(NBUCKET):
            at = timestamps.iloc[di * NBUCKET + bucket]
            for ti, ticker in enumerate(tickers):
                rows.append((at, ticker, *(q[name][di, bucket, ti] for name in
                                           ("bid", "ask", "bid_size", "ask_size"))))
    pd.DataFrame(rows, columns=["minute", "ticker", "bid", "ask", "bid_size",
                                "ask_size"]).to_csv(out / "end_of_bucket_quotes.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "quote_extraction_qc.csv", index=False)
    (out / "README.txt").write_text(
        "Prices use message-only reconstructed best bid/ask at 5-minute bucket END.\n"
        "The minute column is the bucket START, matching OFI rows.\n"
        "Midquote is (bid + ask) / 2 in source price units; log returns are scale-invariant.\n"
        "No ground-truth snapshot matching has been performed.\n"
        "All values are provisional and may contain reconstruction artifacts.\n")
    print("Saved:", out / "midquote_end_5min_stacked.csv", "shape:", wide.shape,
          "missing:", int(wide.iloc[:, 1:].isna().sum().sum()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="team_results_long")
    parser.add_argument("--output", default="team_results_long/prices")
    parser.add_argument("--data-root", default="/lobster/2025")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd


def _bid_order_flow(price: pd.Series, size: pd.Series) -> pd.Series:
    prev_p = price.shift(1)
    prev_q = size.shift(1)
    return pd.Series(
        np.select(
            [price > prev_p, price == prev_p, price < prev_p],
            [size, size - prev_q, -size],
            default=np.nan,
        ),
        index=price.index,
        dtype=float,
    )


def _ask_order_flow(price: pd.Series, size: pd.Series) -> pd.Series:
    prev_p = price.shift(1)
    prev_q = size.shift(1)
    return pd.Series(
        np.select(
            [price > prev_p, price == prev_p, price < prev_p],
            [-size, size - prev_q, size],
            default=np.nan,
        ),
        index=price.index,
        dtype=float,
    )


def add_event_level_ofi(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """Compute event-level bid OF, ask OF, and OFI at each LOB level."""
    out = df.copy()
    for m in range(1, levels + 1):
        bid = _bid_order_flow(out[f"bid_price_{m}"], out[f"bid_size_{m}"])
        ask = _ask_order_flow(out[f"ask_price_{m}"], out[f"ask_size_{m}"])
        out[f"bid_of_{m}"] = bid
        out[f"ask_of_{m}"] = ask
        out[f"ofi_{m}"] = bid - ask
    return out


def bucket_features(df: pd.DataFrame, levels: int = 10, bucket_seconds: int = 10) -> pd.DataFrame:
    """Aggregate event OFI into fixed time buckets and normalize by mean book depth."""
    if "ofi_1" not in df.columns:
        df = add_event_level_ofi(df, levels=levels)

    work = df.sort_values("timestamp").copy()
    work["bucket"] = work["timestamp"].dt.floor(f"{bucket_seconds}s")

    grouped = work.groupby("bucket", sort=True)
    result = pd.DataFrame(index=grouped.size().index)
    result["events"] = grouped.size()
    result["midprice"] = grouped["midprice"].last()
    result["spread"] = grouped["spread"].last()

    depth_cols = []
    for m in range(1, levels + 1):
        depth_cols.extend([f"bid_size_{m}", f"ask_size_{m}"])
    work["depth_mean_event"] = work[depth_cols].mean(axis=1)
    q = work.groupby("bucket")["depth_mean_event"].mean().replace(0, np.nan)
    result["avg_depth"] = q

    for m in range(1, levels + 1):
        raw = grouped[f"ofi_{m}"].sum(min_count=1)
        result[f"ofi_raw_{m}"] = raw
        result[f"ofi_{m}"] = raw / q

    result["log_return"] = np.log(result["midprice"] / result["midprice"].shift(1))
    result.index.name = "timestamp"
    return result.reset_index()

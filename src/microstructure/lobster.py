from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]


@dataclass(frozen=True)
class LobsterDay:
    ticker: str
    date: str
    message_path: Path
    orderbook_path: Path


def orderbook_columns(levels: int = 10) -> list[str]:
    cols: list[str] = []
    for level in range(1, levels + 1):
        cols.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return cols


def extract_7z(archive: str | Path, output_dir: str | Path) -> Path:
    """Extract a LOBSTER .7z archive using py7zr."""
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required to extract LOBSTER archives. Run pip install -r requirements.txt."
        ) from exc

    archive = Path(archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted"
    if marker.exists():
        return output_dir

    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(path=output_dir)
    marker.write_text(archive.name)
    return output_dir


def _infer_ticker_date(path: Path) -> tuple[str, str]:
    name = path.name
    m = re.search(r"([A-Za-z.]+)_(\d{4}-\d{2}-\d{2})_", name)
    if not m:
        raise ValueError(f"Could not infer ticker/date from LOBSTER filename: {name}")
    return m.group(1).upper(), m.group(2)


def discover_days(root: str | Path) -> list[LobsterDay]:
    """Pair LOBSTER message/orderbook CSVs recursively below root."""
    root = Path(root)
    csvs = list(root.rglob("*.csv"))
    messages: dict[tuple[str, str], Path] = {}
    books: dict[tuple[str, str], Path] = {}

    for path in csvs:
        lower = path.name.lower()
        if "message" not in lower and "orderbook" not in lower:
            continue
        ticker, date = _infer_ticker_date(path)
        key = (ticker, date)
        if "message" in lower:
            messages[key] = path
        elif "orderbook" in lower:
            books[key] = path

    pairs: list[LobsterDay] = []
    for key in sorted(set(messages) & set(books)):
        ticker, date = key
        pairs.append(
            LobsterDay(
                ticker=ticker,
                date=date,
                message_path=messages[key],
                orderbook_path=books[key],
            )
        )
    return pairs


def read_lobster_day(day: LobsterDay, levels: int = 10) -> pd.DataFrame:
    """Read one synchronized LOBSTER message/orderbook pair."""
    msg = pd.read_csv(day.message_path, header=None, names=MESSAGE_COLUMNS)
    book = pd.read_csv(
        day.orderbook_path,
        header=None,
        usecols=range(4 * levels),
        names=orderbook_columns(levels),
    )
    if len(msg) != len(book):
        raise ValueError(
            f"Row mismatch for {day.ticker} {day.date}: message={len(msg)}, orderbook={len(book)}"
        )

    df = pd.concat([msg, book], axis=1)
    df["ticker"] = day.ticker
    df["date"] = pd.Timestamp(day.date)
    df["timestamp"] = df["date"] + pd.to_timedelta(df["time"], unit="s")

    price_cols = [c for c in df.columns if "price_" in c]
    df[price_cols] = df[price_cols].replace({9999999999: np.nan, -9999999999: np.nan}) / 10000.0
    df["price"] = df["price"].replace({9999999999: np.nan, -9999999999: np.nan}) / 10000.0
    df["midprice"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def read_many(days: Iterable[LobsterDay], levels: int = 10) -> dict[tuple[str, str], pd.DataFrame]:
    return {(d.ticker, d.date): read_lobster_day(d, levels=levels) for d in days}

"""No-market-data tests for early-close exclusion and history recalculation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lobster2"))
from ofi_team_pipeline import METRICS, NBUCKET
from prepare_regular_sessions import prepare


class RegularSessionTests(unittest.TestCase):
    def test_drop_early_close_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "input"
            out = Path(temp) / "clean"
            base.mkdir()
            days = ["20250703", "20250707", "20250708", "20250709"]
            names = ["AAPL", "MSFT"]
            (base / "manifest.json").write_text(json.dumps({"days": days, "tickers": names}))
            timestamps = []
            for day in days:
                opening = pd.to_datetime(day) + pd.Timedelta(hours=9, minutes=30)
                timestamps += pd.date_range(opening, periods=NBUCKET, freq="5min").strftime("%Y-%m-%d %H:%M").tolist()
            for name in METRICS:
                data = np.full((4 * NBUCKET, 2), 2.0, dtype=float)
                if name == "ofi_raw":
                    data[:NBUCKET] = 500.0
                frame = pd.DataFrame(data, columns=names)
                frame.insert(0, "minute", timestamps)
                frame.to_csv(base / f"{name}_5min_stacked.csv", index=False)
            qc = pd.DataFrame([{"day": day, "ticker": ticker, "status": "processed_provisional"}
                               for day in days for ticker in names])
            qc.to_csv(base / "reconstruction_qc.csv", index=False)
            report = prepare(base, out, ["20250703"])
            self.assertEqual(report["excluded_days"], ["20250703"])
            self.assertEqual(len(report["kept_days"]), 3)
            arr = np.load(out / "ofi_raw_tensor.npy")
            self.assertEqual(arr.shape, (3, 78, 2))
            self.assertTrue(np.all(arr == 2))
            history = pd.read_csv(out / "historical_weighted_raw_ofi.csv")
            self.assertTrue(np.isnan(history.loc[0, "AAPL"]))
            self.assertAlmostEqual(history.loc[NBUCKET, "AAPL"], 2.0)
            self.assertEqual(len(pd.read_csv(base / "ofi_raw_5min_stacked.csv")), 4 * NBUCKET)
            self.assertEqual(len(pd.read_csv(out / "reconstruction_qc.csv")), 6)
            self.assertTrue(json.loads((out / "manifest.json").read_text())["regular_sessions_only"])


if __name__ == "__main__":
    unittest.main()

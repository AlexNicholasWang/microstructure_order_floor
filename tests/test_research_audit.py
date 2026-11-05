"""Market-data-free regression tests for cached-output audit and RPCA sensitivity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lobster2"))
from audit_research import audit, check_ratio
from refine_rpca import decompose, evaluate


class NormalizationTests(unittest.TestCase):
    def test_ratio_formula_and_missing_denominator(self):
        raw = np.array([[4.0, -9.0], [2.0, 8.0]])
        denominator = np.array([[2.0, 3.0], [0.0, 4.0]])
        actual = np.array([[2.0, -3.0], [np.nan, 2.0]])
        result = check_ratio(raw, denominator, actual)
        self.assertEqual(result["ratio_mismatches"], 0)
        self.assertEqual(result["zero_denominators"], 1)
        actual[0, 0] = 7.0
        self.assertEqual(check_ratio(raw, denominator, actual)["ratio_mismatches"], 1)


class CachedPanelAuditTests(unittest.TestCase):
    def test_synthetic_80_stock_three_day_consistency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            days = ["20250729", "20250730", "20250731"]
            tickers = [f"T{i:03d}" for i in range(80)]
            (root / "manifest.json").write_text(json.dumps({"days": days, "tickers": tickers}))
            clock = []
            for day in days:
                first = pd.to_datetime(day) + pd.Timedelta(hours=9, minutes=30)
                clock.extend(pd.date_range(first, periods=78, freq="5min").strftime("%Y-%m-%d %H:%M"))
            raw = np.arange(234 * 80, dtype=float).reshape(234, 80) / 1000 + 1
            volume = np.full_like(raw, 2.0)
            depth = np.full_like(raw, 4.0)
            volume[0, 0] = 0
            values = {
                "ofi_raw": raw,
                "executed_volume": volume,
                "mean_depth": depth,
                "ofi_per_volume": np.divide(raw, volume, out=np.full_like(raw, np.nan),
                                             where=volume > 0),
                "ofi_per_depth": raw / depth,
            }
            for metric, array in values.items():
                frame = pd.DataFrame(array, columns=tickers)
                frame.insert(0, "minute", clock)
                frame.to_csv(root / f"{metric}_5min_stacked.csv", index=False)
            qc = pd.DataFrame([
                {"day": day, "ticker": ticker, "status": "processed_provisional",
                 "book_warnings": "{}", "bad_lines": 0, "invalid_quotes": 0,
                 "nonmonotonic": 0, "zero_volume_buckets": 0}
                for day in days for ticker in tickers
            ])
            qc.to_csv(root / "reconstruction_qc.csv", index=False)
            result = audit(root)
            self.assertEqual(result["status"], "PASS_WITH_PROVISIONAL_QUOTES")
            self.assertEqual(result["panel_missing_values"]["ofi_per_volume"], 1)
            self.assertEqual(result["depth_ratio_check"]["ratio_mismatches"], 0)
            self.assertEqual(result["volume_ratio_check"]["ratio_mismatches"], 0)
            self.assertEqual(result["stock_day_status"]["processed_provisional"], 240)


class RPCASensitivityTests(unittest.TestCase):
    def test_synthetic_low_rank_with_anomaly(self):
        rng = np.random.default_rng(2025)
        low = rng.normal(size=(12, 1)) @ rng.normal(size=(1, 5))
        signal = low.copy()
        signal[4, 3] += 12.0
        L, S, E, iterations, err = decompose(signal, multiplier=1.0, steps=100)
        self.assertTrue(np.allclose(signal, L + S + E, atol=1e-7))
        self.assertTrue(np.isfinite(err))
        self.assertGreater(iterations, 0)
        row = evaluate(signal, signal, multiplier=1.0, steps=100)
        self.assertGreaterEqual(row["train_sparse_nonzero_fraction"], 0)
        self.assertLessEqual(row["train_sparse_nonzero_fraction"], 1)
        self.assertGreaterEqual(row["train_lowrank_numerical_rank"], 1)


if __name__ == "__main__":
    unittest.main()

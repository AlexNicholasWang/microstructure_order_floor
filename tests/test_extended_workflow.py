"""Regression tests for the extended OFI workflow; no proprietary data required."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lobster2"))
from ofi_team_pipeline import METRICS, NBUCKET
from extend_cached_panel import available_days, build
from extract_midquotes import archive_quotes
from predict_residual_returns import (forward_targets, trailing_three,
                                      prepare_factor_model, evaluate_stock, fit_ridge)
from validate_reference_quotes import validate


class WorkflowTests(unittest.TestCase):
    def test_midquote_snapshot_boundary_is_causal(self):
        # t=09:35 event belongs to the following bucket and cannot change
        # the first bucket's closing midquote.
        class FakeProc:
            def __init__(self, lines):
                self.stdout = io.StringIO(lines)
            def wait(self):
                return 0
        message = "\n".join([
            "34201,1,1,100,100,1,",
            "34202,1,2,200,110,-1,",
            "34500,3,2,200,110,-1,",
            "34501,1,3,300,108,-1,",
            "34800,7,0,0,0,0,",
        ]) + "\n"
        with patch("extract_midquotes.subprocess.Popen", return_value=FakeProc(message)):
            data, qc = archive_quotes(Path("synthetic.7z"))
        self.assertEqual(len(data["mid"]), NBUCKET)
        self.assertEqual(data["mid"][0], 105.0)
        self.assertEqual(data["mid"][1], 104.0)
        self.assertEqual(data["bid_size"][0], 100.0)
        self.assertEqual(qc["parse_failures"], 0)

    def test_labels_never_cross_session(self):
        mid = np.array([[100.0], [110], [121], [200], [180], [171]])
        day = np.array(["day1"] * 3 + ["day2"] * 3)
        result = forward_targets(mid, day, horizon=1)
        self.assertAlmostEqual(result[0, 0], math.log(1.1))
        self.assertTrue(np.isnan(result[2, 0]))
        self.assertTrue(np.isnan(result[5, 0]))
        self.assertAlmostEqual(result[3, 0], math.log(0.9))

    def test_trailing_three_does_not_cross_session(self):
        values = np.arange(8.0).reshape(-1, 1)
        day = np.array(["a"] * 4 + ["b"] * 4)
        trail = trailing_three(values, day)
        self.assertTrue(np.isnan(trail[4, 0]))
        self.assertTrue(np.isnan(trail[5, 0]))
        self.assertEqual(trail[6, 0], 5)

    def test_train_only_svd_and_forward_model(self):
        rng = np.random.default_rng(44)
        data = rng.normal(size=(240, 8))
        day = np.array(["2025-07-29"] * 80 + ["2025-07-30"] * 80 + ["2025-07-31"] * 80)
        z, residual, info = prepare_factor_model(data, day, ["2025-07-29", "2025-07-30"], 5, .8, 3)
        self.assertEqual(info["rank"], 3)
        self.assertEqual(z.shape, data.shape)
        self.assertTrue(np.isfinite(residual).all())
        y = residual[:, 0] * 0.001 + rng.normal(0, 0.0001, size=len(data))
        trend = trailing_three(residual, day)
        score = evaluate_stock(y, residual[:, 0], trend[:, 0], day != "2025-07-31",
                               day == "2025-07-31", alpha=10, min_train=20, min_test=10)
        self.assertIsNotNone(score)
        self.assertGreater(score["test_obs"], 50)
        beta = fit_ridge(np.ones((10, 2)), np.ones(10), 10)
        self.assertTrue(np.isfinite(beta).all())

    def test_reference_validator(self):
        reconstructed = pd.DataFrame([
            {"minute": "2025-07-29 09:30", "ticker": "X",
             "bid": 1000, "ask": 1010, "bid_size": 30, "ask_size": 40}
        ])
        ref = reconstructed.copy()
        ref[["bid", "ask"]] /= 100
        stats, matched = validate(reconstructed, ref, 100.0, 0, 0)
        self.assertEqual(stats["matching_keys"], 1)
        self.assertEqual(stats["all_four_fields_match_fraction_among_valid"], 1)
        self.assertEqual(len(matched), 1)

    def test_incremental_days_reuse_prior_stock_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archives = root / "archive"
            base = root / "base"
            out = root / "extended"
            archives.mkdir()
            base.mkdir()
            days = ["20250728", "20250729", "20250730", "20250731"]
            prior_days = days[1:]
            tickers = ["X", "Y"]
            for day in days:
                folder = archives / day
                folder.mkdir()
                for ticker in tickers:
                    (folder / f"{ticker}.7z").touch()
            self.assertEqual(available_days(archives, tickers, "20250731"), days)
            (base / "manifest.json").write_text(json.dumps(
                {"days": prior_days, "tickers": tickers}))
            for metric in METRICS:
                frame = pd.DataFrame(np.ones((3 * NBUCKET, 2)), columns=tickers)
                clock = []
                for day in prior_days:
                    first = pd.to_datetime(day) + pd.Timedelta(hours=9, minutes=30)
                    clock.extend(pd.date_range(first, periods=NBUCKET, freq="5min")
                                 .strftime("%Y-%m-%d %H:%M"))
                frame.insert(0, "minute", clock)
                frame.to_csv(base / (metric + "_5min_stacked.csv"), index=False)
            pd.DataFrame([{"day": d, "ticker": ticker,
                           "status": "processed_provisional"}
                          for d in prior_days for ticker in tickers]).to_csv(
                              base / "reconstruction_qc.csv", index=False)
            class Args:
                pass
            args = Args()
            args.base = str(base)
            args.output = str(out)
            args.data_root = str(archives)
            args.until = "20250731"
            args.max_days = 4
            args.half_life_days = 5.0
            args.dry_run = False
            def synthetic_archive(path):
                stat = {"rows": 10, "bad_lines": 0, "book_warnings": {}}
                return ({m: np.ones(NBUCKET) * 3 for m in METRICS}, stat)
            with patch("extend_cached_panel.one_archive", side_effect=synthetic_archive) as fake:
                build(args)
                self.assertEqual(fake.call_count, 2)
            self.assertTrue((out / "manifest.json").exists())
            stacked = pd.read_csv(out / "ofi_raw_5min_stacked.csv")
            self.assertEqual(stacked.shape, (4 * NBUCKET, 3))
            with patch("extend_cached_panel.one_archive") as fake:
                build(args)
                fake.assert_not_called()  # Resumes from individual cached stock-days.


if __name__ == "__main__":
    unittest.main()

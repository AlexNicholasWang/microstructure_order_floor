# Level 1 OFI matrix (n × 80 target)

## Definition

Each row corresponds to a synchronized 10-second interval. Each column corresponds to a stock. For asset `i` and interval `t`:

```text
X[t, i] = sum_{events n in interval t} (OF_bid[n, level 1] - OF_ask[n, level 1])
```

Values are in shares and are computed from consecutive Level 1 LOBSTER order-book states. This is a **time-by-asset feature matrix**, not the cross-impact coefficient matrix. It is the input to the next cross-sectional factor model.

```text
                  AAPL      AIG       GE       ...       SPY      WMT      ... (80 assets)
2019-01-02 09:30:00   OFI       OFI      OFI        ...       OFI      OFI
2019-01-02 09:30:10   OFI       OFI      OFI        ...       OFI      OFI
2019-01-02 09:30:20   OFI       OFI      OFI        ...       OFI      OFI
        ...            ...       ...      ...        ...       ...      ...
```

The words `OFI` above are placeholders illustrating matrix orientation, **not calculated observations**.

## How to generate it

```bash
pip install -r requirements.txt
python scripts/build_level1_ofi_matrix.py \
  /path/to/AAPL.7z /path/to/AIG.7z /path/to/GE.7z \
  /path/to/PSA.7z /path/to/SPY.7z /path/to/WMT.7z
```

By default the script uses 10-second buckets and an **inner join**, so every retained row has observed data for every supplied ticker. `--join outer` preserves more timestamps but uses NaN wherever an asset has no bucket.

The output files are in `outputs/level1/`:

- `level1_ofi_matrix.csv`: timestamp × asset OFI matrix
- `coverage.csv`: number of observed buckets per asset
- `summary.txt`: shape, names and coverage

To enforce the **exact 80 assets**, create a text file with one desired ticker per line and pass `--tickers-file assets_80.txt`. The script requires exactly 80 unique names and retains missing asset columns as NaN. It **does not manufacture OFI values**.

## Data coverage limitation

Currently uploaded files contain 2019 archives for **AAPL, AIG, GE, PSA, SPY, WMT** and 2016 archives for **FTV, PSA, TFX**. Those dates cannot be combined to form a single synchronized panel. A real `n × 80` numerical matrix requires additional LOBSTER data for the 80 specified assets over matching dates.

No empirical CSV is checked in yet: the pipeline must be run on extracted data (or the supplied archives) before any real matrix values exist.

## Following research step

Once an observed Level 1 matrix `X ∈ R^(n×80)` exists, estimate common order-flow factors, e.g. PCA or explicit market/sector factors, and analyze `X - X_hat` as residual order flow. Keep the fitting and return evaluation windows time-ordered to avoid leakage.

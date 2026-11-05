#!/usr/bin/env python3
"""Train-only Robust PCA sensitivity study from cached five-minute OFI matrices.

Does not reread archives, predict returns, or select an optimum using test data.
Usage on the authorized host:
  OPENBLAS_NUM_THREADS=1 python scripts/lobster2/refine_rpca.py \
    --input team_results --metric ofi_per_depth --train-days 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def soft(X: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0.0)


def decompose(A: np.ndarray, multiplier: float = 1.0,
              steps: int = 150, tolerance: float = 1e-5) -> tuple:
    """Unweighted inexact ALM; lambda multiplier is relative to 1/sqrt(max(shape))."""
    if multiplier <= 0:
        raise ValueError("Lambda multiplier must be positive")
    lam = multiplier / np.sqrt(max(A.shape))
    norm2 = float(np.linalg.norm(A, 2))
    Y = A / max(norm2, float(np.max(np.abs(A))) / lam, 1e-12)
    mu = 1.25 / max(norm2, 1e-12)
    L = np.zeros_like(A)
    S = np.zeros_like(A)
    for iteration in range(steps):
        u, singular, vt = np.linalg.svd(A - S + Y / mu, full_matrices=False)
        L = (u * np.maximum(singular - 1 / mu, 0)) @ vt
        S = soft(A - L + Y / mu, lam / mu)
        remainder = A - L - S
        err = float(np.linalg.norm(remainder) / (np.linalg.norm(A) + 1e-12))
        if err < tolerance:
            break
        Y += mu * remainder
        mu *= 1.5
    return L, S, A - L - S, iteration + 1, err


def load_train_test(folder: Path, metric: str, train_days: int) -> tuple:
    frame = pd.read_csv(folder / f"{metric}_5min_stacked.csv")
    columns = frame.columns[1:]
    X = frame[columns].to_numpy(dtype=float)
    days = frame.minute.str.slice(0, 10)
    unique = sorted(days.unique())
    if not 1 <= train_days < len(unique):
        raise ValueError("Need at least one training and one held-out day")
    finite = np.isfinite(X).all(axis=1)
    train = days.isin(unique[:train_days]).to_numpy() & finite
    test = days.isin(unique[train_days:]).to_numpy() & finite
    if train.sum() < 10 or test.sum() < 2:
        raise ValueError("Insufficient complete training/held-out rows")
    mean = X[train].mean(axis=0)
    scale = X[train].std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid per-stock training variance")
    return (X[train] - mean) / scale, (X[test] - mean) / scale, unique[:train_days], unique[train_days:]


def evaluate(A: np.ndarray, heldout: np.ndarray, multiplier: float,
             steps: int = 150) -> dict:
    L, S, E, n_iter, error = decompose(A, multiplier=multiplier, steps=steps)
    _, spectrum, vt = np.linalg.svd(L, full_matrices=False)
    rank = max(1, int(np.sum(spectrum > 1e-4 * max(spectrum[0], 1e-12))))
    robust_basis = vt[:rank]
    projection = (heldout @ robust_basis.T) @ robust_basis
    # Same-rank unweighted SVD comparison; projection only, not forecasting.
    _, _, baseline_vt = np.linalg.svd(A, full_matrices=False)
    baseline_basis = baseline_vt[:rank]
    baseline = (heldout @ baseline_basis.T) @ baseline_basis
    denom = float(np.sum(heldout * heldout))
    def energy_fraction(estimate):
        return float(1 - np.sum((heldout - estimate) ** 2) / denom) if denom else None
    return {
        "lambda_multiplier": float(multiplier),
        "lambda_absolute": float(multiplier / np.sqrt(max(A.shape))),
        "train_sparse_nonzero_fraction": float(np.mean(S != 0)),
        "train_sparse_squared_energy_share": float(np.sum(S * S) / np.sum(A * A)),
        "train_lowrank_numerical_rank": rank,
        "train_reconstruction_error": error,
        "iterations": n_iter,
        "heldout_projection_energy_fraction": energy_fraction(projection),
        "same_rank_svd_projection_energy_fraction": energy_fraction(baseline),
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("team_results"))
    parser.add_argument("--output", type=Path, default=Path("team_results/rpca_tuning"))
    parser.add_argument("--metric", choices=("ofi_per_depth", "ofi_per_volume"), default="ofi_per_depth")
    parser.add_argument("--train-days", type=int, default=2)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    train, test, trained, heldout = load_train_test(args.input, args.metric, args.train_days)
    rows = [evaluate(train, test, m, steps=args.steps) for m in args.multipliers]
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("lambda_multiplier")
    result.to_csv(args.output / f"{args.metric}_sensitivity.csv", index=False)
    metadata = {
        "metric": args.metric, "train_days": trained, "heldout_days": heldout,
        "training_rows": int(len(train)), "heldout_rows": int(len(test)),
        "warning": "Exploratory rank/sparsity sensitivity; no return labels; heldout observations cannot tune lambda.",
    }
    (args.output / f"{args.metric}_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(result.lambda_multiplier, 100 * result.train_sparse_nonzero_fraction, marker="o")
    axes[0].set(xlabel="RPCA lambda multiplier", ylabel="Nonzero sparse entries (%)",
                title="Training sparsity sensitivity")
    axes[1].plot(result.lambda_multiplier, result.train_lowrank_numerical_rank, marker="o")
    axes[1].set(xlabel="RPCA lambda multiplier", ylabel="Numerical rank of training L",
                title="Low-rank complexity sensitivity")
    fig.suptitle(f"{args.metric}: training-only sensitivity (no returns)")
    fig.tight_layout()
    fig.savefig(args.output / f"{args.metric}_sensitivity.png", dpi=180)
    plt.close(fig)
    print(result.to_string(index=False))
    print("Wrote private RPCA sensitivity results under", args.output)


if __name__ == "__main__":
    main()

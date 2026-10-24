from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler


@dataclass
class IntegratedOFIResult:
    frame: pd.DataFrame
    weights: pd.Series
    explained_variance_ratio: float


@dataclass
class CommonFactorResult:
    factor: pd.Series
    loadings: pd.DataFrame
    residuals: pd.DataFrame
    explained_variance_ratio: float


def integrated_ofi(df: pd.DataFrame, levels: int = 10) -> IntegratedOFIResult:
    """Estimate integrated OFI using PC1 across normalized LOB levels."""
    cols = [f"ofi_{m}" for m in range(1, levels + 1)]
    valid = df[cols].dropna()
    if len(valid) < 3:
        raise ValueError("Need at least 3 complete buckets to estimate integrated OFI.")

    pca = PCA(n_components=1)
    pca.fit(valid.to_numpy())
    w = pca.components_[0].astype(float)
    if w.sum() < 0:
        w = -w

    denom = np.abs(w).sum()
    projection = df[cols].to_numpy() @ w / denom
    out = df.copy()
    out["integrated_ofi"] = projection
    weights = pd.Series(w, index=cols, name="pc1_weight")
    return IntegratedOFIResult(out, weights, float(pca.explained_variance_ratio_[0]))


def align_integrated_ofi(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join integrated OFI series by timestamp."""
    pieces = []
    for ticker, df in frames.items():
        s = df.set_index("timestamp")["integrated_ofi"].rename(ticker)
        pieces.append(s)
    if not pieces:
        raise ValueError("No ticker frames supplied.")
    return pd.concat(pieces, axis=1, join="inner").dropna().sort_index()


def common_ofi_factor(ofi_wide: pd.DataFrame) -> CommonFactorResult:
    """Estimate OFI_i,t = mu_i + gamma_i F_t + tau_i,t."""
    if ofi_wide.shape[1] < 2:
        raise ValueError("Need at least two synchronized stocks for a cross-asset factor model.")

    scaler = StandardScaler()
    z = scaler.fit_transform(ofi_wide)
    pca = PCA(n_components=1)
    factor_values = pca.fit_transform(z).ravel()
    if pca.components_[0].sum() < 0:
        factor_values = -factor_values

    factor = pd.Series(factor_values, index=ofi_wide.index, name="ofi_common_factor")
    load_rows = []
    residuals = pd.DataFrame(index=ofi_wide.index)

    X = factor.to_numpy().reshape(-1, 1)
    for ticker in ofi_wide.columns:
        y = ofi_wide[ticker].to_numpy()
        model = LinearRegression().fit(X, y)
        fitted = model.predict(X)
        residuals[ticker] = y - fitted
        load_rows.append(
            {
                "ticker": ticker,
                "intercept": float(model.intercept_),
                "gamma": float(model.coef_[0]),
                "r2": float(model.score(X, y)),
            }
        )

    loadings = pd.DataFrame(load_rows).set_index("ticker")
    return CommonFactorResult(
        factor=factor,
        loadings=loadings,
        residuals=residuals,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
    )


def align_returns(frames: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    pieces = []
    for ticker, df in frames.items():
        s = df.set_index("timestamp")["log_return"].rename(ticker)
        pieces.append(s)
    return pd.concat(pieces, axis=1, join="inner").reindex(index).dropna()


def cross_impact_matrix(
    returns: pd.DataFrame,
    factor: pd.Series,
    residuals: pd.DataFrame,
    use_lasso: bool = True,
) -> pd.DataFrame:
    """Estimate the CC-style contemporaneous cross-impact coefficient matrix."""
    idx = returns.index.intersection(factor.index).intersection(residuals.index)
    R = returns.loc[idx].dropna()
    idx = R.index
    F = factor.loc[idx]
    T = residuals.loc[idx, R.columns]

    X = np.column_stack([F.to_numpy(), T.to_numpy()])
    Xs = StandardScaler().fit_transform(X)
    matrix = pd.DataFrame(index=R.columns, columns=R.columns, dtype=float)

    for target in R.columns:
        y = R[target].to_numpy()
        if use_lasso and len(y) >= 20:
            cv = min(5, max(2, len(y) // 10))
            model = LassoCV(cv=cv, random_state=0, max_iter=20000).fit(Xs, y)
            coefs = model.coef_[1:]
        else:
            model = LinearRegression().fit(Xs, y)
            coefs = model.coef_[1:]
        matrix.loc[target, :] = coefs
    return matrix

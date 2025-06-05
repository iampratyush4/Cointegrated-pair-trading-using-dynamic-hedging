import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.stattools import coint
import logging

logger = logging.getLogger(__name__)

def half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion for a spread series S_t 
    by fitting: ΔS_t = a + kappa * S_{t-1} + ε_t. 
    Returns: hl = -ln(2) / kappa.
    """
    spread_lag = spread.shift(1).dropna()
    spread_ret = (spread - spread_lag).dropna()
    spread_lag = spread_lag.loc[spread_ret.index]

    # Add constant
    X = np.vstack([np.ones(len(spread_lag)), spread_lag.values]).T
    y = spread_ret.values

    # OLS regression
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    kappa = beta[1]
    if kappa >= 0:
        return np.inf
    hl = -np.log(2) / kappa
    return hl


def zscore(series: pd.Series) -> pd.Series:
    """
    Compute z-score based on rolling mean/std.
    Assumes 'series' has no NaN for window-sized segments.
    """
    mean = series.rolling(series.name + "_mean").mean()  # placeholder
    # But we prefer to pass rolling window explicitly, so this may not be used directly.


def compute_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score over 'window' periods.
    """
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std()
    return (spread - m) / s


def cluster_universe(returns: pd.DataFrame, cluster_size: int) -> dict:
    """
    Perform hierarchical agglomerative clustering on returns to group assets 
    into clusters of approximate size 'cluster_size'. Returns a dict mapping 
    cluster_label -> list of tickers.
    """
    n_assets = returns.shape[1]
    # Compute pairwise distance as 1 - correlation
    corr = returns.corr().fillna(0)
    dist = 1 - corr.abs()
    # Convert to condensed form for clustering if needed, but sklearn Agglomerative supports precomputed.
    n_clusters = max(1, n_assets // cluster_size)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average", affinity="precomputed"
    )
    labels = model.fit_predict(dist.values)
    clusters = {}
    tickers = returns.columns.tolist()
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(tickers[i])
    logger.info(f"Formed {n_clusters} clusters.")
    return clusters


def rolling_cointegration_test(
    series1: pd.Series,
    series2: pd.Series,
    window: int,
    step: int,
    pval_threshold: float,
    min_valid_periods: int
) -> bool:
    """
    Run rolling Engle‐Granger cointegration tests over consecutive windows.
    Return True if at least 'min_valid_periods' consecutive windows have pval < threshold.
    """
    n = len(series1)
    valid = 0
    for start in range(0, n - window + 1, step):
        seg1 = series1.iloc[start : start + window]
        seg2 = series2.iloc[start : start + window]
        score, pval, _ = coint(seg1, seg2)
        if pval < pval_threshold:
            valid += 1
            if valid >= min_valid_periods:
                return True
        else:
            valid = 0
    return False

import pandas as pd
import numpy as np
import itertools
import logging
from utils import cluster_universe, rolling_cointegration_test, half_life

logger = logging.getLogger(__name__)

class PairSelector:
    """
    From a universe of assets, cluster similar ones, then within each cluster 
    run rolling cointegration tests. For stable pairs, compute static hedge ratio 
    via OLS and half‐life of the spread.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        cluster_size: int = 20,
        coint_pval_threshold: float = 0.05,
        rolling_window: int = 252,
        rolling_step: int = 63,
        min_valid_periods: int = 2
    ):
        """
        :param prices: DataFrame of asset prices (aligned), columns = tickers.
        """
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.cluster_size = cluster_size
        self.pval_threshold = coint_pval_threshold
        self.rolling_window = rolling_window
        self.rolling_step = rolling_step
        self.min_valid_periods = min_valid_periods

    def select_pairs(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: [ticker1, ticker2, beta_ols, half_life_days].
        Only includes pairs that pass the rolling cointegration test.
        """
        # 1) Cluster the universe
        clusters = cluster_universe(self.returns, self.cluster_size)

        selected = []
        for cluster_id, tickers in clusters.items():
            if len(tickers) < 2:
                continue
            logger.info(f"Testing cluster {cluster_id} with {len(tickers)} tickers.")
            # For each unique pair in that cluster:
            for t1, t2 in itertools.combinations(tickers, 2):
                s1 = self.prices[t1]
                s2 = self.prices[t2]
                # 2) Rolling cointegration test
                try:
                    is_coint = rolling_cointegration_test(
                        s1, s2,
                        window=self.rolling_window,
                        step=self.rolling_step,
                        pval_threshold=self.pval_threshold,
                        min_valid_periods=self.min_valid_periods
                    )
                except Exception as e:
                    logger.warning(f"Cointegration test failed for {t1}-{t2}: {e}")
                    continue

                if not is_coint:
                    continue

                # 3) Compute static OLS hedge ratio on full period
                # Solve s1 = alpha + beta*s2 + eps
                X = np.vstack([np.ones(len(s2)), s2.values]).T
                y = s1.values
                ols_beta = np.linalg.lstsq(X, y, rcond=None)[0][1]

                # 4) Compute half-life of spread
                spread = s1 - ols_beta * s2
                hl = half_life(spread)

                # 5) Save if half-life is finite and reasonable (< window)
                if np.isfinite(hl) and hl > 0 and hl < self.rolling_window:
                    selected.append({
                        "ticker1": t1,
                        "ticker2": t2,
                        "beta_ols": ols_beta,
                        "half_life": hl
                    })
                    logger.info(f"Selected pair {t1}-{t2}: beta={ols_beta:.4f}, half-life={hl:.1f} days.")
        if not selected:
            logger.warning("No cointegrated pairs found in the universe.")
            return pd.DataFrame(columns=["ticker1", "ticker2", "beta_ols", "half_life"])
        return pd.DataFrame(selected)

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Given spread & dynamic betas, computes z-scores, applies adaptive thresholds, 
    volatility & momentum filters, and outputs daily position signals (signed, float).
    """

    def __init__(
        self,
        price1: pd.Series,
        price2: pd.Series,
        kalman_df: pd.DataFrame,
        config: dict
    ):
        """
        :param price1: Series of prices for ticker1.
        :param price2: Series of prices for ticker2.
        :param kalman_df: DataFrame with columns ['alpha', 'beta', 'spread'] indexed by date.
        :param config: dictionary under key 'signal' from config.yaml.
        """
        self.price1 = price1
        self.price2 = price2
        self.alpha = kalman_df["alpha"]
        self.beta = kalman_df["beta"]
        self.spread = kalman_df["spread"]
        self.dates = kalman_df.index
        self.z_window = config["zscore_window"]
        self.entry_z = config["entry_z"]
        self.exit_z = config["exit_z"]
        self.target_vol = config["target_vol"]
        self.min_vol_pct = config["min_vol_percentile"]
        self.momentum_filter = config["momentum_filter"]

        # Placeholder for signals & positions
        self.zscore = None
        self.entry_thresh = None
        self.exit_thresh = None
        self.positions = None  # will store a DataFrame with columns ['pos1', 'pos2']

    def generate(self, costs: dict, volume: pd.DataFrame) -> pd.DataFrame:
        """
        1. Compute rolling z-score of spread.
        2. Compute dynamic entry/exit thresholds (scaled by rolling vol).
        3. Apply volatility & momentum filters.
        4. Compute volatility‐normalized position sizing:
           pos1 = +1 (long) or -1 (short) * (target_vol / spread_vol) 
           pos2 = -beta * pos1
        5. Return DataFrame with columns ['pos1', 'pos2'], indexed by date.
        """
        n = len(self.spread)
        df = pd.DataFrame(index=self.dates)
        # 1) Rolling mean & std of spread
        rolling_mean = self.spread.rolling(self.z_window).mean()
        rolling_std = self.spread.rolling(self.z_window).std()
        self.zscore = (self.spread - rolling_mean) / rolling_std
        df["zscore"] = self.zscore

        # 2) Dynamic thresholds
        self.entry_thresh = self.entry_z * rolling_std
        self.exit_thresh = self.exit_z * rolling_std
        df["entry_thresh"] = self.entry_thresh
        df["exit_thresh"] = self.exit_thresh

        # 3) Volatility filter: compute rolling volatility percentile rank
        vol_rank = rolling_std.rank(pct=True) * 100
        df["vol_rank"] = vol_rank

        # 4) Momentum filter: Δspread
        dspread = self.spread.diff()
        df["dspread"] = dspread

        # 5) Initialize positions
        pos1 = np.zeros(n)
        pos2 = np.zeros(n)
        current_signal = 0  # +1 for long spread (long price1, short price2), -1 for short

        for t in range(1, n):
            # Check if volatility below threshold
            if vol_rank.iloc[t] > self.min_vol_pct:
                # Too volatile or not enough data
                current_signal = 0
            else:
                z = self.zscore.iloc[t]
                # Check existing position
                if current_signal == 0:
                    # Look for entry
                    if z > self.entry_thresh.iloc[t]:
                        # Spread is high → short spread (sell price1, buy price2)
                        # Momentum filter: require Δspread < 0 (i.e., spread rolling down)
                        if not self.momentum_filter or dspread.iloc[t] < 0:
                            current_signal = -1
                    elif z < -self.entry_thresh.iloc[t]:
                        # Spread is low → long spread (buy price1, sell price2)
                        if not self.momentum_filter or dspread.iloc[t] > 0:
                            current_signal = +1
                elif current_signal == +1:
                    # Already long spread; check exit when z ≥ -exit_thresh (close)
                    if z >= -self.exit_thresh.iloc[t]:
                        current_signal = 0
                elif current_signal == -1:
                    # Already short spread; check exit when z ≤ exit_thresh (close)
                    if z <= self.exit_thresh.iloc[t]:
                        current_signal = 0

            # 6) Position sizing based on volatility normalization
            if current_signal != 0 and rolling_std.iloc[t] > 0:
                # Dollar‐volatility of spread per unit of price1 = rolling_std / price1
                # But simpler: scale pos1 so that spread vol = target_vol
                scale = self.target_vol / rolling_std.iloc[t]
                pos1[t] = current_signal * scale
                pos2[t] = -current_signal * scale * self.beta.iloc[t]
            else:
                pos1[t] = 0.0
                pos2[t] = 0.0

        df["pos1"] = pos1
        df["pos2"] = pos2

        # Attach price series for later P&L calculations
        df["price1"] = self.price1
        df["price2"] = self.price2

        logger.info("Signals generated.")
        return df

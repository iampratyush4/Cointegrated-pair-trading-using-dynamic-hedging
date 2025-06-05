import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    Vectorized backtest engine computing daily P&L, accounting for transaction costs.
    Input: DataFrame with ['pos1','pos2','price1','price2'] columns.
    """

    def __init__(self, trade_df: pd.DataFrame, costs: dict, volume: pd.DataFrame):
        """
        :param trade_df: DataFrame indexed by date that includes:
             ['pos1', 'pos2', 'price1', 'price2'].
        :param costs: dict with keys: 'fixed_per_trade', 'slippage_coefficient'.
        :param volume: DataFrame of volume aligned with prices; used to compute slippage.
        """
        self.df = trade_df.copy()
        self.fixed_cost = costs["fixed_per_trade"]
        self.slip_coeff = costs["slippage_coefficient"]
        self.volume = volume
        self._prepare()

    def _prepare(self):
        """
        1. Compute daily returns for each leg.
        2. Align positions with returns (shift positions by 1 day to avoid look‐ahead).
        3. Compute trades (where positions change).
        """
        self.df["ret1"] = self.df["price1"].pct_change().fillna(0)
        self.df["ret2"] = self.df["price2"].pct_change().fillna(0)

        # Shift positions to align with next day's pnl
        self.df["pos1_lag"] = self.df["pos1"].shift(1).fillna(0)
        self.df["pos2_lag"] = self.df["pos2"].shift(1).fillna(0)

        # Identify when trades occur (pos changes)
        self.df["trade1"] = (self.df["pos1"] != self.df["pos1_lag"]).astype(int)
        self.df["trade2"] = (self.df["pos2"] != self.df["pos2_lag"]).astype(int)

    def run(self) -> pd.DataFrame:
        """
        Compute P&L step by step:
          1) Gross P&L: pos_lag * returns
          2) Subtract transaction costs when trade occurs:
             fixed cost + slippage based on ADV
        Returns a DataFrame augmented with P&L columns.
        """
        df = self.df.copy()

        # 1) Gross P&L (as fraction of capital)
        df["pnl1"] = df["pos1_lag"] * df["ret1"]
        df["pnl2"] = df["pos2_lag"] * df["ret2"]
        df["gross_pnl"] = df["pnl1"] + df["pnl2"]

        # 2) Transaction costs
        # For each trade, cost = fixed + slippage_coefficient * (notional / ADV)
        # Approx ADV: use prev day's volume * price
        adv1 = self.volume[df["pos1"].name].shift(1) * df["price1"].shift(1)
        adv2 = self.volume[df["pos2"].name].shift(1) * df["price2"].shift(1)

        # Avoid division by zero
        adv1 = adv1.replace(0, np.nan).fillna(method="ffill").fillna(1e6)
        adv2 = adv2.replace(0, np.nan).fillna(method="ffill").fillna(1e6)

        df["slip1"] = (
            self.slip_coeff * (abs(df["pos1"] - df["pos1_lag"]) * df["price1"]) 
            / adv1
        )
        df["slip2"] = (
            self.slip_coeff * (abs(df["pos2"] - df["pos2_lag"]) * df["price2"]) 
            / adv2
        )

        df["trans_cost1"] = df["trade1"] * (self.fixed_cost + df["slip1"])
        df["trans_cost2"] = df["trade2"] * (self.fixed_cost + df["slip2"])
        df["total_tc"] = df["trans_cost1"] + df["trans_cost2"]

        # 3) Net P&L
        df["net_pnl"] = df["gross_pnl"] - df["total_tc"]

        # 4) Cumulative returns
        df["strategy_return"] = df["net_pnl"]
        df["cum_return"] = (1 + df["strategy_return"]).cumprod() - 1

        logger.info("Backtest run completed.")
        return df

    def performance_metrics(self, df: pd.DataFrame) -> dict:
        """
        Compute standard metrics: Sharpe, annualized return, max drawdown.
        :return: dict of metrics.
        """
        returns = df["strategy_return"].fillna(0)
        ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

        # Max drawdown
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()

        return {
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        }

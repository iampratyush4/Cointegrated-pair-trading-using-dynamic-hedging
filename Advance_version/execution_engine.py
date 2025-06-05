import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Models realistic slippage based on notional vs. ADV.
    """

    def __init__(self, slippage_coefficient: float):
        """
        :param slippage_coefficient: e.g., 0.0001 (1 bp per 0.1% ADV)
        """
        self.slip_coeff = slippage_coefficient

    def compute_slippage(
        self,
        notional: pd.Series,
        volume: pd.Series,
        price: pd.Series
    ) -> pd.Series:
        """
        Computes slippage cost = slip_coeff * (notional / (ADV * price)).
        :param notional: Series of absolute dollar notional traded.
        :param volume: Series of share volume (ADV proxy).
        :param price:  Series of price to convert volume to ADV notional.
        :return: Series of slippage costs (as fraction of capital).
        """
        # ADV dollar volume
        adv_dollar = volume * price
        adv_dollar = adv_dollar.replace(0, np.nan).fillna(method="ffill").fillna(1e9)
        slip = self.slip_coeff * (notional / adv_dollar)
        return slip

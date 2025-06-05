import pandas as pd
import numpy as np
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskEngine:
    """
    Computes risk metrics for a strategy or portfolio of pair returns.
    Includes VaR (historical and parametric), max drawdown, and 
    placeholder for factor neutrality/regression.
    """

    def __init__(self, returns: pd.Series, config: dict):
        """
        :param returns: Series of daily strategy returns.
        :param config: dict under key 'risk' from config.yaml.
        """
        self.returns = returns.dropna()
        self.daily_var_window = config["daily_var_window"]
        self.var_conf = config["var_confidence"]
        self.max_dd_limit = config["max_drawdown_limit"]

    def historical_var(self) -> float:
        """
        Historical VaR at confidence level.
        """
        window = min(self.daily_var_window, len(self.returns))
        hist = self.returns.tail(window)
        var_h = np.percentile(hist, (1 - self.var_conf) * 100)
        logger.info(f"Historical VaR (window={window}, conf={self.var_conf}) = {var_h:.4%}")
        return var_h

    def parametric_var(self) -> float:
        """
        Parametric VaR assuming normality.
        """
        mean = self.returns.mean()
        std = self.returns.std()
        var_p = mean + std * norm.ppf(1 - self.var_conf)
        logger.info(f"Parametric VaR (conf={self.var_conf}) = {var_p:.4%}")
        return var_p

    def max_drawdown(self) -> float:
        """
        Compute maximum drawdown.
        """
        cum = (1 + self.returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()
        logger.info(f"Maximum drawdown = {max_dd:.4%}")
        return max_dd

    def check_hard_limits(self) -> dict:
        """
        Checks if current drawdown or VaR breaches the configured limits.
        """
        dd = self.max_drawdown()
        var_h = self.historical_var()
        alerts = {}
        if abs(dd) > self.max_dd_limit:
            alerts["drawdown_breach"] = dd
        if var_h < -self.max_dd_limit:
            alerts["var_breach"] = var_h
        return alerts

    def stress_test_returns(self, stress_scenario: pd.Series) -> float:
        """
        Given a stress scenario of returns (Series aligned by date or simply 
        a vector of % shocks), compute expected P&L under that stress.
        For simplicity, sum elementwise product with strategy exposure = 1.
        """
        # This is a placeholder: user should supply a scenario vector.
        pnl = (self.returns * stress_scenario).sum()
        logger.info(f"Stress test scenario P&L = {pnl:.4f}")
        return pnl

    def factor_neutrality(self, factor_returns: pd.DataFrame) -> dict:
        """
        Regress strategy returns on supplied factor returns to compute
        betas and R². Returns a dict of factor exposures.
        """
        import statsmodels.api as sm

        X = sm.add_constant(factor_returns.loc[self.returns.index])
        y = self.returns
        model = sm.OLS(y, X).fit()
        exposures = model.params.to_dict()
        r2 = model.rsquared
        logger.info(f"Factor regression R² = {r2:.4f}, exposures = {exposures}")
        return {"exposures": exposures, "r2": r2}

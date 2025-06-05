import numpy as np
import pandas as pd
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Given a DataFrame of pair‐level returns (columns = pair names, rows = dates),
    solves a minimum‐variance allocation (or any convex objective).
    """

    def __init__(
        self,
        pair_returns: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 0.1
    ):
        """
        :param pair_returns: DataFrame (T×N) of returns for N pairs.
        :param min_weight: lower bound for each weight.
        :param max_weight: upper bound for each weight.
        """
        self.returns = pair_returns.dropna(how="all")
        self.N = self.returns.shape[1]
        self.min_w = min_weight
        self.max_w = max_weight

    def min_variance(self) -> pd.Series:
        """
        Solve: minimize wᵀ Σ w subject to ∑w = 1, and min_w ≤ w_i ≤ max_w.
        Returns a Series of weights indexed by pair names.
        """
        cov = self.returns.cov().values
        w = cp.Variable(self.N)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_w,
            w <= self.max_w
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if w.value is None:
            logger.error("Portfolio optimization failed.")
            # Fallback: equal weights
            w_opt = np.ones(self.N) / self.N
        else:
            w_opt = w.value

        weights = pd.Series(w_opt, index=self.returns.columns)
        logger.info(f"Min‐variance weights computed: {weights.to_dict()}")
        return weights

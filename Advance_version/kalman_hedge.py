import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class KalmanHedge:
    """
    Runs a Kalman filter with an intercept and time‐varying hedge ratio (beta).
    Allows EM estimation of Q and R.
    """

    def __init__(
        self,
        observation_series: pd.Series,
        control_series: pd.Series,
        initial_state_cov: np.ndarray,
        transition_cov: np.ndarray,
        observation_cov: float,
        em_iterations: int = 20
    ):
        """
        :param observation_series: Y_t series (e.g., prices of ticker1).
        :param control_series:     X_t series (e.g., prices of ticker2).
        :param initial_state_cov:  2×2 covariance for [alpha, beta].
        :param transition_cov:     2×2 process noise for [alpha, beta].
        :param observation_cov:    scalar observation noise variance.
        :param em_iterations:      number of EM iterations to refine Q & R.
        """
        self.y = observation_series.values
        self.x = control_series.values
        self.dates = observation_series.index
        self.initial_state_cov = np.array(initial_state_cov)
        self.transition_cov = np.array(transition_cov)
        self.observation_cov = observation_cov
        self.em_iterations = em_iterations

        # Prepare structures to store results
        self.alpha = np.zeros(len(self.y))
        self.beta = np.zeros(len(self.y))
        self.spread = np.zeros(len(self.y))
        self.state_covariances = np.zeros((len(self.y), 2, 2))

        # Build the KalmanFilter object
        self._build_filter()

    def _build_filter(self):
        """
        Initializes a KalmanFilter with a 2D state: [alpha_t, beta_t].
        Observation: y_t = [1, x_t] ⋅ [alpha_t, beta_t] + ε_t
        State Evolution: [alpha_t, beta_t] = [alpha_{t-1}, beta_{t-1}] + η_t
        """
        n_timesteps = len(self.y)
        # Transition matrix: identity (random walk for alpha & beta)
        transition_matrices = np.eye(2)

        # Observation matrices: changes each time because of x_t
        observation_matrices = np.zeros((n_timesteps, 1, 2))
        for t in range(n_timesteps):
            observation_matrices[t, 0, 0] = 1.0       # intercept
            observation_matrices[t, 0, 1] = self.x[t] # slope coefficient

        # Initialize means and covariances
        initial_state_mean = np.zeros(2)
        initial_state_covariance = self.initial_state_cov

        self.kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_covariance=self.transition_cov,
            observation_covariance=self.observation_cov
        )

        # Run EM to refine Q & R
        try:
            self.kf = self.kf.em(
                X=None,  # no endogenous multivariate observations (we pass via filter)
                n_iter=self.em_iterations,
                em_vars=["transition_covariance", "observation_covariance"]
            )
            logger.info("Kalman EM converged: Q and R estimated.")
        except Exception as e:
            logger.warning(f"Kalman EM failed or was skipped: {e}")

    def run_filter(self):
        """
        Executes the predict‐update cycle properly to avoid look‐ahead.
        At each step t:
          1. Predict (alpha_{t|t-1}, beta_{t|t-1})
          2. Compute spread_t = y_t - [alpha_{t|t-1} + beta_{t|t-1} * x_t]
          3. Update with observation y_t
        """
        n = len(self.y)
        # Containers for filtered state
        state_mean = np.zeros((n, 2))
        state_cov = np.zeros((n, 2, 2))

        # Initialize
        state_mean[0] = self.kf.initial_state_mean
        state_cov[0] = self.kf.initial_state_covariance

        for t in range(1, n):
            # 1) Predict step
            mean_pred, cov_pred = self.kf.filter_update(
                filtered_state_mean=state_mean[t - 1],
                filtered_state_covariance=state_cov[t - 1],
                transition_matrices=self.kf.transition_matrices,
                observation=None,
                observation_matrix=None
            )

            # 2) Compute spread at t using predicted state
            a_pred, b_pred = mean_pred
            self.spread[t] = self.y[t] - (a_pred + b_pred * self.x[t])

            # 3) Update step with actual y_t
            mean_filt, cov_filt = self.kf.filter_update(
                filtered_state_mean=mean_pred,
                filtered_state_covariance=cov_pred,
                observation=self.y[t],
                observation_matrix=self.kf.observation_matrices[t]
            )

            state_mean[t] = mean_filt
            state_cov[t] = cov_filt

            # Store alpha/beta
            self.alpha[t] = mean_filt[0]
            self.beta[t] = mean_filt[1]
            self.state_covariances[t] = cov_filt

        # The very first spread uses the initial predicted state
        self.spread[0] = self.y[0] - (state_mean[0][0] + state_mean[0][1] * self.x[0])
        self.alpha[0] = state_mean[0][0]
        self.beta[0] = state_mean[0][1]
        self.state_covariances[0] = state_cov[0]

        # Convert to pandas Series (indexed by dates)
        result = pd.DataFrame({
            "alpha": self.alpha,
            "beta": self.beta,
            "spread": self.spread
        }, index=self.dates)
        logger.info("Kalman filter run completed.")
        return result

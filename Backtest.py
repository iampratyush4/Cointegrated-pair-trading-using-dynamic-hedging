# Import Required Libraries
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# Step 1: Load and Preprocess Data
# ======================================================================

# Define the assets (example: GLD and IAU, two gold ETFs)
symbol1 = "GLD"
symbol2 = "IAU"

# Download historical data using yfinance
data = yf.download([symbol1, symbol2], start="2018-01-01", end="2023-01-01")["Adj Close"]

# Drop missing values and rename columns
data = data.dropna()
data.columns = [symbol1, symbol2]

# Plot prices (optional)
data.plot(title=f"{symbol1} vs {symbol2} Price Series")
plt.show()

# ======================================================================
# Step 2: Cointegration Test (Engle-Granger)
# ======================================================================

# Split data into training (70%) and testing (30%) sets
train_size = int(len(data) * 0.7)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Run cointegration test on training data
score, pvalue, _ = coint(train_data[symbol1], train_data[symbol2])

print(f"Cointegration p-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("Assets are cointegrated (reject null hypothesis)")
else:
    print("Assets are NOT cointegrated")

# ======================================================================
# Step 3: Kalman Filter Setup
# ======================================================================

# Extract training data for Kalman initialization
Y = train_data[symbol1].values  # Dependent variable (e.g., GLD)
X = train_data[symbol2].values  # Independent variable (e.g., IAU)

# Initialize Kalman Filter
kf = KalmanFilter(
    transition_matrices=[1],  # Identity matrix (beta_t = beta_{t-1} + noise)
    observation_matrices=np.ones((1, 1)),  # Observation matrix (Y_t = beta_t * X_t + noise)
    initial_state_mean=np.ones(1),  # Initial hedge ratio guess (beta_0 = 1)
    initial_state_covariance=np.ones((1, 1)),  # Uncertainty in initial state
    observation_covariance=1.0,  # R (measurement noise variance)
    transition_covariance=0.01  # Q (process noise variance)
)

# Fit the Kalman filter to training data to estimate parameters (optional)
# This step uses EM algorithm to optimize Q and R
# kf = kf.em(X.reshape(-1, 1), n_iter=10)

# ======================================================================
# Step 4: Dynamic Hedge Ratio Calculation (Using Kalman Filter)
# ======================================================================

# Initialize arrays to store results
beta = np.ones(len(data))  # Time-varying hedge ratio
spread = np.zeros(len(data))  # Cointegration spread

# Iterate through all data points (train + test)
for t in range(1, len(data)):
    # Get current prices
    y_t = data[symbol1].iloc[t]
    x_t = data[symbol2].iloc[t]

    # Update Kalman filter
    # Predict step: beta_t|t-1 = beta_{t-1}
    beta_pred, cov_pred = kf.filter_update(
        filtered_state_mean=beta[t-1],
        filtered_state_covariance=cov_pred if t > 1 else np.ones(1),
        observation=y_t,
        observation_matrix=np.array([[x_t]]),
        observation_covariance=np.array([[1.0]])
    )

    # Store updated beta and spread
    beta[t] = beta_pred[0]
    spread[t] = y_t - beta[t] * x_t

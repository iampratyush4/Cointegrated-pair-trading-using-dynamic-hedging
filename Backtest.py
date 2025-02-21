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


symbol1 = "WMT"
symbol2 = "TGT"

# Download historical data using yfinance
data = yf.download([symbol1, symbol2], start="2018-01-01", end="2023-01-01")["Close"]

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

# ======================================================================
# Step 5: Generate Trading Signals (Z-Score)
# ======================================================================

# Calculate rolling mean and std of the spread
window = 20  # Lookback period for Z-score
rolling_mean = pd.Series(spread).rolling(window).mean()
rolling_std = pd.Series(spread).rolling(window).std()
z_score = (spread - rolling_mean) / rolling_std

# Define trading rules
entry_threshold = 2.0
exit_threshold = 0.5

# Initialize positions
positions = np.zeros(len(data))  # +1 = long spread, -1 = short spread

for t in range(window, len(data)):
    # Enter long spread (spread is undervalued)
    if z_score[t] < -entry_threshold:
        positions[t] = 1
    # Enter short spread (spread is overvalued)
    elif z_score[t] > entry_threshold:
        positions[t] = -1
    # Exit position when spread crosses zero
    elif np.sign(z_score[t]) != np.sign(z_score[t-1]) and abs(z_score[t]) < exit_threshold:
        positions[t] = 0
    else:
        positions[t] = positions[t-1]  # Hold previous position

# ======================================================================
# Step 6: Backtest Strategy
# ======================================================================

# Calculate daily returns of the spread
spread_returns = np.diff(spread) / spread[:-1]

# Strategy returns (assuming 1:1 capital allocation)
strategy_returns = positions[:-1] * spread_returns

# Cumulative returns
cumulative_returns = np.cumprod(1 + strategy_returns) - 1

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data[symbol1], label=symbol1)
plt.plot(data[symbol2], label=symbol2)
plt.legend()
plt.title("Asset Prices")

plt.subplot(3, 1, 2)
plt.plot(beta, label='Kalman Hedge Ratio')
plt.legend()
plt.title("Dynamic Hedge Ratio (Beta)")

plt.subplot(3, 1, 3)
plt.plot(z_score, label='Z-Score')
plt.axhline(entry_threshold, linestyle='--', color='r')
plt.axhline(-entry_threshold, linestyle='--', color='g')
plt.legend()
plt.title("Z-Score of Spread")

plt.tight_layout()
plt.show()

# Plot cumulative returns
plt.plot(cumulative_returns)
plt.title("Cumulative Strategy Returns")
plt.show()
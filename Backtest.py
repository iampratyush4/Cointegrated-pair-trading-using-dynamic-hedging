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

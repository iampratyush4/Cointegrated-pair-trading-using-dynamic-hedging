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
symbol1 = "V"
symbol2 = "MA"

# Download historical data using yfinance
data = yf.download([symbol1, symbol2], start="2022-01-01", end="2024-03-31")["Close"]

# Drop missing values and rename columns
data = data.dropna()
data.columns = [symbol1, symbol2]

# Plot prices (optional)
plt.figure(figsize=(10, 6))
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
# (Optional) Extract training data for Kalman initialization
Y = train_data[symbol1].values  # Dependent variable
X = train_data[symbol2].values  # Independent variable

# Initialize Kalman Filter
kf = KalmanFilter(
    transition_matrices=[1],               # beta_t = beta_{t-1} + noise
    observation_matrices=np.ones((1, 1)),    # Y_t = beta_t * X_t + noise
    initial_state_mean=np.ones(1),           # Initial hedge ratio guess (beta_0 = 1)
    initial_state_covariance=np.ones((1, 1)),  # Initial uncertainty in beta
    observation_covariance=1.0,              # Measurement noise variance
    transition_covariance=0.01               # Process noise variance
)

# ======================================================================
# Step 4: Dynamic Hedge Ratio Calculation (Using Kalman Filter)
# ======================================================================
# Initialize arrays to store results
beta = np.ones(len(data))     # Time-varying hedge ratio
spread = np.zeros(len(data))    # Cointegration spread

# Initialize covariance prediction variable using the initial covariance
cov_pred = kf.initial_state_covariance

# Iterate through all data points (train + test)
for t in range(1, len(data)):
    y_t = data[symbol1].iloc[t]
    x_t = data[symbol2].iloc[t]

    # Update Kalman filter using the previous beta and covariance
    beta_pred, cov_pred = kf.filter_update(
        filtered_state_mean=beta[t-1],
        filtered_state_covariance=cov_pred,
        observation=y_t,
        observation_matrix=np.array([[x_t]]),
        observation_covariance=np.array([[1.0]])
    )

    beta[t] = beta_pred[0]
    spread[t] = y_t - beta[t] * x_t

# ======================================================================
# Step 5: Generate Trading Signals (Z-Score)
# ======================================================================
window = 20  # Lookback period for rolling statistics
rolling_mean = pd.Series(spread).rolling(window).mean()
rolling_std = pd.Series(spread).rolling(window).std()
z_score = (spread - rolling_mean) / rolling_std

entry_threshold = 1.0
exit_threshold = 0.5

# Initialize positions (1: long spread, -1: short spread, 0: flat)
positions = np.zeros(len(data))

for t in range(window, len(data)):
    if z_score[t] < -entry_threshold:
        positions[t] = 1
    elif z_score[t] > entry_threshold:
        positions[t] = -1
    elif np.sign(z_score[t]) != np.sign(z_score[t-1]) and abs(z_score[t]) < exit_threshold:
        positions[t] = 0
    else:
        positions[t] = positions[t-1]  # Hold previous position

# ======================================================================
# Step 6: Backtest Strategy
# ======================================================================
# Compute daily profit and loss (PnL) directly from the change in spread weighted by the trading position
daily_pnl = positions[:-1] * np.diff(spread)

# Compute cumulative PnL
cumulative_pnl = np.cumsum(daily_pnl)

# ======================================================================
# Step 7: Save Trade Details to CSV
# ======================================================================
# We'll extract rows where a trade occurs (when the position changes)
trade_details = []
# Start from the 'window' index as signals start after enough data is available
for t in range(window, len(data)):
    # Capture the trade if there's a change in position
    if t == window or positions[t] != positions[t-1]:
        trade_details.append({
            "Date": data.index[t],
            f"{symbol1}_Price": data[symbol1].iloc[t],
            f"{symbol2}_Price": data[symbol2].iloc[t],
            "Beta": beta[t],
            "Spread": spread[t],
            "Z-Score": z_score[t],
            "Position": positions[t]
        })

# Convert the list of dictionaries to a DataFrame and save to CSV
trade_df = pd.DataFrame(trade_details)
trade_df.to_csv("trade_details.csv", index=False)
print("Trade details saved to trade_details.csv")

# ======================================================================
# Plot Results
# ======================================================================
plt.figure(figsize=(12, 8))

# Plot asset prices
plt.subplot(3, 1, 1)
plt.plot(data[symbol1], label=symbol1)
plt.plot(data[symbol2], label=symbol2)
plt.legend()
plt.title("Asset Prices")

# Plot dynamic hedge ratio (beta)
plt.subplot(3, 1, 2)
plt.plot(beta, label='Kalman Hedge Ratio')
plt.legend()
plt.title("Dynamic Hedge Ratio (Beta)")

# Plot Z-score of the spread with entry/exit thresholds
plt.subplot(3, 1, 3)
plt.plot(z_score, label='Z-Score')
plt.axhline(entry_threshold, linestyle='--', color='r', label="Entry Threshold")
plt.axhline(-entry_threshold, linestyle='--', color='g', label="Entry Threshold")
plt.legend()
plt.title("Z-Score of Spread")

plt.tight_layout()
plt.show()

# Plot cumulative strategy PnL
plt.figure(figsize=(10, 6))
plt.plot(cumulative_pnl, label="Cumulative PnL")
plt.title("Cumulative Strategy PnL")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL")
plt.legend()
plt.show()

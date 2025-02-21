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

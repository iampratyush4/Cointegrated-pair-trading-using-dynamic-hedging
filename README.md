# Cointegrated Pair Trading Using Dynamic Hedging

## Overview

This project implements a statistical arbitrage strategy known as **cointegrated pair trading** with a focus on **dynamic hedging**. The strategy identifies pairs of assets whose prices move together over time, indicating a cointegrated relationship. By monitoring and trading these pairs, the strategy aims to exploit temporary divergences in their price relationship, profiting from mean-reverting behaviors.

## Key Concepts

- **Cointegration**: A statistical property where two or more time series move together over time, maintaining a consistent mean distance. In trading, cointegrated assets are expected to revert to their mean relationship after deviations.

- **Dynamic Hedging**: An approach that continuously adjusts the hedge ratio between paired assets to account for changing market conditions, ensuring the hedge remains effective over time.

## Project Structure

The repository contains the following key files:

- **`Backtest.py`**: The main script that executes the backtesting of the cointegrated pair trading strategy. It simulates trading scenarios using historical data to evaluate the strategy's performance.

- **`README.md`**: This documentation file provides an overview of the project, its concepts, and instructions for usage.

## How It Works

1. **Data Collection**: Gather historical price data for a set of potential asset pairs.

2. **Cointegration Testing**: For each pair, perform statistical tests (e.g., Engle-Granger or Johansen tests) to determine if the assets are cointegrated.

3. **Model Calibration**: For cointegrated pairs, calibrate a model to understand the relationship between the assets, typically determining the hedge ratio.

4. **Dynamic Hedging Implementation**: Continuously adjust the hedge ratio based on recent data to maintain an effective hedge as market conditions evolve.

5. **Trading Signals Generation**: Monitor the spread between the paired assets. When the spread deviates significantly from the mean (indicating a potential mispricing), generate buy or sell signals.

6. **Backtesting**: Use historical data to simulate trades based on the generated signals, applying transaction costs and slippage to assess the strategy's profitability and risk.

## Usage

To utilize this project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/iampratyush4/Cointegrated-pair-trading-using-dynamic-hedging.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd Cointegrated-pair-trading-using-dynamic-hedging
   ```


3. **Install Required Dependencies**: Ensure you have Python installed along with necessary libraries such as `numpy`, `pandas`, `statsmodels`, and `matplotlib`. You can install them using:

   ```bash
   pip install numpy pandas statsmodels matplotlib
   ```


4. **Prepare Data**: Obtain and format historical price data for the asset pairs you wish to analyze. Ensure the data is in a CSV format with appropriate headers.

5. **Run Backtesting**: Execute the `Backtest.py` script, specifying your data files and parameters as needed. For example:

   ```bash
   python Backtest.py --datafile1 asset1.csv --datafile2 asset2.csv --parameters ...
   ```


   Replace `asset1.csv` and `asset2.csv` with your actual data files and adjust parameters accordingly.

6. **Analyze Results**: Review the output, which may include performance metrics, equity curves, and trade logs, to evaluate the effectiveness of the strategy.

## Notes

- **Data Accuracy**: Ensure that the historical data used is accurate and aligns in terms of timestamps and frequency for both assets in a pair.

- **Parameter Optimization**: Adjust strategy parameters to optimize performance, but be cautious of overfitting to historical data.

- **Risk Management**: Implement appropriate risk management techniques to mitigate potential losses, including setting stop-loss orders and position sizing.
  
---

This README provides a comprehensive overview of the Cointegrated Pair Trading Using Dynamic Hedging project, detailing its purpose, structure, and usage instructions. 

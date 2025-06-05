import pandas as pd
import yfinance as yf
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Fetches and preprocesses price (and volume) data for a given universe.
    Supports daily and intraday via yfinance.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str, interval: str = "1d"):
        """
        :param tickers: List of ticker strings.
        :param start_date: "YYYY-MM-DD"
        :param end_date:   "YYYY-MM-DD"
        :param interval:   "1d", "5m", etc.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Downloads Adj Close and Volume for all tickers between start_date and end_date.
        :return: Tuple (prices_df, volume_df). Both are DataFrames with datetime index.
        """
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date} at interval {self.interval}.")

        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            progress=False
        )

        if raw.empty:
            logger.error("No data fetched. Please check your tickers and date range.")
            raise ValueError("Empty pricing data.")

        # yfinance returns a MultiIndex with (Attribute, Ticker)
        # We extract 'Close' (adjusted) and 'Volume'.
        if "Close" in raw and "Volume" in raw:
            prices = raw["Close"].copy()
            volume = raw["Volume"].copy()
        else:
            # For some intervals, yfinance may label adjusted close as 'Adj Close'
            if "Adj Close" in raw and "Volume" in raw:
                prices = raw["Adj Close"].copy()
                volume = raw["Volume"].copy()
            else:
                logger.error("Unexpected data format from yfinance.")
                raise ValueError("Unexpected data format.")

        # Drop rows where any ticker is missing (to align)
        combined = pd.concat([prices, volume], axis=1, keys=["price", "volume"])
        combined = combined.dropna()
        prices = combined["price"]
        volume = combined["volume"]

        # Ensure columns are sorted alphabetically for consistency
        prices = prices.sort_index(axis=1)
        volume = volume[prices.columns]

        logger.info(f"Downloaded price data with shape {prices.shape}, volume data with shape {volume.shape}.")
        return prices, volume


import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint

# List of ticker pairs
ticker_pairs = [
    ('KO', 'PEP'),      # Coca-Cola & PepsiCo
    ('F', 'GM'),        # Ford & General Motors
    ('XOM', 'CVX'),     # Exxon Mobil & Chevron
    ('WMT', 'TGT'),     # Walmart & Target
    ('HD', 'LOW'),      # Home Depot & Lowe’s
    ('BAC', 'C'),       # Bank of America & Citigroup
    ('JPM', 'GS'),      # JPMorgan Chase & Goldman Sachs
    ('INTC', 'AMD'),    # Intel & Advanced Micro Devices
    ('MCD', 'YUM'),     # McDonald's & Yum! Brands
    ('PG', 'JNJ'),      # Procter & Gamble & Johnson & Johnson
    ('UPS', 'FDX'),     # UPS & FedEx
    ('CAT', 'DE'),      # Caterpillar & Deere & Company
    ('PFE', 'MRK'),     # Pfizer & Merck
    # ('LMT', 'BA'),      # Lockheed Martin & Boeing
    ('V', 'MA'),        # Visa & Mastercard
    ('T', 'VZ'),        # AT&T & Verizon
    ('DAL', 'UAL'),     # Delta Air Lines & United Airlines
    ('SBUX', 'DNKN'),   # Starbucks & Dunkin’ Brands
    ('IBM', 'HPQ'),     # IBM & Hewlett-Packard
    ('DIS', 'CMCSA')    # Walt Disney & Comcast
]

start_date = '2022-01-01'
end_date = '2023-12-31'

best_pair = None
best_pvalue = 1.0

for ticker1, ticker2 in ticker_pairs:
    print(f"\nProcessing pair: {ticker1} & {ticker2}")
    
    # Download adjusted close prices for both tickers at once
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)['Close']
    
    if data.empty:
        print("No data downloaded for this pair.")
        continue
    
    # Drop rows with missing values to ensure dates align.
    df = data.dropna()
    
    if df.empty:
        print("No overlapping data available after dropping NaNs.")
        continue
    
    # Perform the cointegration test.
    series1 = df[ticker1]
    series2 = df[ticker2]
    score, pvalue, _ = coint(series1, series2)
    print(f"Cointegration test p-value: {pvalue:.4f}")
    
    # Track the pair with the lowest p-value.
    if pvalue < best_pvalue:
        best_pvalue = pvalue
        best_pair = (ticker1, ticker2)

if best_pair:
    print(f"\nThe most cointegrated pair from the provided list is {best_pair[0]} and {best_pair[1]} with a p-value of {best_pvalue:.4f}")
else:
    print("No cointegrated pair found among the provided list.")

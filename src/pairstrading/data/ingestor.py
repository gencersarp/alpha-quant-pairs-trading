import pandas as pd
import yfinance as yf

def fetch_pair_data(ticker1: str, ticker2: str, start: str, end: str) -> pd.DataFrame:
    """Fetches aligned adjusted close prices for two assets."""
    data = yf.download([ticker1, ticker2], start=start, end=end, progress=False)["Adj Close"]
    data = data.dropna()
    return data
import yfinance as yf
import pandas as pd
import os

# Ensure the "data" directory exists
os.makedirs("data", exist_ok=True)

def fetch_price_data(ticker, start_date, end_date):
    """
    Fetch historical OHLCV price data and save as CSV.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    filename = f"data/{ticker}_price_data.csv"
    df.to_csv(filename, index=False)
    print(f"[+] Saved price data to {filename}")

def fetch_fundamentals(ticker):
    """
    Fetch basic fundamental info and save as CSV.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    fundamentals = {
        "symbol": ticker,
        "shortName": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "forwardPE": info.get("forwardPE"),
        "trailingPE": info.get("trailingPE"),
        "priceToBook": info.get("priceToBook"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "dividendYield": info.get("dividendYield"),
        "returnOnEquity": info.get("returnOnEquity"),
        "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
        "grossMargins": info.get("grossMargins"),
    }

    df = pd.DataFrame([fundamentals])
    filename = f"data/{ticker}_fundamentals.csv"
    df.to_csv(filename, index=False)
    print(f"[+] Saved fundamentals to {filename}")

def fetch_stock_data(ticker, start_date, end_date):
    """
    Wrapper function that fetches both price and fundamentals.
    """
    print(f"[~] Fetching stock data for {ticker} from {start_date} to {end_date}")
    fetch_price_data(ticker, start_date, end_date)
    fetch_fundamentals(ticker)
    print(f"[✓] All data fetched for {ticker}")

# Optional: for testing standalone
if __name__ == "__main__":
    fetch_stock_data("AAPL", "2023-01-01", "2024-01-01")


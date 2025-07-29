import yfinance as yf
import pandas as pd

def get_fundamentals(ticker_symbol):
    """
    Fetch fundamental data using yfinance.
    Returns a DataFrame with static fundamental features.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    # Extract key fundamentals (you can add more later)
    fundamentals = {
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('trailingPE', 0),
        'forward_pe': info.get('forwardPE', 0),
        'eps': info.get('trailingEps', 0),
        'price_to_book': info.get('priceToBook', 0),
        'dividend_yield': info.get('dividendYield', 0),
        'beta': info.get('beta', 0)
    }

    # Return a DataFrame with these features repeated across the length of your price data
    df = pd.DataFrame([fundamentals])
    return df

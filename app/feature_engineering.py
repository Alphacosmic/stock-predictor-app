import pandas as pd
import pandas_ta as ta
from app.candlestick_patterns import add_candlestick_patterns

def add_technical_indicators(df):
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['macd'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['sma'] = ta.sma(df['Close'], length=14)
    df['ema'] = ta.ema(df['Close'], length=14)
    return df

def enrich_features(df, fundamentals=None, arima_pred=None):
    print("Original df shape:", df.shape)

    df = add_technical_indicators(df)
    df = add_candlestick_patterns(df)

    if fundamentals is not None:
        print("Adding fundamentals:", fundamentals)
        for key, value in fundamentals.items():
            df[key] = value

    if arima_pred is not None:
        print("🔮 Adding ARIMA predictions")
        df['arima_pred'] = arima_pred

    # TEMP: Show missing data before dropping
    print("Nulls before dropna:\n", df.isnull().sum())

    df.dropna(inplace=True)
    print("Enriched feature shape after dropna:", df.shape)
    
    return df

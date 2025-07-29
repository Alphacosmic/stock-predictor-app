import pandas as pd
import ta

def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['Close']).macd()
    df['ema'] = ta.trend.EMAIndicator(close=df['Close'], window=14).ema_indicator()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
    df['roc'] = ta.momentum.ROCIndicator(close=df['Close']).roc()
    df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/AAPL_price_data.csv")  # Make sure this exists
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df.to_csv("data/AAPL_features.csv", index=False)
    print("[+] Saved enriched feature file")

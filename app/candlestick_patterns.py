import pandas as pd
import talib

def add_candlestick_patterns(df):
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']

    candle_functions = {
        'CDL_DOJI': talib.CDLDOJI,
        'CDL_HAMMER': talib.CDLHAMMER,
        'CDL_ENGULFING': talib.CDLENGULFING,
        'CDL_MORNINGSTAR': talib.CDLMORNINGSTAR,
        'CDL_EVENINGSTAR': talib.CDLEVENINGSTAR,
        'CDL_HARAMI': talib.CDLHARAMI,
        'CDL_SHOOTINGSTAR': talib.CDLSHOOTINGSTAR
    }

    for name, func in candle_functions.items():
        try:
            df[name] = func(open_, high, low, close)
        except Exception as e:
            print(f"Failed to apply {name}: {e}")
            df[name] = 0

    return df

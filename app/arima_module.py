import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(df):
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    
    pred = model_fit.predict(start=0, end=len(df) - 1)
    
    print("ARIMA predictions shape:", pred.shape)
    return pred

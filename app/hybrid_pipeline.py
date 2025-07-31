import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

from app.feature_engineering import enrich_features

sns.set(style="whitegrid")

def train_hybrid_model(df):
    df = enrich_features(df)

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("Training XGBoost model...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Training ARIMA model for Close price forecasting...")
    arima_model = ARIMA(df['Close'], order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.predict(start=len(df) - len(y_test), end=len(df) - 1, typ='levels')

    print("ARIMA Forecast (last 5):")
    print(arima_pred.tail(5))
    print("Actual Close Prices (last 5):")
    print(df['Close'].tail(5))

    os.makedirs("outputs", exist_ok=True)
    results_df = pd.DataFrame({
        'Date': df.index[-len(y_test):],
        'Actual_Close': df['Close'].iloc[-len(y_test):].values,
        'Predicted_Trend': y_pred,
        'ARIMA_Forecast': arima_pred.values
    })
    results_df.to_csv("outputs/hybrid_model_results.csv", index=False)
    print("Results saved to outputs/hybrid_model_results.csv")

    # ---------- VISUALIZATION ----------
    print("Generating plots...")

    # 1. ARIMA Forecast vs Actual Close
    plt.figure(figsize=(12, 5))
    plt.plot(results_df['Date'], results_df['Actual_Close'], label='Actual Close', linewidth=2)
    plt.plot(results_df['Date'], results_df['ARIMA_Forecast'], label='ARIMA Forecast', linewidth=2)
    plt.title('ARIMA Forecast vs Actual Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig("outputs/arima_vs_actual.png")
    plt.close()

    # 2. XGBoost Trend Prediction vs Actual Movement
    actual_trend = y_test.reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.plot(actual_trend, label='Actual Trend (Up=1, Down=0)', linestyle='-', marker='o', alpha=0.7)
    plt.plot(y_pred, label='Predicted Trend', linestyle='--', marker='x', alpha=0.7)
    plt.title('XGBoost Predicted Trend vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/xgb_trend_comparison.png")
    plt.close()

    print("📈 Plots saved to outputs/ folder.")


if __name__ == "__main__":
    df = pd.read_csv("data/AAPL_price_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    train_hybrid_model(df)

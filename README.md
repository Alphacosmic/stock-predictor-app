# Stock Predictor App

A hybrid stock prediction tool that combines XGBoost classification with ARIMA time-series forecasting to help visualize and predict short-term trends in stock prices.

## Features

- Upload your own CSV or select from default stock data (like AAPL)
- Visualizes technical indicators and candlestick patterns
- Predicts:
  - Next-day stock trend (bullish/bearish) via XGBoost
  - Closing price forecast via ARIMA
- Downloadable results
- Built with Streamlit for web deployment

## Technologies Used

- Python 3.11+
- pandas, numpy, matplotlib, seaborn
- XGBoost, statsmodels (ARIMA)
- Streamlit
- TA-Lib (via pandas-ta)

## Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app_ui.py
 

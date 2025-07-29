
import streamlit as st
import pandas as pd
import os
from app.hybrid_pipeline import train_hybrid_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Market Trend Predictor (XGBoost + ARIMA)")

# Upload CSV or use default
st.sidebar.header("Upload CSV or use AAPL example")
uploaded_file = st.sidebar.file_uploader("Upload your stock CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use example: AAPL_price_data.csv", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "Uploaded CSV"
elif use_default:
    df = pd.read_csv("data/AAPL_price_data.csv")
    source = "AAPL_price_data.csv"
else:
    st.warning("Please upload a CSV or select the example file.")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Run the hybrid model
with st.spinner(f"Training hybrid model using {source}..."):
    train_hybrid_model(df)

# Load and display results
results_path = "outputs/hybrid_model_results.csv"
if os.path.exists(results_path):
    results = pd.read_csv(results_path)
    results['Date'] = pd.to_datetime(results['Date'])

    st.subheader("📊 ARIMA Forecast vs Actual Close Price")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(results['Date'], results['Actual_Close'], label='Actual Close', color='blue')
    ax1.plot(results['Date'], results['ARIMA_Forecast'], label='ARIMA Forecast', color='orange')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("📈 Predicted Trend (XGBoost)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(results['Date'], results['Predicted_Trend'], drawstyle='steps-post', label='Predicted Trend')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Trend (1=Bullish, 0=Bearish)")
    ax2.set_yticks([0, 1])
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("📄 Output Table")
    st.dataframe(results.tail(10))
else:
    st.error("No results found. Please check if the model ran successfully.")

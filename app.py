import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Trend & Prediction")

# Sidebar for User Inputs
st.sidebar.header("User Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL)", "AAPL")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
predict_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

# Load Data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(symbol, range_val):
    data = yf.download(symbol, period=range_val)
    return data

data = load_data(ticker, period)

if not data.empty:
    # --- DATA VISUALIZATION SECTION ---
    st.subheader(f"Historical Price Trend for {ticker}")
    
    # Calculate Moving Averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
    ax.plot(data.index, data['MA20'], label='20-Day MA', color='orange', linestyle='--')
    ax.plot(data.index, data['MA50'], label='50-Day MA', color='green', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- PREDICTION SECTION ---
    st.divider()
    st.subheader("🤖 Future Price Prediction")

    # Preparing Data for Linear Regression
    df = data[['Close']].reset_index()
    df['Day_Num'] = np.arange(len(df))
    
    X = df[['Day_Num']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)

    # Predict future days
    last_day = df['Day_Num'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1, predict_days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_days)

    # Display Prediction Results
    last_price = df['Close'].iloc[-1]
    predicted_price = future_preds[-1]
    change = predicted_price - last_price

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${last_price:.2f}")
    col2.metric(f"Predicted Price (In {predict_days} days)", f"${predicted_price:.2f}", f"{change:.2f}")

    # Plotting Predictions
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df['Day_Num'], df['Close'], label='Actual Price', color='blue')
    ax2.plot(future_days, future_preds, label='Predicted Trend', color='red', linestyle='--')
    ax2.set_title("Predicted Price Movement")
    ax2.legend()
    st.pyplot(fig2)

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.write(data.tail())
else:
    st.error("Could not fetch data. Please check the Ticker symbol.")

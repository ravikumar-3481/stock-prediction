import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 Real-Time Stock Trend Visualizer & ML Predictor")
st.markdown("Predicting stock trends using Linear Regression and live Yahoo Finance data.")

# Sidebar - User Inputs
st.sidebar.header("Dashboard Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days_to_predict = st.sidebar.slider("Days to Predict into Future", 1, 30, 5)
data_period = st.sidebar.selectbox("History Period", ("1y", "2y", "5y", "max"))

# Fetching Data
@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Fetching live market data...")
data = load_data(ticker_symbol, data_period)
data_load_state.empty()

if data.empty:
    st.error("No data found. Please check the Ticker Symbol (e.g., AAPL, TSLA, GOOGL).")
else:
    # Main Dashboard Metrics
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    price_diff = last_close - prev_close

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${last_close:,.2f}", f"{price_diff:,.2f}")
    col2.metric("52-Week High", f"${data['High'].max():,.2f}")
    col3.metric("52-Week Low", f"${data['Low'].min():,.2f}")

    # --- VISUALIZATION SECTION ---
    st.subheader(f"Historical Price Trend: {ticker_symbol}")
    
    # Adding Technical Indicators (Moving Averages)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price', color='blue', alpha=0.5)
    ax.plot(data['Date'], data['MA20'], label='20-Day MA', color='orange')
    ax.plot(data['Date'], data['MA50'], label='50-Day MA', color='green')
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # --- MACHINE LEARNING SECTION ---
    st.divider()
    st.subheader("🤖 ML Future Price Prediction")

    # Data Preprocessing for Linear Regression
    # We use the ordinal date as the feature (X) and Close price as the label (y)
    df_ml = data[['Date', 'Close']].copy()
    df_ml['Date_Ordinal'] = df_ml['Date'].map(datetime.toordinal)
    
    X = np.array(df_ml['Date_Ordinal']).reshape(-1, 1)
    y = np.array(df_ml['Close']).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    # Creating future dates for prediction
    last_date = df_ml['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_dates_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_dates_ordinal)

    # Display Predictions
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write(f"Forecast for next {days_to_predict} days:")
        st.dataframe(pred_df.style.format({"Predicted Price": "${:,.2f}"}))
    
    with c2:
        # Plotting the Prediction Trend
        fig2, ax2 = plt.subplots()
        ax2.plot(df_ml['Date'].tail(30), df_ml['Close'].tail(30), label='Recent History')
        ax2.plot(pred_df['Date'], pred_df['Predicted Price'], 'ro--', label='Predicted Trend')
        plt.xticks(rotation=45)
        ax2.legend()
        st.pyplot(fig2)

    st.info("Note: Linear Regression assumes a straight-line trend and is meant for educational visualization of momentum, not financial advice.")

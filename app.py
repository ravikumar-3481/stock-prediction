import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide", page_icon="📈")

st.title("📈 Real-Time Stock Trend Visualizer ")
st.markdown("Predicting stock trends using Linear Regression and live Yahoo Finance data.")

# Sidebar - User Inputs
st.sidebar.header("Dashboard Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days_to_predict = st.sidebar.slider("Days to Predict into Future", 1, 90, 5)
data_period = st.sidebar.selectbox("History Period", ("1y", "2y", "5y", "7y", "max"), index=0)

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=3600)  # Cache for 1 hour to prevent API rate limits
def load_data(ticker, period):
    try:
        # Download data
        data = yf.download(ticker, period=period)
        
        if data.empty:
            return None
            
        # FIX: Handle MultiIndex columns in newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Load data with a spinner
with st.spinner(f"Fetching {ticker_symbol} data..."):
    df = load_data(ticker_symbol, data_period)

# --- DASHBOARD LOGIC ---
if df is not None and not df.empty:
    # Ensure 'Close' exists and convert to float to prevent TypeError in st.metric
    try:
        last_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2])
        price_diff = last_close - prev_close
        
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${last_close:,.2f}", f"{price_diff:,.2f}")
        col2.metric("Period High", f"${df['High'].max():,.2f}")
        col3.metric("Period Low", f"${df['Low'].min():,.2f}")

        # --- VISUALIZATION SECTION ---
        st.subheader(f"Historical Price Trend: {ticker_symbol}")
        
        # Calculate Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Close Price', color='#1f77b4', alpha=0.6)
        ax.plot(df['Date'], df['MA20'], label='20-Day MA', color='orange', linestyle='--')
        ax.plot(df['Date'], df['MA50'], label='50-Day MA', color='green', linestyle='--')
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.grid(True, alpha=0.2)
        st.pyplot(fig)

        # --- MACHINE LEARNING SECTION ---
        st.divider()
        st.subheader("🤖 ML Future Price Prediction")

        # Prepare Data for Linear Regression
        df_ml = df[['Date', 'Close']].copy()
        # Convert dates to numbers for the model
        df_ml['Date_Ordinal'] = df_ml['Date'].map(datetime.toordinal)
        
        X = df_ml[['Date_Ordinal']].values
        y = df_ml['Close'].values

        model = LinearRegression()
        model.fit(X, y)

        # Generate future dates
        last_date = df_ml['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_dates_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        
        # Predict
        predictions = model.predict(future_dates_ordinal)

        # Create Prediction DataFrame
        pred_df = pd.DataFrame({
            'Date': future_dates, 
            'Predicted Price': predictions.flatten()
        })
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(f"Forecast for next {days_to_predict} days:")
            st.dataframe(pred_df.style.format({"Predicted Price": "${:,.2f}"}))
        
        with c2:
            # Prediction Visualization
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            # Show last 60 days of history + predictions
            recent_hist = df_ml.tail(60)
            ax2.plot(recent_hist['Date'], recent_hist['Close'], label='Recent History', color='#1f77b4')
            ax2.plot(pred_df['Date'], pred_df['Predicted Price'], 'ro--', label='Predicted Trend')
            plt.xticks(rotation=45)
            ax2.set_title("Predicted Price Movement (Momentum Analysis)")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.write("Debug Info: Verify your CSV/Data structure.")
else:
    st.warning(f"No data found for '{ticker_symbol}'. Please verify the ticker symbol is correct.")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Ravi | AKS University")

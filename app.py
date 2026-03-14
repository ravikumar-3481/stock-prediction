import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockTrend AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FIXED CUSTOM CSS ---
# Changed 'unsafe_all_projects' to 'unsafe_allow_html'
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        # We fetch a bit more data to calculate moving averages properly
        df = stock.history(period=period)
        if df.empty:
            return None, None
        
        # Ensure column names are clean (sometimes yfinance returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df, stock.info
    except Exception:
        return None, None

def perform_prediction(df, days):
    # Prepare data for Linear Regression
    df_ml = df.reset_index()[['Date', 'Close']].copy()
    # Remove timezone info for ordinal conversion
    df_ml['Date'] = pd.to_datetime(df_ml['Date']).dt.tz_localize(None)
    df_ml['Date_Ordinal'] = df_ml['Date'].apply(lambda x: x.toordinal())
    
    X = df_ml[['Date_Ordinal']].values
    y = df_ml['Close'].values

    # Model training
    model = LinearRegression()
    model.fit(X, y)

    # Future dates generation
    last_date = df_ml['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_dates_ordinal)
    
    pred_df = pd.DataFrame({
        'Date': future_dates, 
        'Predicted_Price': predictions.flatten()
    })
    return pred_df

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Navigate to:", ["🏠 Home", "📊 Stock Analysis & Prediction"])

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🚀 StockTrend AI Pro")
    st.subheader("Your Intelligent Market Analysis Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why use this App?
        Predicting the stock market is complex, but visualizing data shouldn't be. This app combines 
        **Real-time Financial Data** with **Linear Regression AI** to provide a clear view of market momentum.
        
        #### 🌟 Key Features:
        - **Live Dashboard:** Real-time metrics for High, Low, and Current prices.
        - **Smart Visuals:** Interactive charts featuring 20-day and 50-day Moving Averages.
        - **Full History:** Access to the last 30 days of trading records.
        - **Future Forecast:** Predict where the stock is headed based on historical price action.
        """)
        
        st.info("💡 **Ready to start?** Switch to the 'Stock Analysis' page in the sidebar and enter a ticker like **AAPL, TSLA, or MSFT**.")

    with col2:
        st.success("""
        #### ⚙️ How it Works:
        1. **Data:** Fetched via YFinance API.
        2. **Process:** Data is cleaned and smoothed.
        3. **AI:** Linear Regression analyzes the trend line.
        4. **Result:** Forecasted prices for your selected period.
        """)
    
    st.divider()
    st.markdown("Developed by **Ravi** | AKS University")

# --- ANALYSIS & PREDICTION PAGE ---
elif page == "📊 Stock Analysis & Prediction":
    st.title("Market Analysis & ML Prediction")
    
    # User Inputs
    st.sidebar.header("Search Parameters")
    ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper().strip()
    data_horizon = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "max"])
    
    if ticker_input:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            df, info = fetch_stock_data(ticker_input, data_horizon)
            
        if df is not None and not df.empty:
            # 1. Dashboard Metrics
            # Ensure values are float to avoid st.metric errors
            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Last Price", f"${last_price:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
            with m2:
                st.metric("Period High", f"${float(df['High'].max()):,.2f}")
            with m3:
                st.metric("Period Low", f"${float(df['Low'].min()):,.2f}")

            # 2. Tabs for visualization and table
            tab1, tab2 = st.tabs(["📊 Interactive Trend Chart", "📜 Last 30 Days History"])
            
            with tab1:
                st.subheader(f"Price Trend: {ticker_input}")
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='#007bff', width=2)))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20D MA (Fast)", line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="50D MA (Slow)", line=dict(dash='dot', color='green')))
                
                fig.update_layout(
                    height=450, 
                    template="plotly_white", 
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Historical Data Table")
                # Show only 1 month of history as requested
                recent_data = df.tail(30).iloc[::-1] # Reverse to show newest first
                st.dataframe(recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"), use_container_width=True)

            # 3. Prediction Module
            st.divider()
            st.header("🤖 AI Forecast Analysis")
            
            p1, p2 = st.columns([1, 2])
            
            with p1:
                st.write("Predict future prices based on linear trend analysis.")
                forecast_days = st.slider("Prediction Horizon (Days)", 1, 90, 15)
                run_ml = st.button("🔮 Calculate Forecast")
            
            if run_ml:
                with st.spinner("Training Model..."):
                    pred_results = perform_prediction(df, forecast_days)
                    
                    with p2:
                        # Visualization
                        fig_ml, ax_ml = plt.subplots(figsize=(10, 5))
                        # Use last 60 days of real data for context
                        context = df.tail(60)
                        ax_ml.plot(context.index.tz_localize(None), context['Close'], label='Actual History', color='#007bff', linewidth=2)
                        ax_ml.plot(pred_results['Date'], pred_results['Predicted_Price'], 'ro--', label='Predicted Path', markersize=4, alpha=0.7)
                        
                        ax_ml.set_title(f"Momentum Forecast for {ticker_input}")
                        ax_ml.legend()
                        plt.xticks(rotation=45)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig_ml)
                    
                    st.success(f"Forecast for next {forecast_days} days generated successfully!")
                    st.dataframe(pred_results.style.format({"Predicted_Price": "${:,.2f}"}), use_container_width=True)
        else:
            st.warning(f"Unable to find data for '{ticker_input}'. Please check the ticker symbol.")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Ravi Kumar | StockTrend AI")

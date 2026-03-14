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

# --- CUSTOM CSS FOR STYLING ---
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
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_all_projects=True)

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None
        return df, stock.info
    except Exception as e:
        return None, None

def perform_prediction(df, days):
    # Prepare data
    df_ml = df.reset_index()[['Date', 'Close']]
    df_ml['Date_Ordinal'] = df_ml['Date'].map(datetime.toordinal)
    
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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Stock Analysis & Prediction"])

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🚀 Welcome to StockTrend AI Pro")
    st.markdown("### Your Intelligent Companion for Market Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        This professional-grade web application leverages **Machine Learning** and **Real-time Market Data** to help investors visualize trends and forecast future stock movements.
        
        #### ✨ Core Features:
        - **Live Data Integration:** Fetches real-time data directly from Yahoo Finance.
        - **Trend Visualization:** Interactive charts with Moving Averages (20-day & 50-day).
        - **Deep History:** View detailed tabular data for the last month.
        - **ML Forecasting:** Uses Linear Regression to predict price trajectories based on historical momentum.
        """)
        
        st.info("💡 **How to use:** Navigate to the 'Stock Analysis' page using the sidebar, enter a stock symbol (like AAPL or TSLA), and explore the data!")

    with col2:
        st.success("""
        #### 🛠️ Tech Stack:
        - **Streamlit** (UI Framework)
        - **YFinance** (Data Source)
        - **Scikit-Learn** (Prediction Model)
        - **Plotly & Matplotlib** (Charts)
        """)
    
    if st.button("Get Started →"):
        st.info("Please select 'Stock Analysis' from the sidebar to begin.")

# --- ANALYSIS & PREDICTION PAGE ---
elif page == "📊 Stock Analysis & Prediction":
    st.title("Market Analysis Dashboard")
    
    # User Inputs in Sidebar
    st.sidebar.header("Search Parameters")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
    data_period = st.sidebar.selectbox("Analysis Horizon", ["1y", "2y", "5y", "max"])
    
    if ticker_symbol:
        with st.spinner(f"Analyzing {ticker_symbol}..."):
            df, info = fetch_stock_data(ticker_symbol, data_period)
            
        if df is not None:
            # 1. Top Level Metrics
            last_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Current Price", f"${last_price:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
            with m2:
                st.metric("52 Week High", f"${df['High'].max():,.2f}")
            with m3:
                st.metric("52 Week Low", f"${df['Low'].min():,.2f}")

            # 2. Historical Data Tabs
            tab1, tab2 = st.tabs(["📈 Price Chart", "📋 1-Month History"])
            
            with tab1:
                st.subheader("Historical Trend with Moving Averages")
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price", line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20D MA", line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="50D MA", line=dict(dash='dot', color='green')))
                fig.update_layout(height=500, template="plotly_white", xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Last 30 Days Record")
                last_month = df.tail(30).sort_index(ascending=False)
                st.dataframe(last_month[['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"))

            # 3. Prediction Section
            st.divider()
            st.header("🤖 Artificial Intelligence Forecasting")
            
            p_col1, p_col2 = st.columns([1, 2])
            
            with p_col1:
                st.write("Configure your prediction model settings below.")
                days = st.number_input("Days to Forecast", min_value=1, max_value=365, value=30)
                predict_btn = st.button("🔮 Run Prediction Model")
            
            if predict_btn:
                with st.spinner("Calculating future trends..."):
                    pred_df = perform_prediction(df, days)
                    
                    with p_col2:
                        # Visualization of Prediction
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                        # Plot actual last 90 days
                        recent = df.tail(90)
                        ax_pred.plot(recent.index, recent['Close'], label='Historical', color='#1f77b4', linewidth=2)
                        # Plot prediction
                        ax_pred.plot(pred_df['Date'], pred_df['Predicted_Price'], '--', label='Predicted', color='red', marker='o', markersize=4)
                        
                        ax_pred.set_title(f"{ticker_symbol} Future Forecast")
                        ax_pred.legend()
                        plt.grid(alpha=0.3)
                        st.pyplot(fig_pred)
                    
                    st.success(f"Successfully generated forecast for {days} days.")
                    st.table(pred_df.head(10)) # Show first 10 predicted days
        else:
            st.error(f"Ticker '{ticker_symbol}' not found. Please check the symbol (e.g., AAPL, MSFT, BTC-USD).")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Developed by Ravi | AKS University\nPowered by Streamlit & Scikit-Learn")

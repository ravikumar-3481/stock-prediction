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
    layout="wide"
)

# --- APP STATE MANAGEMENT ---
# Using session_state to navigate between Home and Dashboard without Sidebar
if 'page' not in st.session_state:
    st.session_state.page = "home"

def go_to_analysis():
    st.session_state.page = "analysis"

def go_to_home():
    st.session_state.page = "home"

# --- FIXED CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        transition: 0.3s;
    }
    .hero-text {
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    /* Hide Sidebar for cleaner mobile look */
    [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING FUNCTIONS ---
@st.cache_data(ttl=600) # Cache for 10 mins
def fetch_stock_data(ticker):
    try:
        # We use a fixed 1y period for analysis to ensure stability
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df is None or df.empty:
            return None, "No data returned from Yahoo Finance."
        
        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df, None
    except Exception as e:
        return None, str(e)

def perform_prediction(df, days):
    df_ml = df.reset_index()[['Date', 'Close']].copy()
    df_ml['Date'] = pd.to_datetime(df_ml['Date']).dt.tz_localize(None)
    df_ml['Date_Ordinal'] = df_ml['Date'].apply(lambda x: x.toordinal())
    
    X = df_ml[['Date_Ordinal']].values
    y = df_ml['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = df_ml['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    predictions = model.predict(future_dates_ordinal)
    
    return pd.DataFrame({
        'Date': future_dates, 
        'Predicted_Price': predictions.flatten()
    })

# --- PAGE: HOME ---
if st.session_state.page == "home":
    st.markdown("<div class='hero-text'>", unsafe_allow_html=True)
    st.title("📈 StockTrend AI Pro")
    st.subheader("Advanced Stock Forecasting & Market Analysis")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Future of Trading Analysis
        This platform provides users with real-time financial insights and predictive modeling to help 
        understand market trends better.
        
        #### 🚀 Key Features:
        * **Real-time Data:** Instant access to global stock markets.
        * **Interactive Analytics:** Visualise price movements with technical indicators.
        * **Machine Learning:** Predict future price trends using Linear Regression.
        * **Historical Review:** Deep dive into 1-month historical price logs.
        
        #### 📖 How to Use:
        1. Click the **Launch Dashboard** button below.
        2. Enter a valid stock ticker (e.g., `AAPL` for Apple, `TSLA` for Tesla).
        3. Review the historical charts and price metrics.
        4. Select a forecast duration and hit **Predict** to see future trends.
        """)
        
        st.button("🚀 Launch Analysis Dashboard", on_click=go_to_analysis, type="primary")

    with col2:
        st.info("💡 **Pro Tip:** For Indian stocks, use the `.NS` suffix (e.g., `RELIANCE.NS`). For Crypto, use `-USD` (e.g., `BTC-USD`).")
        st.success("""
        **System Status:**
        - API Connection: Online ✅
        - ML Engine: Ready ✅
        - Data Source: Yahoo Finance
        """)

# --- PAGE: ANALYSIS ---
elif st.session_state.page == "analysis":
    # Top Navigation Bar
    n1, n2 = st.columns([8, 2])
    with n1:
        st.title("📊 Market Analysis Dashboard")
    with n2:
        st.button("🏠 Back to Home", on_click=go_to_home)
    
    # Input Area
    input_col1, input_col2 = st.columns([3, 1])
    with input_col1:
        ticker_symbol = st.text_input("Enter Stock Ticker Symbol", value="AAPL", help="Enter symbols like AAPL, MSFT, or GOOGL").upper().strip()
    with input_col2:
        st.write(" ") # Padding
        refresh = st.button("🔄 Refresh Data")

    if ticker_symbol:
        df, error_msg = fetch_stock_data(ticker_symbol)
        
        if df is not None:
            # Current Price Metrics
            last_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${last_price:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
            m2.metric("Day High", f"${float(df['High'].iloc[-1]):,.2f}")
            m3.metric("Day Low", f"${float(df['Low'].iloc[-1]):,.2f}")
            m4.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")

            # Visualization Tabs
            t1, t2 = st.tabs(["📉 Price Trend Graph", "📅 Historical List (30 Days)"])
            
            with t1:
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color='#007bff', width=2.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20D MA", line=dict(dash='dash', color='#ffa500')))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="50D MA", line=dict(dash='dot', color='#28a745')))
                
                fig.update_layout(height=500, template="plotly_white", margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                st.subheader("Last 30 Trading Days")
                history_30 = df.tail(30).iloc[::-1] # Newest first
                st.dataframe(history_30[['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"), use_container_width=True)

            # Prediction Section
            st.divider()
            st.header("🤖 AI Price Prediction")
            
            pred_col1, pred_col2 = st.columns([1, 2])
            with pred_col1:
                st.write("Select the number of days you want to forecast into the future.")
                days_to_predict = st.slider("Forecast Days", 1, 60, 15)
                predict_trigger = st.button("🔮 Predict Future Price", type="primary")

            if predict_trigger:
                with st.spinner("Analyzing market momentum..."):
                    pred_df = perform_prediction(df, days_to_predict)
                    
                    with pred_col2:
                        # Prediction Chart
                        fig_ml, ax_ml = plt.subplots(figsize=(10, 5))
                        hist_context = df.tail(60)
                        ax_ml.plot(hist_context.index.tz_localize(None), hist_context['Close'], label='Recent History', color='#007bff', linewidth=2)
                        ax_ml.plot(pred_df['Date'], pred_df['Predicted_Price'], 'ro--', label='AI Forecast', markersize=4)
                        ax_ml.set_title(f"Forecast for {ticker_symbol}")
                        ax_ml.legend()
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig_ml)
                    
                    st.success(f"Prediction for {days_to_predict} days completed!")
                    st.dataframe(pred_df.style.format({"Predicted_Price": "${:,.2f}"}), use_container_width=True)

        else:
            st.error(f"❌ Error: {error_msg}")
            st.info("Check if the ticker is correct. Example: AAPL, TSLA, or RELIANCE.NS")

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed by Ravi | AKS University | © 2026 StockTrend AI</div>", unsafe_allow_html=True)

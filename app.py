import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockTrend AI Pro",
    page_icon="📈",
    layout="wide"
)

# --- SESSION STATE FOR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = "home"

def nav_to(page_name):
    st.session_state.page = page_name

# --- STYLING (REMOVED CUSTOM BG, RESTORED DEFAULT) ---
st.markdown("""
    <style>
    .main-header { 
        text-align: center; 
        padding: 30px; 
        background: #007bff; 
        color: white; 
        border-radius: 12px; 
        margin-bottom: 25px; 
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        font-weight: bold; 
    }
    /* Hide Sidebar */
    [data-testid="stSidebar"] { display: none; } 
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST ANALYTICS FUNCTIONS ---
@st.cache_data(ttl=600)
def get_full_stock_info(ticker):
    try:
        # Step 1: Get History
        # We use a 1-year period as default for better reliability
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if data.empty:
            return None, None, "No historical data found for this ticker."
            
        # Fix MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Step 2: Get Info (Rate limiting usually happens here)
        t_obj = yf.Ticker(ticker)
        try:
            info = t_obj.info
        except:
            info = {"longName": ticker, "sector": "N/A", "industry": "N/A", "longBusinessSummary": "Detailed info temporarily unavailable due to API rate limits."}

        # Calculate Technical Indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data, info, None
    except Exception as e:
        if "Rate limited" in str(e) or "429" in str(e):
            return None, None, "Yahoo Finance Rate Limit: Please wait 1-2 minutes and try again."
        return None, None, f"Error: {str(e)}"

def ml_predict(df, days):
    df_ml = df.reset_index()[['Date', 'Close']].copy()
    # Handle timezone and convert to ordinal
    df_ml['Date'] = pd.to_datetime(df_ml['Date']).dt.tz_localize(None)
    df_ml['Ordinal'] = df_ml['Date'].apply(lambda x: x.toordinal())
    
    model = LinearRegression()
    model.fit(df_ml[['Ordinal']].values, df_ml['Close'].values)
    
    last_date = df_ml['Date'].iloc[-1]
    f_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    f_ordinals = np.array([d.toordinal() for d in f_dates]).reshape(-1, 1)
    preds = model.predict(f_ordinals)
    
    return pd.DataFrame({'Date': f_dates, 'Price': preds})

# --- HOME SCREEN ---
if st.session_state.page == "home":
    st.markdown("<div class='main-header'><h1>📈 StockTrend AI Pro</h1><p>Intelligent Market Analytics Terminal</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Professional Stock Analysis Platform
        Access real-time data and AI-powered predictions for global markets. 
        Designed for clarity, speed, and accuracy.
        
        #### ✨ App Highlights:
        - **Live Market Data:** Powered by Yahoo Finance.
        - **Comprehensive Charts:** Price, Volume, RSI, and MACD indicators.
        - **AI Forecasting:** Trend projection using Linear Regression.
        - **Company Profiles:** In-depth business summaries and sector details.
        """)
        st.button("🎯 Launch Analysis Terminal", on_click=lambda: nav_to("analysis"), type="primary")
    
    with col2:
        st.info("**Instructions:** Use standard symbols like `AAPL`, `MSFT`. For Indian stocks, add `.NS` (e.g., `TATASTEEL.NS`).")
        st.write("---")
        st.caption("Developed by Ravi Kumar | AKS University")

# --- ANALYSIS SCREEN ---
elif st.session_state.page == "analysis":
    # Top Navigation
    nav1, nav2 = st.columns([8, 2])
    with nav1: st.title("📊 Analysis Terminal")
    with nav2: st.button("🏠 Back to Home", on_click=lambda: nav_to("home"))
    
    # Input Area
    s1, s2 = st.columns([4, 1])
    with s1:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="Example: TSLA, GOOGL, RELIANCE.NS").upper().strip()
    with s2:
        st.write("") # Padding
        st.button("🔄 Refresh")

    if ticker:
        with st.spinner(f"Requesting data for {ticker}..."):
            df, info, err = get_full_stock_info(ticker)
        
        if err:
            st.error(f"⚠️ {err}")
            if "Rate Limit" in err:
                st.warning("Streaming finance APIs often limit requests when accessed from cloud servers. Try refreshing in 30 seconds.")
        elif df is not None:
            # 1. Company Header
            full_name = info.get('longName', ticker)
            st.header(f"{full_name} ({ticker})")
            
            # 2. Market Metrics
            l_price = float(df['Close'].iloc[-1])
            prev_p = float(df['Close'].iloc[-2])
            change = l_price - prev_p
            pct = (change / prev_p) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${l_price:,.2f}", f"{change:+.2f} ({pct:+.2f}%)")
            m2.metric("Market Cap", info.get('marketCap', 'N/A'))
            m3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            m4.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):,.2f}")
            
            # 3. Graphing Suite
            st.divider()
            st.subheader("📈 Technical Charts")
            
            # Main Price & Volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), name="20D MA", line=dict(color="orange", dash='dash')), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(0,123,255,0.2)"), row=2, col=1)
            fig.update_layout(height=500, template="plotly_white", margin=dict(t=10, b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI & MACD
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Relative Strength Index (RSI)")
                f_rsi = go.Figure()
                f_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="purple")))
                f_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                f_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                f_rsi.update_layout(height=250, template="plotly_white", margin=dict(t=0,b=0))
                st.plotly_chart(f_rsi, use_container_width=True)
            with g2:
                st.caption("MACD Momentum")
                f_macd = go.Figure()
                f_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
                f_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="orange")))
                f_macd.update_layout(height=250, template="plotly_white", margin=dict(t=0,b=0))
                st.plotly_chart(f_macd, use_container_width=True)

            # 4. Details Tabs
            tab1, tab2 = st.tabs(["📄 Company Profile", "📅 30-Day Historical Data"])
            with tab1:
                st.markdown(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
                st.write(info.get('longBusinessSummary', "No detailed summary available at this time."))
            with tab2:
                recent = df.tail(30).iloc[::-1]
                st.dataframe(recent[['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"), use_container_width=True)

            # 5. Prediction Engine
            st.divider()
            st.subheader("🤖 AI Trend Prediction")
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.write("Forecast the linear trajectory of the stock price.")
                days_to_pred = st.slider("Select Forecast Horizon (Days)", 1, 90, 30)
                if st.button("🔮 Run Forecast Model", type="primary"):
                    with st.spinner("Analyzing momentum..."):
                        preds = ml_predict(df, days_to_pred)
                        with p_col2:
                            fig_p, ax_p = plt.subplots(figsize=(10, 5))
                            # Plot last 60 days
                            ctx = df.tail(60)
                            ax_p.plot(ctx.index.tz_localize(None), ctx['Close'], label='History', color='#007bff')
                            ax_p.plot(preds['Date'], preds['Price'], 'ro--', label='Predicted Trend', markersize=4)
                            ax_p.set_title("ML Prediction Graph")
                            ax_p.legend()
                            st.pyplot(fig_p)
                        st.write("### Predicted Prices")
                        st.dataframe(preds.style.format({"Price": "${:,.2f}"}), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<center><small>StockTrend AI Pro v2.1 | Data: Yahoo Finance | Dev: Ravi Kumar</small></center>", unsafe_allow_html=True)

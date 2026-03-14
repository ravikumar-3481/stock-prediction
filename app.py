import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

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

# --- STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    .main-header { text-align: center; padding: 40px 0; background: linear-gradient(90deg, #007bff, #6610f2); color: white; border-radius: 15px; margin-bottom: 30px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; font-weight: bold; transition: all 0.3s; }
    .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #eee; }
    [data-testid="stSidebar"] { display: none; } /* Hide Sidebar */
    </style>
    """, unsafe_allow_html=True)

# --- ANALYTICS FUNCTIONS ---
@st.cache_data(ttl=600)
def get_full_stock_info(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        if df.empty: return None, None, "No data found."
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1+rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df, t.info, None
    except Exception as e:
        return None, None, str(e)

def ml_predict(df, days):
    df_ml = df.reset_index()[['Date', 'Close']].copy()
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
    st.markdown("<div class='main-header'><h1>📈 StockTrend AI Pro</h1><p>Professional Market Analysis & Future Forecasting</p></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### 🚀 Welcome to the Pro Terminal
        This web application is designed for traders and analysts who need quick, data-driven insights. 
        It uses **Yahoo Finance** for real-time market data and **Scikit-Learn** for trend forecasting.
        
        #### 🛠️ Key Features:
        - **Company Profile:** Auto-detects company name, sector, and business summary.
        - **Technical Suite:** Interactive Price, Volume, RSI, and MACD charts.
        - **ML Engine:** One-click Linear Regression prediction for up to 90 days.
        - **Data Integrity:** Detailed 30-day historical transaction logs.
        """)
        st.button("🎯 Open Analysis Dashboard", on_click=lambda: nav_to("analysis"), type="primary")
    
    with c2:
        st.info("💡 **Supported Markets:** USA (AAPL), India (RELIANCE.NS), Crypto (BTC-USD), and more.")
        st.markdown("""
        <div class='card'>
        <h4>System Specs</h4>
        <hr>
        <li>Refresh Rate: 10 Mins</li>
        <li>ML Model: Linear Reg</li>
        <li>Data: Real-time API</li>
        </div>
        """, unsafe_allow_html=True)

# --- ANALYSIS SCREEN ---
elif st.session_state.page == "analysis":
    # Top Nav
    h1, h2 = st.columns([8, 2])
    with h1: st.title("📊 Market Terminal")
    with h2: st.button("🏠 Home", on_click=lambda: nav_to("home"))
    
    # Search Bar
    s1, s2 = st.columns([4, 1])
    with s1: 
        ticker = st.text_input("Search Stock Ticker Symbol", value="AAPL").upper().strip()
    with s2:
        st.write("") # Spacer
        search_btn = st.button("🔍 Search")

    if ticker:
        df, info, err = get_full_stock_info(ticker)
        
        if df is not None:
            # HEADER SECTION
            comp_name = info.get('longName', ticker)
            st.header(f"{comp_name} ({ticker})")
            
            # QUICK METRICS
            last_p = float(df['Close'].iloc[-1])
            change = last_p - float(df['Close'].iloc[-2])
            pct = (change / float(df['Close'].iloc[-2])) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${last_p:,.2f}", f"{change:+.2f} ({pct:+.2f}%)")
            m2.metric("Market Cap", info.get('marketCap', 'N/A'))
            m3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            m4.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):,.2f}")
            
            # GRAPHS SECTION
            st.divider()
            st.subheader("📈 Technical Analysis Suite")
            
            # 1. Price & Volume (Combined)
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff")), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), name="20D MA", line=dict(dash='dash')), row=1, col=1)
            fig1.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(0,123,255,0.3)"), row=2, col=1)
            fig1.update_layout(height=500, template="plotly_white", margin=dict(t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. RSI & MACD
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Relative Strength Index (RSI)")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="purple")))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(height=250, template="plotly_white", margin=dict(t=0, b=0))
                st.plotly_chart(fig_rsi, use_container_width=True)
            with g2:
                st.caption("MACD Momentum")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="orange")))
                fig_macd.update_layout(height=250, template="plotly_white", margin=dict(t=0, b=0))
                st.plotly_chart(fig_macd, use_container_width=True)

            # COMPANY INFO & HISTORY TABS
            t1, t2 = st.tabs(["📄 Company Profile", "📅 30-Day Historical Log"])
            with t1:
                st.markdown(f"""
                **Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}
                
                **Summary:** {info.get('longBusinessSummary', 'No description available.')}
                """)
            with t2:
                st.dataframe(df.tail(30).iloc[::-1][['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"), use_container_width=True)

            # PREDICTION SECTION
            st.divider()
            st.subheader("🤖 AI Price Prediction (Linear Regression)")
            p1, p2 = st.columns([1, 2])
            with p1:
                st.write("Determine the projected price based on linear growth analysis.")
                days = st.slider("Select Prediction Days", 1, 90, 30)
                btn = st.button("🔮 Run AI Prediction", type="primary")
            
            if btn:
                with st.spinner("Calculating Trend..."):
                    pdf = ml_predict(df, days)
                    with p2:
                        fig_p, ax_p = plt.subplots(figsize=(10, 5))
                        ax_p.plot(df.tail(60).index.tz_localize(None), df.tail(60)['Close'], label='History', color='#007bff')
                        ax_p.plot(pdf['Date'], pdf['Price'], 'ro--', label='Predicted', markersize=4)
                        ax_p.set_title("Future Momentum Forecast")
                        ax_p.legend()
                        st.pyplot(fig_p)
                    st.dataframe(pdf.style.format({"Price": "${:,.2f}"}), use_container_width=True)

        else:
            st.error(f"Error fetching '{ticker}': {err}")

# FOOTER
st.markdown("---")
st.markdown("<center><small>Developed by Ravi Kumar | StockTrend AI v2.0 | Powered by Streamlit</small></center>", unsafe_allow_html=True)

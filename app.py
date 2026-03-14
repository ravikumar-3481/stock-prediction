import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockTrend AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- APP STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = "home"

def nav_to(page_name):
    st.session_state.page = page_name

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    /* Hide Sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Tech Tags styling */
    .tech-tag {
        display: inline-block;
        background: rgba(128, 128, 128, 0.1);
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Social Buttons styling */
    .social-list {
        list-style: none;
        padding: 0;
        display: flex;
        gap: 15px;
        margin-top: 15px;
    }
    .social-btn {
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        color: white !important;
        transition: opacity 0.2s;
    }
    .social-btn:hover { opacity: 0.8; }
    .github { background: #333; }
    .linkedin { background: #0077b5; }
    .portfolio { background: #e84393; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE (UNTOUCHED) ---
@st.cache_data(ttl=600)
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None, "Ticker not found or no historical data available."
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Indicators
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        data['20 SMA'] = data['Close'].rolling(window=20).mean()
        data['20 STD'] = data['Close'].rolling(window=20).std()
        data['Upper Band'] = data['20 SMA'] + (data['20 STD'] * 2)
        data['Lower Band'] = data['20 SMA'] - (data['20 STD'] * 2)
        
        data.dropna(inplace=True)
        return data, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def perform_ml(df, days, model_type):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        preds = model.predict(future_X)
    elif model_type == "Polynomial Regression (Deg 2)":
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X, y)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        preds = model.predict(future_X)
    elif model_type == "Support Vector Regression (SVR)":
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_scaled, y_scaled)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        future_X_scaled = scaler_X.transform(future_X)
        preds_scaled = model.predict(future_X_scaled)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    
    last_date = df.index[-1]
    f_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return pd.DataFrame({'Date': f_dates, 'Price': preds})

# --- PAGE: HOME (STANDARD LAYOUT) ---
if st.session_state.page == "home":
    # Hero Title
    st.title("🚀 StockTrend AI Pro")
    st.subheader("Next-Generation Market Intelligence & Predictive Analytics")
    st.divider()

    # Main Grid for Project Information
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.header("Project Overview")
        st.write("""
        StockTrend AI Pro is an advanced financial monitoring terminal designed to bridge the gap between complex 
        quantitative analysis and individual retail investors. By leveraging real-time market data APIs and 
        machine learning algorithms, the platform transforms raw historical price action into actionable 
        visual intelligence.
        """)

        st.header("The Challenge: Data Overload")
        st.write("""
        Modern financial markets generate billions of data points. Retail traders often face:
        * **Information Asymmetry:** Lack of institutional-grade AI models.
        * **Technical Complexity:** Difficulty in manual technical indicator calculation.
        * **Predictive Uncertainty:** Moving averages only show the past, not the future.
        """)

        st.header("The Solution")
        st.write("""
        This project provides a centralized, automated analysis pipeline. It fetches live data from global 
        exchanges, performs technical computations, and applies regression models to project future trends, 
        allowing users to focus on strategy rather than math.
        """)

    with col2:
        st.header("Unique Value Proposition")
        st.write("""
        * **Hybrid Analytics:** Combines traditional indicators with modern ML.
        * **Dynamic Forecasts:** Customize prediction windows from 1 to 90 days.
        * **Algorithm Comparison:** Compare Linear, Polynomial, and SVR models.
        * **Reactive Performance:** High-speed rendering for real-time decision making.
        """)

        st.header("Technology Stack")
        st.markdown("""
            <span class="tech-tag">Python 3.10</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">Scikit-Learn</span>
            <span class="tech-tag">YFinance API</span>
            <span class="tech-tag">Plotly</span>
            <span class="tech-tag">Pandas</span>
            <span class="tech-tag">NumPy</span>
        """, unsafe_allow_html=True)

        st.header("How to Use")
        st.write("""
        1. Click the **Get Started** button below.
        2. Enter a symbol like **AAPL**, **TSLA**, or **BTC-USD**.
        3. Analyze the technical charts and volume data.
        4. Select an AI model to forecast future price movements.
        """)

    # CTA Button
    st.write("---")
    st.button("🎯 Get Started - Open Analysis Terminal", on_click=lambda: nav_to("analysis"), type="primary", use_container_width=True)

    # Developer Section at the bottom
    st.write("---")
    dev_col1, dev_col2 = st.columns([1, 2])
    with dev_col1:
        st.header("About the Developer")
    with dev_col2:
        st.write("**Ravi Kumar Vishwakarma**")
        st.write("Computer Science Engineering Student | AKS University")
        st.write("Specializing in AI and Data Engineering.")
        st.markdown("""
            <div class="social-list">
                <a href="https://github.com/ravikumar-3481" target="_blank" class="social-btn github">GitHub</a>
                <a href="https://www.linkedin.com/in/ravi-vishwakarma67" target="_blank" class="social-btn linkedin">LinkedIn</a>
                <a href="https://profileravi.netlify.app" target="_blank" class="social-btn portfolio">Portfolio</a>
            </div>
        """, unsafe_allow_html=True)

# --- PAGE: ANALYSIS (UNTOUCHED LOGIC) ---
elif st.session_state.page == "analysis":
    n1, n2 = st.columns([9, 1])
    n1.title("📈 Analysis Terminal")
    n2.button("🏠 Home", on_click=lambda: nav_to("home"))
    
    s1, s2, s3 = st.columns([4, 2, 1])
    with s1:
        ticker = st.text_input("Search Ticker Symbol" , placeholder = "e.g., AAPL, TSLA, BTC-USD").upper().strip()
        search_btn = st.button("🔍 Search", type="primary")
    with s2:
        period_choice = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with s3:
        st.write("") 
        st.button("🔄 Refresh", use_container_width=True)

    if ticker:
        with st.spinner(f"Fetching {ticker}..."):
            df, err = get_stock_data(ticker, period=period_choice)
        
        if err:
            st.error(f"⚠️ {err}")
        elif df is not None:
            # Metrics
            last_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
            chg, pct = last_close - prev_close, ((last_close - prev_close) / prev_close) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Close", f"${last_close:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            m2.metric("Last Open", f"${float(df['Open'].iloc[-1]):,.2f}")
            m3.metric("Period High", f"${float(df['High'].max()):,.2f}")
            m4.metric("Period Low", f"${float(df['Low'].min()):,.2f}")
            
            # Charting
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20 SMA'], name="20D SMA", line=dict(color="orange", dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], name="Upper BB", line=dict(color="rgba(128,128,128,0.2)")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], name="Lower BB", fill='tonexty', line=dict(color="rgba(128,128,128,0.2)")), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(0,123,255,0.3)"), row=2, col=1)
            
            fig.update_layout(height=500, margin=dict(t=0,b=0), hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators
            c1, c2 = st.columns(2)
            with c1:
                st.caption("RSI (Relative Strength Index)")
                fr = go.Figure(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="purple")))
                fr.add_hline(y=70, line_dash="dash", line_color="red"); fr.add_hline(y=30, line_dash="dash", line_color="green")
                fr.update_layout(height=200, margin=dict(t=0,b=0), template="plotly_white")
                st.plotly_chart(fr, use_container_width=True)
            with c2:
                st.caption("MACD Momentum")
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
                fm.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="orange")))
                fm.update_layout(height=200, margin=dict(t=0,b=0), template="plotly_white")
                st.plotly_chart(fm, use_container_width=True)

            # AI Predictions
            st.divider()
            st.subheader("🤖 AI Price Prediction Suite")
            p1, p2 = st.columns([1, 2])
            with p1:
                ml_model = st.selectbox("Select ML Model", ["Linear Regression", "Polynomial Regression (Deg 2)", "Support Vector Regression (SVR)"])
                days = st.slider("Forecast Horizon (Days)", 1, 90, 30)
                if st.button("🔮 Run AI Model", type="primary", use_container_width=True):
                    preds = perform_ml(df, days, ml_model)
                    with p2:
                        fig_p, ax_p = plt.subplots(figsize=(10, 4))
                        ctx = df.tail(60)
                        ax_p.plot(ctx.index.tz_localize(None), ctx['Close'], label='History', color='#007bff')
                        ax_p.plot(preds['Date'], preds['Price'], 'r--', label='Forecast')
                        ax_p.legend(); ax_p.grid(alpha=0.3)
                        st.pyplot(fig_p)
                    st.dataframe(preds.style.format({"Price": "${:,.2f}"}), use_container_width=True, height=200)

st.markdown("<br><center><small>StockTrend AI Pro | Built by Ravi Kumar Vishwakarma</small></center>", unsafe_allow_html=True)

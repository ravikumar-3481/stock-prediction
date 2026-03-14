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

# --- ADVANCED STYLING ---
st.markdown("""
    <style>
    /* Global Styles */
    [data-testid="stSidebar"] { display: none; }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 50px 30px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .section-container {
        background: #ffffff;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #e0e0e0;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }

    .highlight-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        height: 100%;
    }

    .tech-pill {
        display: inline-block;
        background: #e9ecef;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #495057;
    }

    .dev-card {
        background: white;
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        border: 1px solid #eee;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
    }

    .social-btn {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 50px;
        text-decoration: none !important;
        font-weight: bold;
        color: white !important;
        margin: 5px;
        transition: transform 0.2s;
    }
    .social-btn:hover { transform: scale(1.05); }
    .github { background: #24292e; }
    .linkedin { background: #0077b5; }
    .portfolio { background: #ff4757; }

    /* Fix for white backgrounds in plots */
    .stPlotlyChart { background: white; border-radius: 10px; padding: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
@st.cache_data(ttl=600)
def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty: return None, "Ticker not found or no historical data available."
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
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
        return None, str(e)

def perform_ml(df, days, model_type):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values.flatten()
    
    if model_type == "Linear Regression":
        model = LinearRegression().fit(X, y)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        preds = model.predict(future_X)
    elif model_type == "Polynomial Regression (Deg 2)":
        model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        preds = model.predict(future_X)
    elif model_type == "Support Vector Regression (SVR)":
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        model = SVR(kernel='rbf', C=100, gamma=0.1).fit(X_scaled, y_scaled)
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        preds = scaler_y.inverse_transform(model.predict(scaler_X.transform(future_X)).reshape(-1, 1)).ravel()
    
    last_date = df.index[-1]
    f_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return pd.DataFrame({'Date': f_dates, 'Price': preds})

# --- PAGE: HOME ---
if st.session_state.page == "home":
    st.markdown("""
        <div class="main-header">
            <h1 style='font-size: 3.5rem; margin-bottom: 10px;'>🚀 StockTrend AI Pro</h1>
            <p style='font-size: 1.3rem; opacity: 0.9;'>Advanced Quantitative Analysis & Machine Learning Terminal</p>
        </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
            <div class="section-container">
                <h3 style='color: #007bff;'>📋 Project Overview</h3>
                <p>StockTrend AI Pro is a sophisticated FinTech platform designed to bridge the gap between retail trading and institutional-grade analytics. By leveraging the <b>yfinance</b> ecosystem and <b>Scikit-Learn</b>, the app provides real-time data visualization and mathematical forecasting in a single, unified interface.</p>
            </div>
            
            <div class="section-container">
                <h3 style='color: #d63031;'>⚠️ Problem Statement</h3>
                <p>Modern investors face three critical challenges:</p>
                <ul>
                    <li><b>Information Overload:</b> Too many fragmented sources for news, prices, and charts.</li>
                    <li><b>The "Black Box" Barrier:</b> Most AI tools are either too simple or too complex for the average user to interpret.</li>
                    <li><b>Static Analysis:</b> Traditional charts show where a stock <i>was</i>, but rarely offer mathematical projections of where it <i>might go</i>.</li>
                </ul>
            </div>

            <div class="section-container">
                <h3 style='color: #27ae60;'>💡 The Solution</h3>
                <p>We provide a <b>clean, high-frequency terminal</b> that automates technical indicators (RSI, MACD, Bollinger Bands) and applies <b>Regression Analysis</b> to historical trends. It transforms raw numbers into actionable visual intelligence without requiring a PhD in Data Science.</p>
            </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
            <div class="section-container">
                <h3 style='color: #2c3e50;'>🛠️ Technologies Used</h3>
                <div style='margin-bottom: 15px;'>
                    <span class="tech-pill">Python</span>
                    <span class="tech-pill">Streamlit</span>
                    <span class="tech-pill">Plotly</span>
                    <span class="tech-pill">Scikit-Learn</span>
                    <span class="tech-pill">Pandas</span>
                    <span class="tech-pill">YFinance</span>
                    <span class="tech-pill">NumPy</span>
                </div>
            </div>

            <div class="section-container">
                <h3 style='color: #f39c12;'>✨ Why it's Unique</h3>
                <p>Unlike standard trading apps, StockTrend AI Pro offers <b>On-Demand Machine Learning</b>. Users can switch between <i>Linear</i>, <i>Polynomial</i>, and <i>SVR</i> models instantly to see how different mathematical assumptions change the 90-day price forecast.</p>
            </div>

            <div class="section-container" style="background: #f1f2f6;">
                <h3>🚦 How to Use</h3>
                <p><b>1. Launch Terminal:</b> Click the primary button below.<br>
                <b>2. Enter Ticker:</b> Use symbols like AAPL, TSLA, or BTC-USD.<br>
                <b>3. Review TA:</b> Inspect the RSI and MACD charts for momentum.<br>
                <b>4. Forecast:</b> Scroll to the AI suite, pick a model, and hit 'Run AI'.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        st.button("🎯 Enter Analysis Terminal", on_click=lambda: nav_to("analysis"), type="primary", use_container_width=True)
        st.button("📖 About Developer", on_click=lambda: nav_to("about"), use_container_width=True)

# --- PAGE: ABOUT ---
elif st.session_state.page == "about":
    st.button("⬅️ Back to Home", on_click=lambda: nav_to("home"))
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
            <div class="dev-card">
                <h1 style='color: #2c3e50; margin-bottom: 5px;'>👨‍💻 Developer Profile</h1>
                <h3 style='color: #007bff; margin-top: 0;'>Ravi Kumar Vishwakarma</h3>
                <p style='font-size: 1.1rem; color: #666;'>Computer Science Student | AKS University</p>
                <hr style='border: 0; border-top: 1px solid #eee; margin: 25px 0;'>
                <p style='text-align: left; line-height: 1.6;'>
                    Passionate about the intersection of Finance and Technology. This project was developed 
                    to demonstrate the power of Python in creating real-world utility tools for data 
                    analysis and predictive modeling.
                </p>
                <div style='margin-top: 30px;'>
                    <a href="https://github.com/ravikumar-3481" target="_blank" class="social-btn github">GitHub</a>
                    <a href="https://www.linkedin.com/in/ravi-vishwakarma67" target="_blank" class="social-btn linkedin">LinkedIn</a>
                    <a href="https://profileravi.netlify.app" target="_blank" class="social-btn portfolio">Portfolio</a>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📝 App Specifications & Documentation"):
            st.write("""
            **Data Fetching:** utilizes `yfinance` for OHLCV data.  
            **Technical Indicators:** Custom vectorized calculations for RSI (14-period), MACD (12,26,9), and Bollinger Bands (20-day, 2-std).  
            **ML Models:** - *Linear Regression:* Best for long-term stable trends.
            - *Polynomial (2nd Degree):* Captures parabolic curves and reversals.
            - *SVR (RBF Kernel):* High-sensitivity non-linear fitting for volatile assets.
            """)

# --- PAGE: ANALYSIS ---
elif st.session_state.page == "analysis":
    n1, n2 = st.columns([9, 1])
    n1.title("📈 Analysis Terminal")
    n2.button("🏠 Home", on_click=lambda: nav_to("home"))
    
    s1, s2, s3 = st.columns([4, 2, 1])
    with s1:
        ticker = st.text_input("Ticker Symbol", placeholder="e.g., NVDA, MSFT, RELIANCE.NS").upper().strip()
        search_btn = st.button("🔍 Analyze Now", type="primary")
    with s2:
        period_choice = st.selectbox("Historical Window", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with s3:
        st.write("")
        st.button("🔄 Refresh")

    if ticker:
        with st.spinner(f"Querying Market Data for {ticker}..."):
            df, err = get_stock_data(ticker, period=period_choice)
        
        if err:
            st.error(f"⚠️ {err}")
        elif df is not None:
            # Metrics
            last_close, last_open = float(df['Close'].iloc[-1]), float(df['Open'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_close
            chg, pct = last_close - prev_close, ((last_close - prev_close) / prev_close) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${last_close:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            m2.metric("Today's Open", f"${last_open:,.2f}")
            m3.metric("52W High", f"${df['High'].max():,.2f}")
            m4.metric("52W Low", f"${df['Low'].min():,.2f}")
            
            # Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], name="Upper BB", line=dict(color="rgba(150,150,150,0.2)", dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], name="Lower BB", line=dict(color="rgba(150,150,150,0.2)", dash='dot'), fill='tonexty', fillcolor='rgba(150,150,150,0.05)'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="#d1d8e0"), row=2, col=1)
            
            fig.update_layout(height=500, margin=dict(t=0, b=0), hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sub-indicators
            c1, c2 = st.columns(2)
            with c1:
                st.caption("RSI Momentum")
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="#a55eea")))
                fr.add_hline(y=70, line_dash="dash", line_color="#eb4d4b")
                fr.add_hline(y=30, line_dash="dash", line_color="#6ab04c")
                fr.update_layout(height=200, margin=dict(t=0,b=0), template="plotly_white")
                st.plotly_chart(fr, use_container_width=True)
            with c2:
                st.caption("MACD Divergence")
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="#007bff")))
                fm.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="#f0932b")))
                fm.update_layout(height=200, margin=dict(t=0,b=0), template="plotly_white")
                st.plotly_chart(fm, use_container_width=True)

            # Prediction
            st.divider()
            st.subheader("🤖 AI Price Prediction Suite")
            p1, p2 = st.columns([1, 2])
            with p1:
                ml_model = st.selectbox("Select ML Model", ["Linear Regression", "Polynomial Regression (Deg 2)", "Support Vector Regression (SVR)"])
                days = st.slider("Forecast Horizon (Days)", 1, 90, 30)
                if st.button("🔮 Run AI Model", type="primary", use_container_width=True):
                    preds = perform_ml(df, days, ml_model)
                    with p2:
                        fig_p, ax_p = plt.subplots(figsize=(10, 5))
                        ctx = df.tail(60)
                        ax_p.plot(ctx.index.tz_localize(None), ctx['Close'], label='History', color='#007bff', lw=2)
                        ax_p.plot(preds['Date'], preds['Price'], 'r--', label='Forecast', lw=2)
                        ax_p.set_title(f"{ticker} Forecast - {ml_model}")
                        ax_p.legend()
                        ax_p.grid(alpha=0.3)
                        st.pyplot(fig_p)
                    st.dataframe(preds.style.format({"Price": "${:,.2f}"}), use_container_width=True)

st.markdown("<br><center><small>StockTrend AI Pro | Built with ❤️ by Ravi Kumar</small></center>", unsafe_allow_html=True)

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
    
    /* Home Screen Styling */
    .hero-section {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 60px 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,123,255,0.2);
    }
    .feature-card {
        background: rgba(128, 128, 128, 0.05); /* Adaptive background */
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        height: 100%;
        margin-bottom: 20px;
    }
    .dev-card {
        background: rgba(128, 128, 128, 0.08); /* Adaptive background */
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .social-btn {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        color: white !important;
    }
    .github { background-color: #333; }
    .linkedin { background-color: #0077b5; }
    .portfolio { background-color: #e84393; }
    </style>
    """, unsafe_allow_html=True)

# --- ROBUST DATA ENGINE ---
@st.cache_data(ttl=600)
def get_stock_data(ticker):
    try:
        # Download price data
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            return None, None, "Ticker not found or no data available."
        
        # Clean columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Attempt to get info with robust error handling for missing yfinance data
        t_obj = yf.Ticker(ticker)
        try:
            raw_info = t_obj.info
            if not isinstance(raw_info, dict):
                raw_info = {}
        except Exception:
            raw_info = {}
            
        # Create safe dictionary with all required details defaulting properly
        info = {
            "longName": raw_info.get("longName", raw_info.get("shortName", ticker)),
            "sector": raw_info.get("sector", "N/A"),
            "industry": raw_info.get("industry", "N/A"),
            "longBusinessSummary": raw_info.get("longBusinessSummary", "Company profile metadata is currently limited or unavailable via the API."),
            "fiftyTwoWeekHigh": raw_info.get("fiftyTwoWeekHigh", 0.0),
            "fiftyTwoWeekLow": raw_info.get("fiftyTwoWeekLow", 0.0)
        }
        
        # Format Market Cap cleanly
        mc = raw_info.get("marketCap", "N/A")
        if isinstance(mc, (int, float)):
            if mc >= 1e12:
                info["marketCap"] = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                info["marketCap"] = f"${mc/1e9:.2f}B"
            elif mc >= 1e6:
                info["marketCap"] = f"${mc/1e6:.2f}M"
            else:
                info["marketCap"] = f"${mc:,.2f}"
        else:
            info["marketCap"] = str(mc)
            
        # Technical Indicators
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data, info, None
    except Exception as e:
        return None, None, str(e)

def perform_ml(df, days):
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

# --- PAGE: HOME ---
if st.session_state.page == "home":
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 style='font-size: 3rem;'>🚀 StockTrend AI Pro</h1>
            <p style='font-size: 1.2rem; opacity: 0.9;'>Empowering Investors with Real-Time Analytics & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # About & Why
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📖 About This App</h3>
            <p>StockTrend AI Pro is a comprehensive financial terminal built for modern traders. 
            It integrates live market feeds with advanced mathematical models to visualize 
            price momentum, volatility, and future trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🛠️ How to Use</h3>
            <ol>
                <li>Click <b>'Get Started'</b> to open the terminal.</li>
                <li>Enter any global ticker (e.g., <b>AAPL</b>, <b>TSLA</b>, or <b>RELIANCE.NS</b>).</li>
                <li>Analyze interactive charts, RSI, and MACD indicators.</li>
                <li>Use the ML slider to forecast prices for the next 90 days.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌟 Why Choose AI Pro?</h3>
            <ul>
                <li><b>Live Data:</b> Accurate real-time synchronization with global exchanges.</li>
                <li><b>No-Lag Engine:</b> Optimized data fetching to bypass rate limits.</li>
                <li><b>Predictive Power:</b> Uses Linear Regression to find the market's true momentum.</li>
                <li><b>Clean UI:</b> Distraction-free interface designed for desktop and mobile.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.button("🎯 Get Started - Open Terminal", on_click=lambda: nav_to("analysis"), type="primary", use_container_width=True)

    # Developer Section
    st.markdown("---")
    dev_col1, dev_col2, dev_col3 = st.columns([1, 2, 1])
    with dev_col2:
        st.markdown(f"""
            <div class="dev-card">
                <h3>👨‍💻 About the Developer</h3>
                <p><b>Ravi Kumar</b><br>Computer Science Student | AKS University</p>
                <p>Passionate about FinTech, AI, and building data-driven web applications.</p>
                <div style="margin-top: 20px;">
                    <a href="https://github.com" target="_blank" class="social-btn github">GitHub</a>
                    <a href="https://linkedin.com" target="_blank" class="social-btn linkedin">LinkedIn</a>
                    <a href="#" target="_blank" class="social-btn portfolio">Portfolio</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- PAGE: ANALYSIS ---
elif st.session_state.page == "analysis":
    # Top Bar
    n1, n2 = st.columns([9, 1])
    n1.title("📈 Analysis Terminal")
    n2.button("🏠 Home", on_click=lambda: nav_to("home"))
    
    # Search Input
    s1, s2 = st.columns([5, 1])
    with s1:
        ticker = st.text_input("Search Ticker Symbol (e.g., AAPL, MSFT, BTC-USD)", value="AAPL").upper().strip()
    with s2:
        st.write("") # Spacer
        st.button("🔄 Refresh")

    if ticker:
        with st.spinner("Processing Market Data..."):
            df, info, err = get_stock_data(ticker)
        
        if err:
            st.error(f"⚠️ {err}")
            st.info("Tip: If you see 'Rate Limited', please wait 30 seconds and refresh.")
        elif df is not None:
            # Identity
            st.header(f"{info.get('longName')}")
            
            # Key Metrics
            lp = float(df['Close'].iloc[-1])
            chg = lp - float(df['Close'].iloc[-2])
            pct = (chg / float(df['Close'].iloc[-2])) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${lp:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            m2.metric("Market Cap", info.get('marketCap'))
            m3.metric("52W High", f"${info.get('fiftyTwoWeekHigh'):,.2f}")
            m4.metric("52W Low", f"${info.get('fiftyTwoWeekLow'):,.2f}")
            
            # Graphs
            st.subheader("Technical Analysis Suite")
            
            # Adapt Plotly to current theme (remove white template so it's transparent)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), name="20D MA", line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(0,123,255,0.4)"), row=2, col=1)
            fig.update_layout(height=500, margin=dict(t=10, b=10), hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators
            c1, c2 = st.columns(2)
            with c1:
                st.caption("RSI (Relative Strength Index)")
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="purple")))
                fr.add_hline(y=70, line_dash="dash", line_color="red")
                fr.add_hline(y=30, line_dash="dash", line_color="green")
                fr.update_layout(height=250, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fr, use_container_width=True)
            with c2:
                st.caption("MACD Momentum")
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
                fm.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="orange")))
                fm.update_layout(height=250, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fm, use_container_width=True)

            # Info Tabs
            t1, t2 = st.tabs(["📄 Company Profile", "📅 1-Month Data"])
            with t1:
                st.markdown(f"**Sector:** {info.get('sector')} | **Industry:** {info.get('industry')}")
                st.write(info.get('longBusinessSummary'))
            with t2:
                st.dataframe(df.tail(30).iloc[::-1][['Open', 'High', 'Low', 'Close', 'Volume']].style.format("${:,.2f}"), use_container_width=True)

            # Prediction
            st.divider()
            st.subheader("🤖 AI Price Prediction")
            p1, p2 = st.columns([1, 2])
            with p1:
                days = st.slider("Forecast Days", 1, 90, 30)
                if st.button("🔮 Run AI Model", type="primary"):
                    with st.spinner("Computing..."):
                        preds = perform_ml(df, days)
                        with p2:
                            # Using matplotlib with transparent background to match system theme
                            fig_p, ax_p = plt.subplots(figsize=(10, 5))
                            fig_p.patch.set_alpha(0.0)
                            ax_p.patch.set_alpha(0.0)
                            
                            # Adapt text color based on Streamlit's theme context if needed
                            # Setting neutral colors for axes that work reasonably well on both
                            ax_p.tick_params(colors='gray')
                            for spine in ax_p.spines.values():
                                spine.set_edgecolor('gray')
                                
                            ctx = df.tail(60)
                            ax_p.plot(ctx.index.tz_localize(None), ctx['Close'], label='History', color='#007bff')
                            ax_p.plot(preds['Date'], preds['Price'], 'ro--', label='Forecast', markersize=4)
                            ax_p.legend()
                            st.pyplot(fig_p)
                        st.write("### Predicted Timeline")
                        st.dataframe(preds.style.format({"Price": "${:,.2f}"}), use_container_width=True)

st.markdown("<br><center><small>StockTrend AI Pro | Built with ❤️ by Ravi Kumar</small></center>", unsafe_allow_html=True)

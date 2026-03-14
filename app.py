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
        background: rgba(128, 128, 128, 0.05);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        height: 100%;
        margin-bottom: 20px;
    }
    .dev-card {
        background: rgba(128, 128, 128, 0.08);
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

# --- ROBUST DATA ENGINE (YFINANCE PURELY) ---
@st.cache_data(ttl=600)
def get_stock_data(ticker, period="1y"):
    try:
        # Download price data using yfinance
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        if data.empty:
            return None, "Ticker not found or no historical data available."
        
        # Clean columns if they are MultiIndex (common in newer yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Ensure we have the required columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in data.columns:
                return None, f"Data missing required column: {col}"

        # Calculate Technical Indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # MACD
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['20 SMA'] = data['Close'].rolling(window=20).mean()
        data['20 STD'] = data['Close'].rolling(window=20).std()
        data['Upper Band'] = data['20 SMA'] + (data['20 STD'] * 2)
        data['Lower Band'] = data['20 SMA'] - (data['20 STD'] * 2)
        
        # Drop NaNs to clean up the charts
        data.dropna(inplace=True)
        
        if data.empty:
            return None, "Not enough data to calculate technical indicators."

        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def perform_ml(df, days, model_type):
    # Prepare data using days index for better scaling across different models
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    # Select and configure the model
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
        # SVR requires feature scaling for good results
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_scaled, y_scaled)
        
        future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
        future_X_scaled = scaler_X.transform(future_X)
        preds_scaled = model.predict(future_X_scaled)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    
    # Map back to future dates
    last_date = df.index[-1]
    f_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
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
                <li>Analyze interactive charts, RSI, MACD, and Bollinger Bands.</li>
                <li>Select from multiple AI models to forecast prices.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌟 Why Choose AI Pro?</h3>
            <ul>
                <li><b>Pure Price Action:</b> Focused entirely on historical technical data.</li>
                <li><b>Multiple Timeframes:</b> Analyze data from 1 month to 5 years.</li>
                <li><b>Advanced AI Suite:</b> Features Linear, Polynomial, and SVR ML models.</li>
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
    
    # Search Input & Period Selector
    s1, s2, s3 = st.columns([4, 2, 1])
    with s1:
        ticker = st.text_input("Search Ticker Symbol (e.g., AAPL, MSFT, BTC-USD)", value="AAPL").upper().strip()
    with s2:
        period_choice = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with s3:
        st.write("") # Spacer
        st.button("🔄 Refresh", use_container_width=True)

    if ticker:
        with st.spinner(f"Processing Market Data for {ticker}..."):
            df, err = get_stock_data(ticker, period=period_choice)
        
        if err:
            st.error(f"⚠️ {err}")
        elif df is not None:
            st.header(f"Ticker: {ticker}")
            
            # Key Metrics calculation
            last_close = float(df['Close'].iloc[-1])
            last_open = float(df['Open'].iloc[-1])
            
            if len(df) > 1:
                prev_close = float(df['Close'].iloc[-2])
                chg = last_close - prev_close
                pct = (chg / prev_close) * 100
            else:
                chg = 0.0
                pct = 0.0
                
            period_high = float(df['High'].max())
            period_low = float(df['Low'].min())
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Close", f"${last_close:,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
            m2.metric("Last Open", f"${last_open:,.2f}")
            m3.metric("Period High", f"${period_high:,.2f}")
            m4.metric("Period Low", f"${period_low:,.2f}")
            
            # --- MAIN CHARTS ---
            st.subheader("Technical Analysis Suite")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Price and Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="#007bff")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['20 SMA'], name="20D SMA", line=dict(color="orange", dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], name="Upper BB", line=dict(color="rgba(128,128,128,0.3)", dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], name="Lower BB", line=dict(color="rgba(128,128,128,0.3)", dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color="rgba(0,123,255,0.4)"), row=2, col=1)
            
            # Force Absolute White Background and Black Fonts (fixes dark mode rendering issues)
            fig.update_layout(
                height=550, 
                margin=dict(t=10, b=10), 
                hovermode="x unified",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black'),
                xaxis2=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black'),
                yaxis2=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- INDICATORS ---
            c1, c2 = st.columns(2)
            with c1:
                st.caption("RSI (Relative Strength Index)")
                fr = go.Figure()
                fr.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="purple")))
                fr.add_hline(y=70, line_dash="dash", line_color="red")
                fr.add_hline(y=30, line_dash="dash", line_color="green")
                fr.update_layout(
                    height=250, margin=dict(t=0,b=0), 
                    plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black')
                )
                st.plotly_chart(fr, use_container_width=True)
            with c2:
                st.caption("MACD Momentum")
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")))
                fm.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="orange")))
                fm.update_layout(
                    height=250, margin=dict(t=0,b=0), 
                    plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'),
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='black')
                )
                st.plotly_chart(fm, use_container_width=True)

            # --- DATA TABLE & EXPORT ---
            st.markdown("### 🗃️ Raw Historical Data")
            d1, d2 = st.columns([8, 2])
            with d1:
                # Format to $ for table
                st.dataframe(df.iloc[::-1][['Open', 'High', 'Low', 'Close', 'Volume']].style.format({
                    "Open": "${:,.2f}", "High": "${:,.2f}", "Low": "${:,.2f}", "Close": "${:,.2f}"
                }), use_container_width=True, height=200)
            with d2:
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f'{ticker}_historical_data.csv',
                    mime='text/csv',
                    use_container_width=True
                )

            # --- PREDICTION MODULE ---
            st.divider()
            st.subheader("🤖 AI Price Prediction Suite")
            p1, p2 = st.columns([1, 2])
            
            with p1:
                st.info("Select an algorithm to forecast future price movement based on historical trends.")
                ml_model = st.selectbox("Select ML Model", [
                    "Linear Regression", 
                    "Polynomial Regression (Deg 2)", 
                    "Support Vector Regression (SVR)"
                ])
                days = st.slider("Forecast Horizon (Days)", 1, 90, 30)
                
                if st.button("🔮 Run AI Model", type="primary", use_container_width=True):
                    with st.spinner(f"Computing using {ml_model}..."):
                        preds = perform_ml(df, days, ml_model)
                        
                        with p2:
                            # Strict White background for Matplotlib
                            fig_p, ax_p = plt.subplots(figsize=(10, 5))
                            fig_p.patch.set_facecolor('white')
                            ax_p.set_facecolor('white')
                                
                            # Plot last 100 days for visual context
                            ctx = df.tail(100)
                            ax_p.plot(ctx.index.tz_localize(None), ctx['Close'], label='History', color='#007bff', linewidth=2)
                            ax_p.plot(preds['Date'], preds['Price'], 'r--', label=f'{ml_model} Forecast', linewidth=2)
                            
                            ax_p.set_title(f"{ticker} - {days} Day Forecast", color='black')
                            ax_p.tick_params(colors='black')
                            ax_p.grid(color='#f0f0f0', linestyle='--', linewidth=0.5)
                            
                            # Styling legend
                            legend = ax_p.legend(facecolor='white', edgecolor='black', labelcolor='black')
                            frame = legend.get_frame()
                            frame.set_facecolor('white')
                            
                            st.pyplot(fig_p)
                            
                        # Data table for predictions
                        with p1:
                            st.write("### Predicted Timeline")
                            st.dataframe(preds.style.format({"Price": "${:,.2f}"}), use_container_width=True, height=250)

st.markdown("<br><center><small>StockTrend AI Pro | Built with ❤️ by Ravi Kumar</small></center>", unsafe_allow_html=True)

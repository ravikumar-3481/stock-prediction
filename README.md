# 📈 Real-Time Stock Trend Visualizer & ML Predictor

An interactive web application built with **Streamlit** that fetches live stock market data and uses **Machine Learning** to predict future price movements.

🔗 **Live Demo:** [Stock Prediction App](https://stock-prediction-ravikumar-3481.streamlit.app/)

---

## 🚀 Overview
This project provides a comprehensive dashboard for stock market analysis. It combines real-time data fetching with technical indicators and a predictive model to help users visualize market momentum.

### Key Features:
* **Live Data Integration:** Fetches real-time market data using the `yfinance` API.
* **Technical Analysis:** Automatically calculates and plots **20-day** and **50-day Moving Averages (MA)**.
* **ML Price Prediction:** Uses a **Linear Regression** model to forecast price trends for up to 30 days.
* **Interactive UI:** Users can search for any global ticker (e.g., AAPL, TSLA, BTC-USD) and adjust prediction windows dynamically.
* **Robust Error Handling:** Optimized to handle MultiIndex dataframes and API connection issues.

---

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/)
* **Visualization:** [Matplotlib](https://matplotlib.org/)
* **API:** [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/)

---

## 📋 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/stock-prediction.git](https://github.com/your-username/stock-prediction.git)
   cd stock-prediction

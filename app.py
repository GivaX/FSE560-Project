import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
import plotly.graph_objects as go

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Caching model and scaler loading to speed up predictions
@st.cache_resource
def load_artifacts():
    model  = keras.models.load_model("sp500_mlp_model.keras")
    scaler = joblib.load("sp500_scaler.pkl")
    return model, scaler

# Feature engineering function - same as training
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create the target variable: 1 if the next day's closing price is higher than today's, else 0
    df["Next Day Close"] = df["Close"].shift(-1)
    df["Target"] = ((df["Next Day Close"] - df["Close"]) / df["Close"] > 0.002).astype(int)

    # Create historical rolling features for each ticker
    df["Return_1d"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].transform(
        lambda x: x.rolling(window=5).mean()
    )

    df["MA_10"] = df["Close"].transform(
        lambda x: x.rolling(window=10).mean()
    )

    df["Volume_MA_5"] = df["Volume"].transform(
        lambda x: x.rolling(window=5).mean()
    )

    df["Volatility_5"] = df["Close"].transform(
        lambda x: x.pct_change().rolling(window=5).std()
    )

    df["Close_vs_MA5"] = (df["Close"] - df["MA_5"]) / df["MA_5"]
    df["Close_vs_MA10"] = (df["Close"] - df["MA_10"]) / df["MA_10"]
    df["MA5_vs_MA10"] = (df["MA_5"] - df["MA_10"]) / df["MA_10"]
    df["Volume_vs_MA5"] = (df["Volume"] - df["Volume_MA_5"]) / df["Volume_MA_5"]

    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)
    df["Close_lag3"] = df["Close"].shift(3)

    df["Return_lag1"] = df["Return_1d"].shift(1)
    df["Return_lag2"] = df["Return_1d"].shift(2)

    return df

# Chart function
def make_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=ticker,
        increasing_line_color="#1D9E75",
        decreasing_line_color="#D85A30",
    ))

    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Stock Market Trend - AI Predictor", page_icon="📈", layout="centered")

    st.title("Stock Market Trend - AI Predictor")
    st.write("Enter a ticker symbol and click Predict.")

    st.subheader("Select Prediction Day")
    day = st.slider("Get trend prediction of the next day from seletion (e.g. 1 = Latest Trading Day, 5 = 4 days before latest)",
                            min_value=1, max_value=40, value=1)

    ticker = st.text_input("Enter Stock Ticker (e.g., NVDA, AAPL)")

    if st.button("Predict"):

        if ticker == "":
            st.warning("Please enter a ticker symbol.")
            st.stop()

        try:
            with st.spinner("Loading model and scaler..."):
                model, scaler = load_artifacts()

        except Exception as e:
            st.error(f"Error loading model/scaler: {e}")
            st.stop()

        with st.spinner("Fetching stock data..."):
            end = pd.Timestamp.now()
            start = end - pd.Timedelta(days=90)
            data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
            print(data.iloc[-5:])

        if data.empty or len(data) < 20:
            st.error("Not enough data to make prediction.")
            st.stop()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.loc[:, ~data.columns.duplicated()]

        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        df = data.copy()

        df = compute_features(df)

        latest = df.iloc[-day]

        df = df.dropna() 

        features = np.array([[
            latest["Open"],
            latest["High"],
            latest["Low"],
            latest["Close"],
            latest["Volume"],
            latest["Return_1d"],
            latest["MA_5"],
            latest["MA_10"],
            latest["Volume_MA_5"],
            latest["Volatility_5"],
            latest["Close_vs_MA5"],
            latest["Close_vs_MA10"],
            latest["MA5_vs_MA10"],
            latest["Volume_vs_MA5"],
            latest["Close_lag1"],
            latest["Close_lag2"],
            latest["Close_lag3"],
            latest["Return_lag1"],
            latest["Return_lag2"],
        ]])

        features_scaled = scaler.transform(features)

        with st.spinner("Running prediction..."):
            probs = model(features_scaled, training=False).numpy()[0]

        st.subheader(f"Prediction for {ticker.upper()}")

        col1, col2 = st.columns(2)
        col1.metric(label="Bearish Probability", value=f"{probs[0] * 100:.2f}%")
        col2.metric(label="Bullish Probability", value=f"{probs[1] * 100:.2f}%")


        if probs[1] > 0.5:
            st.success(f"Bullish Signal the next day from {latest.name.strftime('%b %d, %Y')}!")
        else:
            st.error(f"Bearish Signal the next day from {latest.name.strftime('%b %d, %Y')}!")

        st.plotly_chart(make_chart(data, ticker), use_container_width=True)

if __name__ == "__main__":
    main()

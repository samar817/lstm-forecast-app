import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ABSOLUTE PATH TO YOUR FOLDER
BASE_DIR = r"f:\job\LSTM training script (offline)"
WINDOW = 60

@st.cache_resource
def load_model(path: str):
    return keras.models.load_model(path)

@st.cache_data
def load_scaler() -> MinMaxScaler:
    scaler = MinMaxScaler()
    data_min = np.load(BASE_DIR + r"\scaler_min.npy")
    data_max = np.load(BASE_DIR + r"\scaler_max.npy")
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.scale_ = 1.0 / (data_max - data_min + 1e-9)
    scaler.min_ = -data_min * scaler.scale_
    return scaler

def forecast_prices(ticker: str, horizon: int):
    data = yf.download(ticker, period="5y", auto_adjust=True)
    if data.empty:
        raise ValueError("No data for this ticker")

    close = data["Close"].values.reshape(-1, 1)

    scaler = load_scaler()
    scaled = scaler.transform(close)

    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    model = load_model(BASE_DIR + r"\lstm_model.h5")

    preds_scaled = []
    window_seq = last_window.copy()

    for _ in range(horizon):
        pred = model.predict(window_seq, verbose=0)[0, 0]
        preds_scaled.append(pred)
        window_seq = np.roll(window_seq, -1, axis=1)
        window_seq[0, -1, 0] = pred

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    last_date = data.index[-1]
    future_idx = pd.date_range(last_date, periods=horizon + 1, freq="B")[1:]

    hist_df = data[["Close"]].copy()
    forecast_df = pd.DataFrame({"Forecast": preds}, index=future_idx)
    return hist_df, forecast_df

def main():
    st.title("Stock Price Forecasting (LSTM)")

    ticker = st.text_input("Ticker symbol", "RELIANCE.NS")
    horizon = st.slider("Forecast horizon (days)", 5, 60, 15)

    if st.button("Run forecast"):
        try:
            hist_df, forecast_df = forecast_prices(ticker, horizon)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        st.subheader("Historical Close Price")
        st.line_chart(hist_df["Close"])

        st.subheader("Historical + Forecast")
        combined = pd.concat([hist_df.tail(120), forecast_df], axis=0)
        st.line_chart(combined)

        st.subheader("Forecast values")
        st.dataframe(forecast_df)

if __name__ == "__main__":
    main()

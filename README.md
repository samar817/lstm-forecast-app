# LSTM Stock Price Forecast App

An end-to-end stock forecasting web app built with Python and Streamlit.  
The project trains an LSTM model on historical price data and exposes an interactive web UI to generate short‑term forecasts for any stock ticker supported by Yahoo Finance.
## Features

- Downloads up to 5 years of historical price data using `yfinance`.
- Scales data and trains an LSTM model on closing prices.
- Forecasts future prices for a user-selected horizon (e.g., 5–60 business days).
- Interactive Streamlit UI:
  - Input any ticker (US, NSE `.NS`, BSE `.BO`, indices).
  - Plots historical prices and historical + forecasted curve.
  - Displays a table of forecast values by future date.
- Packaged as a reproducible project with saved model (`lstm_model.h5`) and scaler (`.npy` files).
## Project structure

- `lstm_app.py` – Streamlit app that loads the trained LSTM model and serves forecasts.
- `train_lstm.py` – Offline script to download data and train/save the LSTM model.
- `lstm_model.h5` – Saved Keras LSTM model.
- `scaler_min.npy`, `scaler_max.npy` – Saved scaler parameters for inverse-transform.
- `requirements.txt` – Python dependencies for running the app.
## Usage

- Enter a valid stock ticker in the **Ticker symbol** box.
  - Examples:
    - US: `AAPL`, `MSFT`, `TSLA`
    - NSE: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
    - BSE: `RELIANCE.BO`
- Choose the **Forecast horizon (days)** with the slider.
- Click **Run forecast**.
- The app will:
  - Show historical close prices.
  - Plot the last 120 days plus forecasted prices.
  - Display a table of forecast values for each future business day.

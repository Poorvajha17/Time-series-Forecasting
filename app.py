#importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to check stationarity
def check_stationarity(data):
    result = adfuller(data)
    st.write("Augmented Dickey-Fuller Test:")
    st.write(f"ADF Statistic: {result[0]}")
    st.write(f"p-value: {result[1]}")
    if result[1] < 0.05:
        st.write("The data is stationary.")
    else:
        st.write("The data is not stationary.")

# Function to plot autocorrelation
def plot_autocorrelation(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(data, ax=ax)
    st.pyplot(fig)

# Function to train ARIMA model with hyperparameter tuning
def train_arima(data, order=(5,1,2), steps=1):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit

# Function to train SARIMA model
def train_sarima(data, order=(1,1,1), seasonal_order=(1,1,1,12), steps=1):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit


# Function to evaluate model performance
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    st.write("### Model Evaluation")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Streamlit UI
st.title("Stock Price Prediction")

st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-03-01"))

if st.sidebar.button("Load Data"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    if not stock_data.empty:
        st.subheader(f"Stock Data for {ticker}")
        st.write(stock_data.tail(10))
        st.line_chart(stock_data["Close"])
        
        check_stationarity(stock_data['Close'])
        plot_autocorrelation(stock_data['Close'])
    else:
        st.error("Failed to load stock data. Please check the ticker symbol.")

st.sidebar.header("Forecasting")
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA"])
forecast_duration = st.sidebar.radio("Forecast Period", ["Next Day", "Next Week", "Next Month"])

if forecast_duration == "Next Day":
    steps = 1
elif forecast_duration == "Next Week":
    steps = 7
else:
    steps = 30

if st.sidebar.button("Predict Future Prices"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if not stock_data.empty:
        last_date = stock_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
        
        if model_choice == "ARIMA":
            forecast_values, model = train_arima(stock_data["Close"], steps=steps)
        elif model_choice == "SARIMA":
            forecast_values, model = train_sarima(stock_data["Close"], steps=steps)
        
        st.subheader(f"Predicted Prices for {ticker} ({forecast_duration})")
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close Price": forecast_values})
        forecast_df.set_index("Date", inplace=True)
        st.write(forecast_df)
        
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index, stock_data["Close"], label="Historical Prices", color="blue")
        plt.plot(forecast_df.index, forecast_df["Predicted Close Price"], label="Forecasted Prices", color="red", linestyle="dashed")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.title(f"Stock Price Prediction for {ticker} ({forecast_duration})")
        st.pyplot(plt)
        
        evaluate_model(stock_data["Close"].iloc[-steps:], forecast_values)
    else:
        st.error("Failed to load stock data. Please try again.")
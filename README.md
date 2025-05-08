# Time-series-Forecasting

## Overview
This is a Streamlit-based web application for predicting stock prices using time series forecasting models. The app fetches real-time stock data from Yahoo Finance and applies forecasting models like ARIMA and SARIMA to predict future prices.

## Features
- Fetch historical stock data from Yahoo Finance.
- Check stationarity of stock prices using the Augmented Dickey-Fuller (ADF) test.
- Visualize stock price trends and autocorrelation.
- Predict future stock prices using ARIMA and SARIMA models.
- Evaluate model performance using MAE, RMSE, and MAPE.
- Interactive UI with options for selecting stock symbols, date ranges, and forecast duration.

## Installation

### Prerequisites
Ensure you have Python installed (Python 3.7 or higher is recommended). Install the required dependencies using:

pip install streamlit pandas numpy matplotlib seaborn yfinance statsmodels scikit-learn xgboost

## Usage
Run the application using:

streamlit run app.py

## How It Works
1. **Load Data**: Enter a stock ticker symbol (e.g., AAPL, TSLA) and select a date range to fetch historical stock data.
2. **Check Stationarity**: Perform the Augmented Dickey-Fuller test to check if the stock price series is stationary.
3. **Forecasting Models**:
   - **ARIMA**: Uses AutoRegressive Integrated Moving Average for forecasting.
   - **SARIMA**: Seasonal ARIMA for capturing seasonality in stock prices.
4. **Evaluate Performance**: The model is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Model Selection
- ARIMA: Best suited for non-seasonal stock price trends.
- SARIMA: Recommended if stock prices exhibit seasonality.

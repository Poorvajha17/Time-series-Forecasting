# Time Series Forecasting Platform  
## Introduction

This project presents a modern **Stock Price Forecasting Platform** implemented as a full-stack web application using **FastAPI** (backend) and **HTML/CSS/JavaScript + Chart.js** (frontend).  
The application allows users to fetch real historical stock data, perform detailed exploratory analysis, train multiple forecasting models (statistical and machine learning), generate future price predictions with confidence intervals, and export data for Tableau visualization.  
The goal is to provide deep insights into stock price movements and help users make more informed decisions.

## Approach

The project follows a structured approach to implement stock price forecasting:

1. **Data Collection**  
   Fetching historical stock data from Yahoo Finance using `yfinance`.

2. **Exploratory Data Analysis (EDA)**  
   - Checking stationarity using the Augmented Dickey-Fuller (ADF) test  
   - Visualizing historical stock prices  
   - Seasonality & trend decomposition  
   - Plotting Autocorrelation (ACF) and Partial Autocorrelation (PACF) to analyze time dependencies  
   - Time window analysis (daily/weekly/monthly averages & volatility)  
   - Day-of-week price pattern visualization

3. **Model Selection & Training**  
   - **ARIMA** (AutoRegressive Integrated Moving Average): A standard time-series forecasting model  
   - **SARIMA** (Seasonal ARIMA): An extension that incorporates seasonality  
   - **Random Forest**: Machine learning ensemble model with rich feature engineering  
   - **XGBoost**: Advanced gradient boosting model  
   - Feature engineering and hyperparameter considerations for optimal performance

4. **Forecasting**  
   - Predicting stock prices for different time periods (next day, next week, next month, next quarter)  
   - Generating 95% confidence intervals  
   - Visualizing forecasted prices along with historical trends

5. **Performance Evaluation**  
   - Metrics used: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),  
     Mean Absolute Percentage Error (MAPE)  
   - Additional metrics: R² score (ML models), AIC/BIC (statistical models)

## Methodology

### 1. Data Preprocessing
- The application allows users to select a stock symbol, start date, and end date  
- Stock data is retrieved using the Yahoo Finance API (`yfinance`)

### 2. Data Cleaning
- **Handling Missing Values**: Forward-fill and back-fill methods  
- **Outlier Detection**: Handling of extreme values to improve model stability  
- **Ensuring Data Integrity**: Removal of duplicates, timezone standardization, date format consistency  
- Cleaning report: total records, missing values filled, duplicates removed

### 3. Stationarity Check
- **ADF Test**: Determines whether the time series is stationary or requires differencing  
- **Autocorrelation Analysis**: ACF and PACF plots to understand time dependencies

### 4. Model Training

**ARIMA Model**  
- Trained with order parameters (p, d, q), where:  
  - p (AutoRegressive term) determines the number of lag observations  
  - d (Differencing order) ensures stationarity  
  - q (Moving Average term) defines the size of the error component  
- Forecasting is performed using the fitted model

**SARIMA Model**  
- Extends ARIMA by incorporating seasonal components (P, D, Q, s)  
- Captures seasonal patterns in stock price behavior

**Random Forest & XGBoost Models**  
- Advanced feature engineering (lags, moving averages, volatility, momentum, time features)  
- Recursive multi-step forecasting approach

### 5. Forecasting & Visualization
- Users select forecasting model and period (day/week/month/quarter)  
- Predictions are generated with confidence intervals  
- Interactive Chart.js visualizations show historical data, forecast, and confidence bands

### 6. Performance Evaluation
- Predicted values are evaluated against historical/test data  
- Evaluation metrics:  
  - MAE (Mean Absolute Error)  
  - RMSE (Root Mean Squared Error)  
  - MAPE (Mean Absolute Percentage Error)  
  - R² Score (for ML models)  
  - AIC/BIC (for statistical models)

## Results & Conclusion

- The application successfully forecasts stock prices with reasonable to good accuracy across multiple models  
- Seasonal trends are effectively captured using SARIMA when present  
- Machine learning models (especially XGBoost) often show superior performance on complex real-world stock data  
- Future improvements could involve incorporating deep learning models (LSTM, Transformers), real-time data, or ensemble approaches

## Technologies Used

- **Python** (pandas, numpy)  
- **FastAPI** + **Uvicorn** (backend API)  
- **yfinance** (stock data retrieval)  
- **statsmodels** (ARIMA and SARIMA modeling)  
- **scikit-learn** (Random Forest)  
- **xgboost** (gradient boosting)  
- **HTML5 / CSS3 / JavaScript** + **Chart.js** (modern interactive frontend)

## Deployment

- The application can be deployed on platforms such as Render, Railway, Vercel, AWS, or DigitalOcean  
- Provides real-time stock forecasting through a web browser interface

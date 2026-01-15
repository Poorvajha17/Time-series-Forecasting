from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

app = FastAPI(title="Time-Series Forecasting API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class DataRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

class ForecastRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    model: str = "ARIMA"
    period: str = "week"

class MultiTickerRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str

# ============================================================================
# MILESTONE 1: DATA RETRIEVAL
# ============================================================================

class DataRetriever:
    @staticmethod
    def fetch_stock_data(ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=30)
            
            if data is None or data.empty:
                return None, f"No data found for {ticker}."
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return data, None
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"
    
    @staticmethod
    def validate_data(data):
        if data is None or data.empty:
            return False, "Data is empty"
        if len(data) < 30:
            return False, "Insufficient data points (minimum 30 required)"
        return True, "Data validated successfully"

# ============================================================================
# MILESTONE 2: DATA CLEANING
# ============================================================================

class DataPreprocessor:
    @staticmethod
    def clean_data(data):
        df = data.copy()
        
        # Handle missing values
        initial_missing = df.isnull().sum().sum()
        df = df.ffill().bfill()
        
        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = initial_len - len(df)
        
        # Ensure timezone consistency
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        # Convert to datetime
        df.index = pd.to_datetime(df.index)
        
        cleaning_report = {
            'missing_values_filled': int(initial_missing),
            'duplicates_removed': duplicates_removed,
            'final_records': len(df),
            'date_format': 'YYYY-MM-DD HH:MM:SS'
        }
        
        return df, cleaning_report
    
    @staticmethod
    def create_features(data):
        """FIXED: Better feature engineering"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        
        # Volatility
        df['Volatility_5'] = df['Returns'].rolling(window=5, min_periods=1).std()
        df['Volatility_10'] = df['Returns'].rolling(window=10, min_periods=1).std()
        
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
        
        # Time features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        
        # Lag features
        for i in [1, 2, 3, 5, 10]:
            df[f'Lag_{i}'] = df['Close'].shift(i)
        
        # Fill NaN with forward/backward fill
        df = df.ffill().bfill()
        
        return df

# ============================================================================
# MILESTONE 3: EDA
# ============================================================================

class EDAAnalyzer:
    @staticmethod
    def basic_statistics(data):
        stats = {
            'mean': float(data['Close'].mean()),
            'median': float(data['Close'].median()),
            'std': float(data['Close'].std()),
            'min': float(data['Close'].min()),
            'max': float(data['Close'].max()),
            'range': float(data['Close'].max() - data['Close'].min()),
            'variance': float(data['Close'].var()),
            'skewness': float(data['Close'].skew()),
            'kurtosis': float(data['Close'].kurtosis())
        }
        return stats
    
    @staticmethod
    def check_stationarity(data):
        result = adfuller(data['Close'].dropna())
        is_stationary = result[1] < 0.05
        
        return {
            'adf_statistic': float(result[0]),
            'p_value': float(result[1]),
            'critical_values': {k: float(v) for k, v in result[4].items()},
            'is_stationary': bool(is_stationary),
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary (differencing needed)'
        }
    
    @staticmethod
    def analyze_seasonality(data, period=30):
        try:
            if len(data) < period * 2:
                return {'has_trend': False, 'has_seasonality': False, 
                       'message': 'Insufficient data'}
            
            decomposition = seasonal_decompose(
                data['Close'], model='additive', period=min(period, len(data)//2)
            )
            
            trend_var = decomposition.trend.dropna().var()
            resid_var = decomposition.resid.dropna().var()
            seasonal_var = decomposition.seasonal.dropna().var()
            
            trend_strength = 1 - (resid_var / (trend_var + resid_var)) if (trend_var + resid_var) > 0 else 0
            seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0
            
            return {
                'has_trend': True,
                'has_seasonality': True,
                'period': period,
                'trend_strength': float(max(0, min(1, trend_strength))),
                'seasonal_strength': float(max(0, min(1, seasonal_strength))),
                'trend_component': decomposition.trend.dropna().tail(30).tolist(),
                'seasonal_component': decomposition.seasonal.tail(30).tolist()
            }
        except Exception as e:
            return {'has_trend': False, 'has_seasonality': False, 'error': str(e)}
    
    @staticmethod
    def calculate_acf_pacf(data, lags=40):
        max_lags = min(lags, len(data)//2 - 1)
        
        acf_values = acf(data['Close'].dropna(), nlags=max_lags)
        pacf_values = pacf(data['Close'].dropna(), nlags=max_lags)
        
        confidence_interval = 1.96 / np.sqrt(len(data))
        significant_lags = [i for i, val in enumerate(acf_values) 
                          if abs(val) > confidence_interval and i > 0]
        
        return {
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'significant_lags': significant_lags[:10],
            'confidence_interval': float(confidence_interval)
        }
    
    @staticmethod
    def time_window_analysis(data):
        daily_avg = float(data['Close'].mean())
        weekly_data = data['Close'].resample('W').mean()
        weekly_avg = float(weekly_data.mean())
        monthly_data = data['Close'].resample('M').mean()
        monthly_avg = float(monthly_data.mean())
        
        daily_vol = float(data['Close'].std())
        weekly_vol = float(weekly_data.std())
        monthly_vol = float(monthly_data.std())
        
        return {
            'daily_average': daily_avg,
            'weekly_average': weekly_avg,
            'monthly_average': monthly_avg,
            'daily_volatility': daily_vol,
            'weekly_volatility': weekly_vol,
            'monthly_volatility': monthly_vol
        }
    
    @staticmethod
    def generate_heatmap_data(data):
        df = data.copy()
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        # Create hour x day of week heatmap matrix
        heatmap_matrix = []
        for hour in range(24):
            row = []
            for day in range(7):
                mask = (df['Hour'] == hour) & (df['DayOfWeek'] == day)
                avg_price = df.loc[mask, 'Close'].mean() if mask.any() else None
                row.append(float(avg_price) if pd.notna(avg_price) else None)
            heatmap_matrix.append(row)
        
        # Aggregate patterns
        hourly_pattern = df.groupby('Hour')['Close'].mean().to_dict()
        daily_pattern = df.groupby('DayOfWeek')['Close'].mean().to_dict()
        monthly_pattern = df.groupby('Month')['Close'].mean().to_dict()
        
        return {
            'matrix': heatmap_matrix, 
            'hourly': {int(k): float(v) for k, v in hourly_pattern.items()},
            'daily': {int(k): float(v) for k, v in daily_pattern.items()},
            'monthly': {int(k): float(v) for k, v in monthly_pattern.items()}
        }

# ============================================================================
# MILESTONE 4: FORECASTING MODELS - FIXED
# ============================================================================

class ForecastingModels:
    @staticmethod
    def train_arima(data, order=(5,1,2), steps=30):
        try:
            model = ARIMA(data['Close'], order=order)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=steps)
            
            # In-sample metrics
            predictions = model_fit.fittedvalues
            actual = data['Close'].iloc[len(data['Close'])-len(predictions):]
            predictions = predictions.iloc[-len(actual):]
            
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            return forecast.values, {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'aic': float(model_fit.aic),
                'bic': float(model_fit.bic),
                'order': order
            }
        except Exception as e:
            return None, {'error': str(e)}
    
    @staticmethod
    def train_sarima(data, order=(1,1,1), seasonal_order=(1,1,1,12), steps=30):
        try:
            model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, maxiter=200)
            
            forecast = model_fit.forecast(steps=steps)
            
            predictions = model_fit.fittedvalues
            actual = data['Close'].iloc[-len(predictions):]
            
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            return forecast.values, {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'aic': float(model_fit.aic),
                'bic': float(model_fit.bic),
                'order': order,
                'seasonal_order': seasonal_order
            }
        except Exception as e:
            return None, {'error': str(e)}
    
    @staticmethod
    def train_random_forest(data, steps=30, n_estimators=100):
        try:
            df = DataPreprocessor.create_features(data)
            
            # Select features
            feature_cols = ['Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility_5', 
                          'Momentum_5', 'DayOfWeek', 'Month', 'Lag_1', 'Lag_2', 'Lag_3']
            
            # Ensure no NaN
            df = df.dropna()
            
            if len(df) < 50:
                return None, {'error': 'Insufficient data after feature creation'}
            
            X = df[feature_cols].values
            y = df['Close'].values
            
            # Train-test split (80-20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=15, 
                                        min_samples_split=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluation
            test_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            r2 = r2_score(y_test, test_pred)
            
            # Recursive forecasting with proper state tracking
            forecast = []
            last_row = df.iloc[-1].copy()
            close_history = df['Close'].tail(50).tolist()
            
            for step in range(steps):
                # Build feature vector from current state
                if len(close_history) >= 20:
                    ma_5 = np.mean(close_history[-5:])
                    ma_10 = np.mean(close_history[-10:])
                    ma_20 = np.mean(close_history[-20:])
                else:
                    ma_5 = ma_10 = ma_20 = close_history[-1]
                
                returns = (close_history[-1] - close_history[-2]) / close_history[-2] if len(close_history) >= 2 else 0
                vol_5 = np.std(np.diff(close_history[-6:])) if len(close_history) >= 6 else 0
                momentum_5 = close_history[-1] - close_history[-6] if len(close_history) >= 6 else 0
                
                current_date = df.index[-1] + timedelta(days=step+1)
                
                feature_vector = np.array([[
                    returns, ma_5, ma_10, ma_20, vol_5, momentum_5,
                    current_date.dayofweek, current_date.month,
                    close_history[-1], close_history[-2] if len(close_history) >= 2 else close_history[-1],
                    close_history[-3] if len(close_history) >= 3 else close_history[-1]
                ]])
                
                pred = model.predict(feature_vector)[0]
                forecast.append(float(pred))
                close_history.append(pred)
            
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            return np.array(forecast), {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2),
                'feature_importance': {k: float(v) for k, v in feature_importance.items()}
            }
        except Exception as e:
            return None, {'error': f'Random Forest error: {str(e)}'}
    
    @staticmethod
    def train_xgboost(data, steps=30, n_estimators=100):
        if not XGBOOST_AVAILABLE:
            return None, {'error': 'XGBoost not installed'}
        
        try:
            df = DataPreprocessor.create_features(data)
            
            feature_cols = ['Returns', 'MA_5', 'MA_10', 'MA_20', 'Volatility_5', 
                          'Momentum_5', 'DayOfWeek', 'Month', 'Lag_1', 'Lag_2', 'Lag_3']
            
            df = df.dropna()
            
            if len(df) < 50:
                return None, {'error': 'Insufficient data'}
            
            X = df[feature_cols].values
            y = df['Close'].values
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.1, 
                               max_depth=7, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            test_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            r2 = r2_score(y_test, test_pred)
            
            # Same recursive forecasting as RF
            forecast = []
            close_history = df['Close'].tail(50).tolist()
            
            for step in range(steps):
                if len(close_history) >= 20:
                    ma_5 = np.mean(close_history[-5:])
                    ma_10 = np.mean(close_history[-10:])
                    ma_20 = np.mean(close_history[-20:])
                else:
                    ma_5 = ma_10 = ma_20 = close_history[-1]
                
                returns = (close_history[-1] - close_history[-2]) / close_history[-2] if len(close_history) >= 2 else 0
                vol_5 = np.std(np.diff(close_history[-6:])) if len(close_history) >= 6 else 0
                momentum_5 = close_history[-1] - close_history[-6] if len(close_history) >= 6 else 0
                
                current_date = df.index[-1] + timedelta(days=step+1)
                
                feature_vector = np.array([[
                    returns, ma_5, ma_10, ma_20, vol_5, momentum_5,
                    current_date.dayofweek, current_date.month,
                    close_history[-1], close_history[-2] if len(close_history) >= 2 else close_history[-1],
                    close_history[-3] if len(close_history) >= 3 else close_history[-1]
                ]])
                
                pred = model.predict(feature_vector)[0]
                forecast.append(float(pred))
                close_history.append(pred)
            
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            return np.array(forecast), {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2),
                'feature_importance': {k: float(v) for k, v in feature_importance.items()}
            }
        except Exception as e:
            return None, {'error': f'XGBoost error: {str(e)}'}

# ============================================================================
# MILESTONE 5: FORECAST GENERATION
# ============================================================================

class ForecastGenerator:
    @staticmethod
    def calculate_confidence_interval(forecast, data, confidence=0.95):
        forecast = np.asarray(forecast)
        
        # Use historical volatility
        historical_std = data['Close'].pct_change().std() * data['Close'].mean()
        
        # Expanding uncertainty over time
        z_score = 1.96  # 95% confidence
        steps = np.arange(1, len(forecast) + 1)
        
        # Increasing margin as we go further into future
        margin = z_score * historical_std * np.sqrt(steps) * 0.5
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        return lower_bound, upper_bound
    
    @staticmethod
    def generate_forecast_dates(last_date, steps):
        dates = []
        current_date = pd.Timestamp(last_date)
        for i in range(1, steps + 1):
            next_date = current_date + timedelta(days=i)
            dates.append(next_date.strftime('%Y-%m-%d'))
        return dates
    
    @staticmethod
    def generate_insights(forecast, data, metrics):
        current_price = float(data['Close'].iloc[-1])
        forecast_avg = float(np.mean(forecast))
        forecast_trend = 'upward' if forecast_avg > current_price else 'downward'
        
        expected_return = ((forecast[-1] - current_price) / current_price) * 100
        forecast_volatility = float(np.std(forecast))
        
        mape = metrics.get('mape', 100)
        if mape < 5:
            confidence_level = 'High'
        elif mape < 10:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'current_price': current_price,
            'forecast_average': forecast_avg,
            'trend': forecast_trend,
            'expected_return_percent': float(expected_return),
            'forecast_volatility': forecast_volatility,
            'model_confidence': confidence_level,
            'recommendation': 'This is a statistical forecast. Always do your own research before investing.'
        }

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>Time-Series Forecasting API</h1>")

@app.post("/api/fetch_data")
async def fetch_data(request: DataRequest):
    try:
        stock_data, error = DataRetriever.fetch_stock_data(
            request.ticker, request.start_date, request.end_date
        )
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        is_valid, message = DataRetriever.validate_data(stock_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        cleaned_data, cleaning_report = DataPreprocessor.clean_data(stock_data)
        
        response = {
            'success': True,
            'data': {
                'dates': cleaned_data.index.strftime('%Y-%m-%d').tolist(),
                'close': cleaned_data['Close'].tolist(),
                'volume': cleaned_data['Volume'].tolist(),
                'high': cleaned_data['High'].tolist(),
                'low': cleaned_data['Low'].tolist(),
                'open': cleaned_data['Open'].tolist()
            },
            'cleaning_report': cleaning_report,
            'total_records': len(cleaned_data)
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_data(request: DataRequest):
    try:
        stock_data, error = DataRetriever.fetch_stock_data(
            request.ticker, request.start_date, request.end_date
        )
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        cleaned_data, _ = DataPreprocessor.clean_data(stock_data)
        
        stats = EDAAnalyzer.basic_statistics(cleaned_data)
        stationarity = EDAAnalyzer.check_stationarity(cleaned_data)
        seasonality = EDAAnalyzer.analyze_seasonality(cleaned_data)
        acf_pacf = EDAAnalyzer.calculate_acf_pacf(cleaned_data)
        time_windows = EDAAnalyzer.time_window_analysis(cleaned_data)
        heatmap_data = EDAAnalyzer.generate_heatmap_data(cleaned_data)
        
        plot_data = {
            'dates': cleaned_data.index.strftime('%Y-%m-%d').tolist(),
            'close': cleaned_data['Close'].tolist()
        }
        
        response = {
            'success': True,
            'statistics': stats,
            'stationarity': stationarity,
            'seasonality': seasonality,
            'acf_pacf': acf_pacf,
            'time_windows': time_windows,
            'heatmap_data': heatmap_data,
            'plot_data': plot_data
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast")
async def forecast(request: ForecastRequest):
    try:
        period_map = {'day': 1, 'week': 7, 'month': 30, 'quarter': 90}
        steps = period_map.get(request.period, 7)
        
        stock_data, error = DataRetriever.fetch_stock_data(
            request.ticker, request.start_date, request.end_date
        )
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        cleaned_data, _ = DataPreprocessor.clean_data(stock_data)
        
        # Train model
        if request.model == 'ARIMA':
            forecast_values, metrics = ForecastingModels.train_arima(cleaned_data, steps=steps)
        elif request.model == 'SARIMA':
            forecast_values, metrics = ForecastingModels.train_sarima(cleaned_data, steps=steps)
        elif request.model == 'RandomForest':
            forecast_values, metrics = ForecastingModels.train_random_forest(cleaned_data, steps=steps)
        elif request.model == 'XGBoost':
            forecast_values, metrics = ForecastingModels.train_xgboost(cleaned_data, steps=steps)
        else:
            raise HTTPException(status_code=400, detail='Invalid model')
        
        if forecast_values is None:
            raise HTTPException(status_code=400, detail=metrics.get('error', 'Forecasting failed'))
        
        lower, upper = ForecastGenerator.calculate_confidence_interval(forecast_values, cleaned_data)
        last_date = cleaned_data.index[-1]
        forecast_dates = ForecastGenerator.generate_forecast_dates(last_date, steps)
        insights = ForecastGenerator.generate_insights(forecast_values, cleaned_data, metrics)
        
        historical_data = {
            'dates': cleaned_data.index[-60:].strftime('%Y-%m-%d').tolist(),
            'values': cleaned_data['Close'].iloc[-60:].tolist()
        }
        
        response = {
            'success': True,
            'forecast': {
                'dates': forecast_dates,
                'values': forecast_values.tolist(),
                'lower_bound': lower.tolist(),
                'upper_bound': upper.tolist()
            },
            'metrics': metrics,
            'insights': insights,
            'historical_data': historical_data,
            'model': request.model
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export_tableau_all")
async def export_tableau_all(request: MultiTickerRequest):
    try:
        all_data = []

        for ticker in request.tickers:
            stock_data, error = DataRetriever.fetch_stock_data(
                ticker, request.start_date, request.end_date
            )

            if error:
                continue

            cleaned_data, _ = DataPreprocessor.clean_data(stock_data)

            df = cleaned_data.copy()
            df['Date'] = df.index
            df['Ticker'] = ticker

            # Analytics-friendly fields
            df['MA_7'] = df['Close'].rolling(7, min_periods=1).mean()
            df['MA_30'] = df['Close'].rolling(30, min_periods=1).mean()
            df['Returns'] = df['Close'].pct_change()

            df = df.reset_index(drop=True)
            all_data.append(df)

        if not all_data:
            raise HTTPException(status_code=400, detail="No valid data fetched")

        final_df = pd.concat(all_data, ignore_index=True)
        csv_data = final_df.to_csv(index=False)

        return JSONResponse(content={
            "success": True,
            "filename": "stock_timeseries_all.csv",
            "csv_data": csv_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "xgboost_available": XGBOOST_AVAILABLE
    }

if __name__ == '__main__':
    import uvicorn
    print("Time-Series Forecasting Application")
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
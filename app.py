from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


app = FastAPI(title="Time-Series Forecasting API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request Models
# =========================
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


# =========================
# Data Retrieval
# =========================
class DataRetriever:
    @staticmethod
    def fetch_stock_data(ticker, start_date, end_date):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=30
            )
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
        if len(data) < 60:
            return False, "Insufficient data points (minimum 60 recommended)"
        if "Close" not in data.columns:
            return False, "Close column missing"
        return True, "Data validated successfully"


# =========================
# Preprocessing + Features
# =========================
class DataPreprocessor:
    @staticmethod
    def clean_data(data):
        df = data.copy()
        initial_missing = int(df.isnull().sum().sum())

        df = df.ffill()
        df = df.bfill()

        initial_len = len(df)
        df = df[~df.index.duplicated(keep="first")]
        duplicates_removed = int(initial_len - len(df))

        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Business-day index (Mon-Fri) + forward fill
        df = df.asfreq("B")
        df = df.ffill()

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)

        report = {
            "missing_values_filled": initial_missing,
            "duplicates_removed": duplicates_removed,
            "final_records": int(len(df)),
            "date_format": "YYYY-MM-DD HH:MM:SS"
        }
        return df, report

    @staticmethod
    def create_features(data):
        df = data.copy()

        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

        df["MA_5"] = df["Close"].rolling(window=5, min_periods=5).mean()
        df["MA_10"] = df["Close"].rolling(window=10, min_periods=10).mean()
        df["MA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()

        df["Volatility_5"] = df["Returns"].rolling(window=5, min_periods=5).std()
        df["Volatility_10"] = df["Returns"].rolling(window=10, min_periods=10).std()

        df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

        if "Volume" in df.columns:
            df["Volume_MA_5"] = df["Volume"].rolling(window=5, min_periods=5).mean()
            df["Volume_Change"] = df["Volume"].pct_change()

        df["DayOfWeek"] = df.index.dayofweek
        df["Month"] = df.index.month

        df["Lag_1"] = df["Close"].shift(1)
        df["Lag_2"] = df["Close"].shift(2)
        df["Lag_3"] = df["Close"].shift(3)
        df["Lag_5"] = df["Close"].shift(5)
        df["Lag_10"] = df["Close"].shift(10)

        return df


# =========================
# EDA
# =========================
class EDAAnalyzer:
    @staticmethod
    def basic_statistics(data):
        return {
            "mean": float(data["Close"].mean()),
            "median": float(data["Close"].median()),
            "std": float(data["Close"].std()),
            "min": float(data["Close"].min()),
            "max": float(data["Close"].max()),
            "range": float(data["Close"].max() - data["Close"].min()),
            "variance": float(data["Close"].var()),
            "skewness": float(data["Close"].skew()),
            "kurtosis": float(data["Close"].kurtosis())
        }

    @staticmethod
    def check_stationarity(data):
        series = data["Close"].dropna()
        result = adfuller(series)
        p_val = float(result[1])
        is_stationary = bool(p_val < 0.05)
        return {
            "adf_statistic": float(result[0]),
            "p_value": p_val,
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary": is_stationary,
            "interpretation": "Stationary" if is_stationary else "Non-stationary (differencing needed)"
        }

    @staticmethod
    def calculate_acf_pacf(data, lags=40):
        series = data["Close"].dropna()
        max_lags = min(lags, max(5, len(series) // 3))

        a = acf(series, nlags=max_lags)
        p = pacf(series, nlags=max_lags)

        ci = 1.96 / np.sqrt(len(series))

        sig_acf = []
        sig_pacf = []

        i = 1
        while i < len(a):
            if abs(a[i]) > ci:
                sig_acf.append(i)
            i += 1

        j = 1
        while j < len(p):
            if abs(p[j]) > ci:
                sig_pacf.append(j)
            j += 1

        return {
            "acf": a.tolist(),
            "pacf": p.tolist(),
            "confidence_interval": float(ci),
            "significant_acf_lags": sig_acf[:10],
            "significant_pacf_lags": sig_pacf[:10]
        }

    @staticmethod
    def suggest_arima_params(data):
        station = EDAAnalyzer.check_stationarity(data)
        d = 0
        if station["p_value"] > 0.05:
            d = 1

        acf_pacf = EDAAnalyzer.calculate_acf_pacf(data, lags=40)
        sig_acf = acf_pacf["significant_acf_lags"]
        sig_pacf = acf_pacf["significant_pacf_lags"]

        p = 1
        q = 1

        if len(sig_pacf) > 0:
            p = min(5, int(sig_pacf[0]))
        if len(sig_acf) > 0:
            q = min(5, int(sig_acf[0]))

        return (int(p), int(d), int(q))


# =========================
# Forecast Models
# =========================
class ForecastingModels:
    @staticmethod
    def _time_split(series, test_ratio=0.2):
        n = len(series)
        split_idx = int(n * (1 - test_ratio))
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        return train, test

    @staticmethod
    def _metrics(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        denom = np.where(y_true == 0, 1e-9, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

        return float(mae), float(rmse), float(mape)

    @staticmethod
    def infer_seasonal_period(index: pd.DatetimeIndex) -> int:
        # Infer frequency from median time delta
        if len(index) < 10:
            return 5

        diffs = index.to_series().diff().dropna()
        median_days = float(diffs.median() / pd.Timedelta(days=1))

        # Daily business data (with weekends missing) ~ 1-2 days median
        if median_days <= 2:
            return 5  # trading week
        # Weekly
        if 6 <= median_days <= 8:
            return 52
        # Monthly
        if 25 <= median_days <= 35:
            return 12

        return 5

    @staticmethod
    def auto_arima_fit(series, d_hint):
        best_aic = None
        best_order = None
        best_fit = None

        p_values = [0, 1, 2, 3]
        q_values = [0, 1, 2, 3]
        d_values = [d_hint]

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    if p == 0 and q == 0:
                        continue

                    # Trend choice: include constant only when not differencing
                    trend = "c" if d == 0 else "n"

                    try:
                        m = ARIMA(
                            series,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            trend=trend
                        )
                        f = m.fit(method_kwargs={"maxiter": 300})

                        aic = float(f.aic)
                        if best_aic is None or aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_fit = f
                    except:
                        pass

        if best_fit is None:
            trend = "c" if d_hint == 0 else "n"
            m = ARIMA(series, order=(1, d_hint, 1), trend=trend)
            best_fit = m.fit()
            best_order = (1, d_hint, 1)
            best_aic = float(best_fit.aic)

        return best_fit, best_order, best_aic

    @staticmethod
    def auto_sarima_fit(series, d_hint, seasonal_period=5):
        best_aic = None
        best_order = None
        best_seasonal = None
        best_fit = None

        p_values = [0, 1, 2]
        q_values = [0, 1, 2]
        d_values = [d_hint]
        P_values = [0, 1]
        Q_values = [0, 1]
        D_values = [0, 1]

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                try:
                                    m = SARIMAX(
                                        series,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    f = m.fit(disp=False, maxiter=300)

                                    aic = float(f.aic)
                                    if best_aic is None or aic < best_aic:
                                        best_aic = aic
                                        best_order = (p, d, q)
                                        best_seasonal = (P, D, Q, seasonal_period)
                                        best_fit = f
                                except:
                                    pass

        if best_fit is None:
            m = SARIMAX(
                series,
                order=(1, d_hint, 1),
                seasonal_order=(1, 0, 1, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            best_fit = m.fit(disp=False, maxiter=200)
            best_order = (1, d_hint, 1)
            best_seasonal = (1, 0, 1, seasonal_period)
            best_aic = float(best_fit.aic)

        return best_fit, best_order, best_seasonal, best_aic

    @staticmethod
    def train_arima(data, steps=30):
        series = data["Close"].dropna()
        train, test = ForecastingModels._time_split(series)

        p, d, q = EDAAnalyzer.suggest_arima_params(data)
        d_hint = d

        fit, order, aic = ForecastingModels.auto_arima_fit(train, d_hint=d_hint)

        pred_obj = fit.get_forecast(steps=len(test))
        pred = pred_obj.predicted_mean.values

        mae, rmse, mape = ForecastingModels._metrics(test.values, pred)

        final_fit, final_order, _ = ForecastingModels.auto_arima_fit(series, d_hint=d_hint)

        future_obj = final_fit.get_forecast(steps=steps)
        future_mean = future_obj.predicted_mean.values
        ci = future_obj.conf_int(alpha=0.05).values
        lower, upper = ci[:, 0], ci[:, 1]

        return future_mean, lower, upper, {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "aic": float(final_fit.aic),
            "bic": float(final_fit.bic),
            "order": tuple(final_order),
            "selection": "AIC grid search around EDA hints"
        }

    @staticmethod
    def train_sarima(data, steps=30):
        series = data["Close"].dropna()
        train, test = ForecastingModels._time_split(series)

        p, d, q = EDAAnalyzer.suggest_arima_params(data)
        d_hint = d

        seasonal_period = ForecastingModels.infer_seasonal_period(data.index)

        fit_train, order_train, seasonal_train, _ = ForecastingModels.auto_sarima_fit(
            train, d_hint=d_hint, seasonal_period=seasonal_period
        )

        pred_obj = fit_train.get_forecast(steps=len(test))
        pred = pred_obj.predicted_mean.values
        mae, rmse, mape = ForecastingModels._metrics(test.values, pred)

        fit_full, order_full, seasonal_full, _ = ForecastingModels.auto_sarima_fit(
            series, d_hint=d_hint, seasonal_period=seasonal_period
        )

        future_obj = fit_full.get_forecast(steps=steps)
        future_mean = future_obj.predicted_mean.values
        ci = future_obj.conf_int(alpha=0.05).values
        lower, upper = ci[:, 0], ci[:, 1]

        return future_mean, lower, upper, {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "aic": float(fit_full.aic),
            "bic": float(fit_full.bic),
            "order": tuple(order_full),
            "seasonal_order": tuple(seasonal_full),
            "seasonal_period_used": int(seasonal_period),
            "selection": "Train/Test for metrics, Refit on full for final forecast"
        }

    @staticmethod
    def _returns_vol_5(close_history):
        # std of last 5 returns
        if len(close_history) < 6:
            return 0.0
        rets = []
        i = 1
        while i <= 5:
            prev_val = close_history[-(i + 1)]
            cur_val = close_history[-i]
            if prev_val == 0:
                rets.append(0.0)
            else:
                rets.append((cur_val - prev_val) / prev_val)
            i += 1
        return float(np.std(rets))

    @staticmethod
    def train_random_forest(data, steps=30):
        df = DataPreprocessor.create_features(data)
        df["Target"] = df["Close"].shift(-1)

        feature_cols = [
            "Returns", "MA_5", "MA_10", "MA_20", "Volatility_5",
            "Momentum_5", "DayOfWeek", "Month", "Lag_1", "Lag_2", "Lag_3", "Lag_5"
        ]

        df = df.dropna()
        if len(df) < 200:
            raise ValueError("Not enough data after feature creation (need ~200+ rows).")

        X = df[feature_cols].values
        y = df["Target"].values

        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)

        mae, rmse, mape = ForecastingModels._metrics(y_test, test_pred)
        r2 = float(r2_score(y_test, test_pred))

        resid = y_test - test_pred
        resid_std = float(np.std(resid))

        close_history = data["Close"].dropna().tolist()
        forecast = []

        last_date = data.index[-1]

        for step in range(steps):
            current_close = float(close_history[-1])
            prev_close = float(close_history[-2]) if len(close_history) >= 2 else current_close

            returns = 0.0
            if prev_close != 0:
                returns = (current_close - prev_close) / prev_close

            ma_5 = float(np.mean(close_history[-5:])) if len(close_history) >= 5 else current_close
            ma_10 = float(np.mean(close_history[-10:])) if len(close_history) >= 10 else current_close
            ma_20 = float(np.mean(close_history[-20:])) if len(close_history) >= 20 else current_close

            vol_5 = ForecastingModels._returns_vol_5(close_history)

            momentum_5 = 0.0
            if len(close_history) >= 6:
                momentum_5 = float(close_history[-1] - close_history[-6])

            next_date = pd.bdate_range(last_date, periods=step + 2)[-1]
            dayofweek = int(next_date.dayofweek)
            month = int(next_date.month)

            lag_1 = float(close_history[-1])
            lag_2 = float(close_history[-2]) if len(close_history) >= 2 else lag_1
            lag_3 = float(close_history[-3]) if len(close_history) >= 3 else lag_1
            lag_5 = float(close_history[-6]) if len(close_history) >= 6 else lag_1

            feature_vector = np.array([[
                returns, ma_5, ma_10, ma_20, vol_5,
                momentum_5, dayofweek, month,
                lag_1, lag_2, lag_3, lag_5
            ]])

            pred = float(model.predict(feature_vector)[0])
            forecast.append(pred)
            close_history.append(pred)

        forecast = np.array(forecast)

        z = 1.96
        step_arr = np.arange(1, steps + 1)
        margin = z * resid_std * np.sqrt(step_arr)
        lower = forecast - margin
        upper = forecast + margin

        return forecast, lower, upper, {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2_score": r2,
            "residual_std": resid_std
        }

    @staticmethod
    def train_xgboost(data, steps=30):
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not installed")

        df = DataPreprocessor.create_features(data)
        df["Target"] = df["Close"].shift(-1)

        feature_cols = [
            "Returns", "MA_5", "MA_10", "MA_20", "Volatility_5",
            "Momentum_5", "DayOfWeek", "Month", "Lag_1", "Lag_2", "Lag_3", "Lag_5"
        ]

        df = df.dropna()
        if len(df) < 200:
            raise ValueError("Not enough data after feature creation (need ~200+ rows).")

        X = df[feature_cols].values
        y = df["Target"].values

        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)

        mae, rmse, mape = ForecastingModels._metrics(y_test, test_pred)
        r2 = float(r2_score(y_test, test_pred))

        resid = y_test - test_pred
        resid_std = float(np.std(resid))

        close_history = data["Close"].dropna().tolist()
        forecast = []

        last_date = data.index[-1]

        for step in range(steps):
            current_close = float(close_history[-1])
            prev_close = float(close_history[-2]) if len(close_history) >= 2 else current_close

            returns = 0.0
            if prev_close != 0:
                returns = (current_close - prev_close) / prev_close

            ma_5 = float(np.mean(close_history[-5:])) if len(close_history) >= 5 else current_close
            ma_10 = float(np.mean(close_history[-10:])) if len(close_history) >= 10 else current_close
            ma_20 = float(np.mean(close_history[-20:])) if len(close_history) >= 20 else current_close

            vol_5 = ForecastingModels._returns_vol_5(close_history)

            momentum_5 = 0.0
            if len(close_history) >= 6:
                momentum_5 = float(close_history[-1] - close_history[-6])

            next_date = pd.bdate_range(last_date, periods=step + 2)[-1]
            dayofweek = int(next_date.dayofweek)
            month = int(next_date.month)

            lag_1 = float(close_history[-1])
            lag_2 = float(close_history[-2]) if len(close_history) >= 2 else lag_1
            lag_3 = float(close_history[-3]) if len(close_history) >= 3 else lag_1
            lag_5 = float(close_history[-6]) if len(close_history) >= 6 else lag_1

            feature_vector = np.array([[
                returns, ma_5, ma_10, ma_20, vol_5,
                momentum_5, dayofweek, month,
                lag_1, lag_2, lag_3, lag_5
            ]])

            pred = float(model.predict(feature_vector)[0])
            forecast.append(pred)
            close_history.append(pred)

        forecast = np.array(forecast)

        z = 1.96
        step_arr = np.arange(1, steps + 1)
        margin = z * resid_std * np.sqrt(step_arr)
        lower = forecast - margin
        upper = forecast + margin

        return forecast, lower, upper, {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2_score": r2,
            "residual_std": resid_std
        }


# =========================
# Forecast Output Helpers
# =========================
class ForecastGenerator:
    @staticmethod
    def generate_forecast_dates(last_date, steps):
        rng = pd.bdate_range(pd.Timestamp(last_date), periods=steps + 1)
        out = []
        i = 1
        while i < len(rng):
            out.append(rng[i].strftime("%Y-%m-%d"))
            i += 1
        return out

    @staticmethod
    def generate_insights(forecast, data, metrics):
        forecast = np.asarray(forecast)
        current_price = float(data["Close"].iloc[-1])
        forecast_avg = float(np.mean(forecast))
        trend = "UPWARD" if forecast_avg > current_price else "DOWNWARD"
        expected_return = float(((forecast[-1] - current_price) / current_price) * 100)
        forecast_vol = float(np.std(forecast))

        mape = float(metrics.get("mape", 100.0))
        if mape < 5:
            conf = "High"
        elif mape < 10:
            conf = "Medium"
        else:
            conf = "Low"

        return {
            "current_price": current_price,
            "forecast_average": forecast_avg,
            "trend": trend,
            "expected_return_percent": expected_return,
            "forecast_volatility": forecast_vol,
            "model_confidence": conf,
            "note": "Forecast is probabilistic, not investment advice."
        }


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>Time-Series Forecasting API</h1>")


@app.post("/api/fetch_data")
async def fetch_data(request: DataRequest):
    stock_data, error = DataRetriever.fetch_stock_data(request.ticker, request.start_date, request.end_date)
    if error:
        raise HTTPException(status_code=400, detail=error)

    ok, msg = DataRetriever.validate_data(stock_data)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    cleaned, report = DataPreprocessor.clean_data(stock_data)

    return JSONResponse(content={
        "success": True,
        "data": {
            "dates": cleaned.index.strftime("%Y-%m-%d").tolist(),
            "close": cleaned["Close"].tolist(),
            "volume": cleaned["Volume"].tolist() if "Volume" in cleaned.columns else [],
            "high": cleaned["High"].tolist() if "High" in cleaned.columns else [],
            "low": cleaned["Low"].tolist() if "Low" in cleaned.columns else [],
            "open": cleaned["Open"].tolist() if "Open" in cleaned.columns else []
        },
        "cleaning_report": report,
        "total_records": int(len(cleaned))
    })


@app.post("/api/analyze")
async def analyze_data(request: DataRequest):
    stock_data, error = DataRetriever.fetch_stock_data(request.ticker, request.start_date, request.end_date)
    if error:
        raise HTTPException(status_code=400, detail=error)

    cleaned, _ = DataPreprocessor.clean_data(stock_data)

    stats = EDAAnalyzer.basic_statistics(cleaned)
    stationarity = EDAAnalyzer.check_stationarity(cleaned)
    acf_pacf = EDAAnalyzer.calculate_acf_pacf(cleaned)
    suggested_order = EDAAnalyzer.suggest_arima_params(cleaned)

    return JSONResponse(content={
        "success": True,
        "statistics": stats,
        "stationarity": stationarity,
        "acf_pacf": acf_pacf,
        "suggested_arima_order": {
            "p": suggested_order[0],
            "d": suggested_order[1],
            "q": suggested_order[2]
        },
        "plot_data": {
            "dates": cleaned.index.strftime("%Y-%m-%d").tolist(),
            "close": cleaned["Close"].tolist()
        }
    })


@app.post("/api/forecast")
async def forecast(request: ForecastRequest):
    # Trading-day horizons (better for stocks)
    period_map = {"day": 1, "week": 5, "month": 21, "quarter": 63}
    steps = int(period_map.get(request.period, 5))

    stock_data, error = DataRetriever.fetch_stock_data(request.ticker, request.start_date, request.end_date)
    if error:
        raise HTTPException(status_code=400, detail=error)

    cleaned, _ = DataPreprocessor.clean_data(stock_data)

    model_name = request.model.strip()

    try:
        if model_name == "ARIMA":
            vals, lower, upper, metrics = ForecastingModels.train_arima(cleaned, steps=steps)
        elif model_name == "SARIMA":
            vals, lower, upper, metrics = ForecastingModels.train_sarima(cleaned, steps=steps)
        elif model_name == "RandomForest":
            vals, lower, upper, metrics = ForecastingModels.train_random_forest(cleaned, steps=steps)
        elif model_name == "XGBoost":
            vals, lower, upper, metrics = ForecastingModels.train_xgboost(cleaned, steps=steps)
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Use ARIMA, SARIMA, RandomForest, XGBoost.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    last_date = cleaned.index[-1]
    forecast_dates = ForecastGenerator.generate_forecast_dates(last_date, steps)
    insights = ForecastGenerator.generate_insights(vals, cleaned, metrics)

    historical = {
        "dates": cleaned.index[-60:].strftime("%Y-%m-%d").tolist(),
        "values": cleaned["Close"].iloc[-60:].tolist()
    }

    return JSONResponse(content={
        "success": True,
        "model": model_name,
        "forecast": {
            "dates": forecast_dates,
            "values": vals.tolist(),
            "lower_bound": lower.tolist(),
            "upper_bound": upper.tolist()
        },
        "metrics": metrics,
        "insights": insights,
        "historical_data": historical
    })


@app.post("/api/export_tableau_all")
async def export_tableau_all(request: MultiTickerRequest):
    all_data = []
    for ticker in request.tickers:
        stock_data, error = DataRetriever.fetch_stock_data(ticker, request.start_date, request.end_date)
        if error:
            continue

        cleaned, _ = DataPreprocessor.clean_data(stock_data)

        df = cleaned.copy()
        df["Date"] = df.index
        df["Ticker"] = ticker
        df["MA_7"] = df["Close"].rolling(7, min_periods=7).mean()
        df["MA_30"] = df["Close"].rolling(30, min_periods=30).mean()
        df["Returns"] = df["Close"].pct_change()

        df = df.reset_index(drop=True)
        all_data.append(df)

    if len(all_data) == 0:
        raise HTTPException(status_code=400, detail="No valid data fetched")

    final_df = pd.concat(all_data, ignore_index=True)
    csv_data = final_df.to_csv(index=False)

    return JSONResponse(content={
        "success": True,
        "filename": "stock_timeseries_all.csv",
        "csv_data": csv_data
    })


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0", "xgboost_available": XGBOOST_AVAILABLE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

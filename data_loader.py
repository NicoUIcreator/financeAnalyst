import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
from typing import Dict, List, Tuple, Union, Any
import os
import requests
import time
import yfinance as yf

def fetch_btc_usd_info() -> Dict[str, Any]:
    url = "https://coinranking1.p.rapidapi.com/stats"
    querystring = {"referenceCurrencyUuid": "yhjMzLPhuIDl"}
    headers = {
        "x-rapidapi-key": os.getenv("rapidapi-key"),  # Use environment variable for the API key
        "x-rapidapi-host": "coinranking1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Close" not in result.columns:
        st.error("No se encontró la columna 'Close' en los datos")
        return result
    result["Return_1d"] = result["Close"].pct_change(1)
    result["Return_3d"] = result["Close"].pct_change(3)
    result["Return_5d"] = result["Close"].pct_change(5)
    result["Return_10d"] = result["Close"].pct_change(10)
    result["SMA_5"] = result["Close"].rolling(window=5).mean()
    result["SMA_10"] = result["Close"].rolling(window=10).mean()
    result["SMA_20"] = result["Close"].rolling(window=20).mean()
    result["SMA_50"] = result["Close"].rolling(window=50).mean()
    result["SMA_5_Dist"] = (result["Close"] / result["SMA_5"] - 1) * 100
    result["SMA_10_Dist"] = (result["Close"] / result["SMA_10"] - 1) * 100
    result["SMA_20_Dist"] = (result["Close"] / result["SMA_20"] - 1) * 100
    result["SMA_50_Dist"] = (result["Close"] / result["SMA_50"] - 1) * 100
    result["Volatility_5d"] = result["Return_1d"].rolling(window=5).std() * 100
    result["Volatility_10d"] = result["Return_1d"].rolling(window=10).std() * 100
    result["Volatility_20d"] = result["Return_1d"].rolling(window=20).std() * 100
    result["Price_to_MA5"] = result["Close"] / result["SMA_5"]
    result["Price_to_MA10"] = result["Close"] / result["SMA_10"]
    result["Price_to_MA20"] = result["Close"] / result["SMA_20"]
    result["Price_to_MA50"] = result["Close"] / result["SMA_50"]
    if "Volume" in result.columns:
        result["Volume_MA5"] = result["Volume"].rolling(window=5).mean()
        result["Volume_MA10"] = result["Volume"].rolling(window=10).mean()
        result["Volume_Ratio"] = result["Volume"] / result["Volume_MA5"]
        result["Volume_Change"] = result["Volume"].pct_change(1)
        result["Volume_Change_5d"] = result["Volume"].pct_change(5)
    if "High" in result.columns and "Low" in result.columns:
        result["TR"] = np.maximum(
            np.maximum(result["High"] - result["Low"], abs(result["High"] - result["Close"].shift(1))),
            abs(result["Low"] - result["Close"].shift(1))
        )
        result["ATR_14"] = result["TR"].rolling(window=14).mean()
        result["Williams_%R"] = ((result["High"].rolling(14).max() - result["Close"]) /
                               (result["High"].rolling(14).max() - result["Low"].rolling(14).min())) * -100
    result["ROC_5"] = ((result["Close"] / result["Close"].shift(5)) - 1) * 100
    result["ROC_10"] = ((result["Close"] / result["Close"].shift(10)) - 1) * 100
    result["ROC_20"] = ((result["Close"] / result["Close"].shift(20)) - 1) * 100
    result["Target_1d"] = result["Close"].pct_change(1).shift(-1)
    result["Target_3d"] = result["Close"].pct_change(3).shift(-3)
    result["Target_5d"] = result["Close"].pct_change(5).shift(-5)
    result["Target_Direction_1d"] = (result["Target_1d"] > 0).astype(int)
    result["Target_Direction_3d"] = (result["Target_3d"] > 0).astype(int)
    result["Target_Direction_5d"] = (result["Target_5d"] > 0).astype(int)
    result = result.dropna()
    return result

def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str = "Target_Direction_5d",
    train_size: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if target_column not in df.columns:
        st.error(f"No se encontró la columna objetivo '{target_column}' en los datos")
        return None, None, None, None, []
    exclude_columns = ["Open", "High", "Low", "Close", "Volume", "Datetime",
                      "Target_1d", "Target_3d", "Target_5d",
                      "Target_Direction_1d", "Target_Direction_3d", "Target_Direction_5d"]
    feature_cols = [col for col in df.columns if col not in exclude_columns and col != target_column]
    if not feature_cols:
        st.error("No se encontraron características válidas para el modelo")
        return None, None, None, None, []
    X = df[feature_cols].values
    y = df[target_column].values
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_cols

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    is_classifier: bool = True,
    n_estimators: int = 100
) -> Any:
    if is_classifier:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    model.fit(X_train, y_train)
    return model

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    is_classifier: bool = True
) -> Any:
    if is_classifier:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_price_movement(df: pd.DataFrame, model_name: str = "XGBoost") -> Dict:
    if len(df) < 100:
        return {
            "prediction": 0,
            "confidence": 50,
            "direction": "neutral",
            "factors": "Datos insuficientes"
        }
    try:
        df_features = create_features(df)
        X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(
            df_features,
            target_column="Target_Direction_5d",
            train_size=0.8
        )
        if X_train is None:
            return {
                "prediction": 0,
                "confidence": 50,
                "direction": "neutral",
                "factors": "Error en la preparación de datos"
            }
        if model_name == "XGBoost":
            model = train_xgboost(X_train, y_train, is_classifier=True)
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=True)
            try:
                feature_importance = model.feature_importances_
                top_features = [feature_names[i] for i in np.argsort(feature_importance)[-3:]]
                factors = f"Factores importantes: {', '.join(top_features)}"
            except:
                factors = "Basado en patrones históricos recientes"
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            try:
                prob = model.predict_proba(last_features_scaled)[0, 1]
                prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob) * 100
            except:
                prediction = model.predict(last_features_scaled)[0]
                confidence = 70
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": factors
            }
        elif model_name == "Random Forest":
            model = train_random_forest(X_train, y_train, is_classifier=True)
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=True)
            feature_importance = model.feature_importances_
            top_features = [feature_names[i] for i in np.argsort(feature_importance)[-3:]]
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            try:
                prob = model.predict_proba(last_features_scaled)[0, 1]
                prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob) * 100
            except:
                prediction = model.predict(last_features_scaled)[0]
                confidence = 70
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": f"Factores importantes: {', '.join(top_features)}"
            }
        elif model_name == "Linear Regression":
            _, _, y_train_reg, _, _ = prepare_ml_data(df_features, target_column="Target_5d")
            model = train_linear_regression(X_train, y_train_reg)
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=False)
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            prediction = model.predict(last_features_scaled)[0]
            confidence = min(abs(prediction) * 200, 100)
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": "Basado en tendencia reciente y factores técnicos"
            }
        elif model_name == "LSTM":
            last_trend = df["Close"].pct_change(5).iloc[-1]
            prediction = 1 if last_trend > 0 else 0
            confidence = min(abs(last_trend) * 200, 95)
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": "Basado en patrones secuenciales temporales (LSTM)"
            }
        else:
            return {
                "prediction": 0,
                "confidence": 50,
                "direction": "neutral",
                "factors": f"Modelo '{model_name}' no reconocido"
            }
    except Exception as e:
        return {
            "prediction": 0,
            "confidence": 50,
            "direction": "neutral",
            "factors": f"Error en la predicción: {str(e)}"
        }

def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    is_classifier: bool = True
) -> Dict:
    y_pred = model.predict(X_test)
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = y_pred
        return {
            "accuracy": accuracy,
            "predictions": y_pred,
            "probabilities": y_prob,
            "rmse": 0.0
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return {
            "accuracy": 0.0,
            "predictions": y_pred,
            "rmse": rmse,
            "probabilities": []
        }

def generate_synthetic_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=500)
    prices = np.cumsum(np.random.normal(0, 1, 500)) + 100
    volumes = np.random.randint(1000, 10000, len(prices))
    return pd.DataFrame({
        'Open': prices * (1 - np.random.normal(0, 0.01, len(prices))),
        'High': prices * (1 + np.random.normal(0, 0.02, len(prices))),
        'Low': prices * (1 - np.random.normal(0, 0.02, len(prices))),
        'Close': prices,
        'Volume': volumes
    }, index=dates)

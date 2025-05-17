import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
import os


def cargar_datos(archivo):
    return pd.read_csv(archivo)


def listar_bases_de_datos(directorio="data"):
    return [f for f in os.listdir(directorio) if f.endswith(".csv")]


def limpiar_binance_csv(df):
    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "").replace("%", "pct").replace(".", "") for c in df.columns]

    # Mapeo a nombres estándar
    rename_map = {
        "Date": "date",
        "Price": "price",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "vol",
        "Change %": "changepct"
    }

    df = df.rename(columns={col: rename_map[col] for col in df.columns if col in rename_map})

    if "date" not in df.columns:
        posibles_fechas = [col for col in df.columns if "date" in col or "fecha" in col]
        if posibles_fechas:
            df = df.rename(columns={posibles_fechas[0]: "date"})
        else:
            raise ValueError("No se encontró una columna de fecha válida en el archivo CSV.")

    # Procesar fechas
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df.sort_values("date").set_index("date")

    # Asegurar columnas requeridas
    columnas_necesarias = ["price", "open", "high", "low", "vol", "changepct"]
    for col in columnas_necesarias:
        if col not in df.columns:
            df[col] = np.nan

    df = df[columnas_necesarias]
    df = df.replace(to_replace=r",|%", value="", regex=True)
    df = df.replace({"M": "0000", "K": "0", "B": "0000000"}, regex=True)
    df["changepct"] = df["changepct"].astype(str).str.replace(",", ".")
    df = df.astype(float)

    return df.dropna()


def entrenar_modelo(df, modelo_tipo="lstm"):
    datos = df["price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(datos)
    secuencia = 30
    X, y = [], []
    for i in range(len(datos_escalados) - secuencia):
        X.append(datos_escalados[i:i+secuencia])
        y.append(datos_escalados[i+secuencia])
    X, y = np.array(X), np.array(y)

    if modelo_tipo == "lstm":
        modelo = Sequential()
        modelo.add(LSTM(50, return_sequences=True, input_shape=(secuencia, 1)))
        modelo.add(LSTM(50))
        modelo.add(Dense(1))
        modelo.compile(optimizer='adam', loss='mean_squared_error')
        modelo.fit(X, y, epochs=5, batch_size=32, verbose=0)
    elif modelo_tipo == "xgboost":
        modelo = XGBRegressor(objective='reg:squarederror')
        X = X.reshape((X.shape[0], X.shape[1]))
        modelo.fit(X, y)
    elif modelo_tipo == "linear":
        modelo = LinearRegression()
        X = X.reshape((X.shape[0], X.shape[1]))
        modelo.fit(X, y)
    else:
        raise ValueError("Modelo no soportado")

    X_test = X[-30:]
    y_test = y[-30:]

    return modelo, scaler, X_test, y_test


def predecir_precio(df, modelo, scaler, dias=1, modelo_tipo="lstm"):
    datos = df["price"].values.reshape(-1, 1)
    datos_escalados = scaler.transform(datos)
    secuencia = 30
    entrada = datos_escalados[-secuencia:].copy()
    predicciones = []

    for _ in range(dias):
        if modelo_tipo == "lstm":
            entrada_reshaped = np.expand_dims(entrada, axis=0)
            pred = modelo.predict(entrada_reshaped)
        else:
            entrada_reshaped = entrada.reshape((1, secuencia))
            pred = modelo.predict(entrada_reshaped).reshape(1, 1)

        predicciones.append(pred[0][0])
        entrada = np.vstack([entrada[1:], pred])

    predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
    return predicciones


def calcular_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return 100 - (mae / np.mean(y_true) * 100)

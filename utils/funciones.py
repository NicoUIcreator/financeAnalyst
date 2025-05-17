import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os


def cargar_datos(archivo):
    return pd.read_csv(archivo)


def listar_bases_de_datos(directorio="data"):
    return [f for f in os.listdir(directorio) if f.endswith(".csv")]


def limpiar_binance_csv(df):
    # Renombrar columnas desde inglés con símbolos a nombres estándar
    df = df.rename(columns={
        "Date": "date",
        "Price": "price",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "vol",
        "Change %": "changepct"
    })

    # Convertir fechas y ordenar
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df.sort_values("date").set_index("date")

    # Limpiar valores numéricos usando el método comprobado
    df = df[["price", "open", "high", "low", "vol", "changepct"]]
    df = df.replace(to_replace=(",|%"), value="", regex=True)
    df = df.replace(to_replace="M", value="0000", regex=True)
    df = df.replace(to_replace="K", value="0", regex=True)
    df = df.replace(to_replace="B", value="0000000", regex=True)
    df = df.astype(float)

    return df.dropna()


def entrenar_modelo(df):
    datos = df["price"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(datos)

    secuencia = 30
    X, y = [], []
    for i in range(len(datos_escalados) - secuencia):
        X.append(datos_escalados[i:i+secuencia])
        y.append(datos_escalados[i+secuencia])
    X, y = np.array(X), np.array(y)

    modelo = Sequential()
    modelo.add(LSTM(50, return_sequences=True, input_shape=(secuencia, 1)))
    modelo.add(LSTM(50))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return modelo, scaler, X_test := X[-30:], y_test := y[-30:]


def predecir_precio(df, modelo, scaler, dias=1):
    datos = df["price"].values.reshape(-1, 1)
    datos_escalados = scaler.transform(datos)
    secuencia = 30
    entrada = datos_escalados[-secuencia:].copy()

    predicciones = []
    for _ in range(dias):
        entrada_reshaped = np.expand_dims(entrada, axis=0)
        pred = model.predict(entrada_reshaped)
        predicciones.append(pred[0][0])
        entrada = np.vstack([entrada[1:], pred])

    predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
    return predicciones


def calcular_accuracy(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    return 100 - (mae / np.mean(y_true) * 100)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def cargar_datos(archivo):
    return pd.read_csv(archivo)


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

    return modelo, scaler


def predecir_precio(df, modelo, scaler):
    datos = df["price"].values.reshape(-1, 1)
    datos_escalados = scaler.transform(datos)
    secuencia = 30
    ultima_secuencia = datos_escalados[-secuencia:]
    ultima_secuencia = np.expand_dims(ultima_secuencia, axis=0)
    prediccion = modelo.predict(ultima_secuencia)
    precio_predicho = scaler.inverse_transform(prediccion)[0][0]
    precio_actual = datos[-1][0]
    cambio_pct = ((precio_predicho - precio_actual) / precio_actual) * 100
    return precio_actual, precio_predicho, cambio_pct

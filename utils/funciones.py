import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def cargar_datos(archivo):
    return pd.read_csv(archivo)


def limpiar_dataframe(df):
    # Renombrar columnas en formato consistente
    df.columns = [col.strip().lower().replace(" ", "").replace("%", "pct").replace(".", "") for col in df.columns]

    # Asegurar existencia de columnas necesarias
    required_cols = ["date", "price", "open", "high", "low", "vol", "changepct"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Faltan columnas necesarias. Columnas actuales: {df.columns.tolist()}")

    # Parsear fechas y ordenar
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df.sort_values("date").set_index("date")

    # Limpieza rápida de texto (coma, %, K, M, B)
    for col in ["price", "open", "high", "low", "vol", "changepct"]:
        df[col] = (
            df[col]
            .astype(str)
            .replace({",": "", "%": "", "M": "0000", "K": "0", "B": "0000000"}, regex=True)
        )

    # Convertir a float
    for col in ["price", "open", "high", "low", "vol", "changepct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


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
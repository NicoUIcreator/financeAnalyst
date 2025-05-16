import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

def cargar_datos(archivo):
    df = pd.read_csv(archivo, decimal=",", thousands=".")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def entrenar_modelo(df):
    datos = df["price"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float).values.reshape(-1, 1)
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

    # Guardar el modelo y el scaler
    if not os.path.exists("utils/modelos"):
        os.makedirs("utils/modelos")
    joblib.dump(modelo, "utils/modelos/modelo_lstm.pkl")
    joblib.dump(scaler, "utils/modelos/scaler.pkl")

    return modelo, scaler

def predecir_precio(df, modelo, scaler):
    datos = df["price"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float).values.reshape(-1, 1)
    datos_escalados = scaler.transform(datos)
    secuencia = 30
    ultima_secuencia = datos_escalados[-secuencia:]
    ultima_secuencia = np.expand_dims(ultima_secuencia, axis=0)
    prediccion = modelo.predict(ultima_secuencia)
    precio_predicho = scaler.inverse_transform(prediccion)[0][0]
    precio_actual = datos[-1][0]
    cambio_pct = ((precio_predicho - precio_actual) / precio_actual) * 100
    return precio_actual, precio_predicho, cambio_pct

def limpiar_dataframe(df):
    # 1. Renombrar columnas a formato estándar
    df.columns = [col.strip().lower().replace(" ", "").replace("%", "pct").replace(".", "") for col in df.columns]

    # 2. Parsear fechas y establecer como índice
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df.sort_values("date").set_index("date")

    # 3. Columnas numéricas con coma decimal y punto de miles
    columnas_numericas = ["price", "open", "high", "low"]
    for col in columnas_numericas:
        df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Limpiar volumen (vol) con K/M/B
    def convertir_volumen(val):
        try:
            val = val.replace(",", "").upper()
            if "K" in val:
                return float(val.replace("K", "")) * 1_000
            elif "M" in val:
                return float(val.replace("M", "")) * 1_000_000
            elif "B" in val:
                return float(val.replace("B", "")) * 1_000_000_000
            else:
                return float(val)
        except:
            return np.nan

    df["vol"] = df["vol"].astype(str).apply(convertir_volumen)

    # 5. Limpiar cambio porcentual
    df["changepct"] = df["changepct"].astype(str).str.replace("%", "").str.replace(",", ".")
    df["changepct"] = pd.to_numeric(df["changepct"], errors="coerce")

    # 6. Eliminar registros con valores faltantes
    df = df.dropna()

    return df

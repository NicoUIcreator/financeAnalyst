import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from lightweight_charts import Chart

# Configuración de la página
st.set_page_config(page_title="Predicción BTC-USD", layout="wide")

st.title("📈 Predicción de Precio BTC-USD")
st.markdown("Sube un archivo `.csv` con las columnas: Date, Price, Open, High, Low, Vol., Change %")

# Carga del archivo CSV
uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Intentar leer el archivo con diferentes configuraciones
        try:
            df = pd.read_csv(uploaded_file, decimal=",", thousands=".", encoding="utf-8")
        except Exception:
            df = pd.read_csv(uploaded_file, decimal=",", thousands=".", encoding="latin1")

        # Normalizar nombres de columnas
        df.columns = [col.strip().lower().replace(" ", "").replace("%", "pct") for col in df.columns]

        # Mapear nombres esperados
        column_mapping = {
            "date": "date",
            "price": "price",
            "open": "open",
            "high": "high",
            "low": "low",
            "vol.": "vol",
            "vol": "vol",
            "change%": "changepct",
            "changepct": "changepct"
        }

        df.rename(columns=column_mapping, inplace=True)

        # Verificar que las columnas necesarias estén presentes
        required_columns = ["date", "price", "open", "high", "low", "vol", "changepct"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan las siguientes columnas en el archivo CSV: {', '.join(missing_cols)}")
        else:
            # Convertir la columna 'date' a formato datetime
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Visualización de los datos originales
            st.subheader("Datos Históricos")
            st.dataframe(df.tail())

            # Preparación de los datos para el modelo
            data = df["price"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float).values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            # Creación de secuencias para LSTM
            def create_sequences(data, seq_length):
                X = []
                y = []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(X), np.array(y)

            seq_length = 30
            X, y = create_sequences(scaled_data, seq_length)

            # División en entrenamiento y prueba
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Construcción del modelo
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenamiento del modelo
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            # Predicción
            last_sequence = scaled_data[-seq_length:]
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction = model.predict(last_sequence)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

            # Cálculo del cambio porcentual
            current_price = data[-1][0]
            change_percent = ((predicted_price - current_price) / current_price) * 100

            # Visualización de la predicción
            st.subheader("Predicción")
            st.markdown(f"Precio actual: ${current_price:.2f}")
            st.markdown(f"Precio predicho: ${predicted_price:.2f}")
            if change_percent >= 0:
                st.markdown(f"🔺 Cambio esperado: +{change_percent:.2f}%", unsafe_allow_html=True)
            else:
                st.markdown(f"🔻 Cambio esperado: {change_percent:.2f}%", unsafe_allow_html=True)

            # Preparación de datos para el gráfico
            chart_data = df[["date", "open", "high", "low", "price"]].copy()
            chart_data.rename(columns={
                "date": "time",
                "open": "open",
                "high": "high",
                "low": "low",
                "price": "close"
            }, inplace=True)
            chart_data["time"] = chart_data["time"].dt.strftime('%Y-%m-%d')

            # Agregar la predicción al gráfico
            future_date = (df["date"].iloc[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            new_row = pd.DataFrame({
                "time": [future_date],
                "open": [np.nan],
                "high": [np.nan],
                "low": [np.nan],
                "close": [predicted_price]
            })
            chart_data = pd.concat([chart_data, new_row], ignore_index=True)

            # Visualización del gráfico
            st.subheader("Gráfico de Precios")
            chart = Chart()
            chart.set(chart_data)
            chart.show()

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

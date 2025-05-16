import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from lightweight_charts import Chart

# Configuración de la página
st.set_page_config(page_title="Predicción BTC-USD", layout="wide")

st.title("📈 Predicción de Precio BTC-USD")
st.markdown("Sube un archivo `.csv` con las columnas: Date, Price, Open, High, Low, Vol., Change %")

# Carga del archivo CSV
uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type="csv")

if uploaded_file is not None:
    # Lectura del archivo
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Visualización de los datos originales
    st.subheader("Datos Históricos")
    st.dataframe(df.tail())

    # Preparación de los datos para el modelo
    data = df['Price'].values.reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
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
    current_price = df['Price'].iloc[-1]
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
    chart_data = df[['Date', 'Open', 'High', 'Low', 'Price']].copy()
    chart_data.rename(columns={'Date': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Price': 'close'}, inplace=True)
    chart_data['time'] = chart_data['time'].dt.strftime('%Y-%m-%d')

    # Agregar la predicción al gráfico
    future_date = (df['Date'].iloc[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    new_row = pd.DataFrame({
        'time': [future_date],
        'open': [np.nan],
        'high': [np.nan],
        'low': [np.nan],
        'close': [predicted_price]
    })
    chart_data = pd.concat([chart_data, new_row], ignore_index=True)

    # Visualización del gráfico
    st.subheader("Gráfico de Precios")
    chart = Chart()
    chart.set(chart_data)
    chart.show()
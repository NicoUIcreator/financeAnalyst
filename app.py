import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.funciones import (
    listar_bases_de_datos,
    cargar_datos,
    limpiar_binance_csv,
    entrenar_modelo,
    predecir_precio,
    calcular_accuracy
)

st.set_page_config(page_title="Predicción BTC", layout="wide")

st.title("📈 Predicción de Precio de BTC")

# Selección de base de datos desde la carpeta data
st.sidebar.header("Base de datos")
bases = listar_bases_de_datos()
archivo_seleccionado = st.sidebar.selectbox("Selecciona un archivo de /data", bases)

# Selección de modelo
st.sidebar.header("Modelo de predicción")
modelo_tipo = st.sidebar.selectbox("Selecciona el modelo", ["lstm", "xgboost", "linear"])

if archivo_seleccionado:
    df = cargar_datos(f"data/{archivo_seleccionado}")
    df = limpiar_binance_csv(df)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.tail())

    st.subheader("Gráfico de precios históricos")
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["price"], label="Histórico")
    plt.title("Precio BTC")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

    # Entrenamiento y predicción
    modelo, scaler, X_test, y_test = entrenar_modelo(df, modelo_tipo=modelo_tipo)
    pred_30 = predecir_precio(df, modelo, scaler, dias=30, modelo_tipo=modelo_tipo)
    pred_90 = predecir_precio(df, modelo, scaler, dias=90, modelo_tipo=modelo_tipo)

    # Accuracy
    y_pred_test = modelo.predict(X_test)
    if modelo_tipo != "lstm":
        y_pred_test = y_pred_test.reshape(-1, 1)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test).flatten()
    ultimos_reales = df["price"].values[-30:]
    acc = calcular_accuracy(ultimos_reales, y_pred_test_inv)

    st.subheader("Predicción")
    st.write(f"🔮 Accuracy estimada sobre últimos datos reales: {acc:.2f}%")

    fechas_futuras_30 = pd.date_range(df.index[-1], periods=31, freq='D')[1:]
    fechas_futuras_90 = pd.date_range(df.index[-1], periods=91, freq='D')[1:]

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["price"], label="Histórico")
    plt.plot(fechas_futuras_30, pred_30, label="Predicción 30 días")
    plt.plot(fechas_futuras_90, pred_90, label="Predicción 90 días", linestyle='--')
    plt.title("Predicción futura del precio BTC")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

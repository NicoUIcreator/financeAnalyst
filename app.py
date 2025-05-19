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
from utils.generar_reporte import mostrar_reporte_markdown

st.set_page_config(page_title="Predicción BTC", layout="wide")

st.title("📈 Predicción Inteligente del Precio de BTC")
st.image("utils/img/bitcoin-btc-logo.svg", width=100)
st.markdown("""
Bienvenido a **CryptoForecast**, una herramienta de predicción del precio de Bitcoin (BTC) basada en aprendizaje automático.
Carga tus propios datasets, límpialos automáticamente, visualiza tendencias históricas, selecciona modelos y genera predicciones.
""")

seccion = st.sidebar.radio("🧭 Ir a la sección:", [
    "📂 Carga y limpieza de datos",
    "📊 Visualización de datos",
    "🤖 Predicción y resultados",
    "📋 Comparación y exportación"
])

bases = listar_bases_de_datos()
archivo_seleccionado = st.sidebar.selectbox("Selecciona un archivo CSV", bases)

if "df" not in st.session_state:
    st.session_state.df = None
if "df_limpio" not in st.session_state:
    st.session_state.df_limpio = None

if archivo_seleccionado:
    try:
        df_cargado = cargar_datos(f"data/{archivo_seleccionado}")
        df_limpio = limpiar_binance_csv(df_cargado)
        st.session_state.df = df_limpio
        st.session_state.df_limpio = df_limpio.copy()
    except Exception:
        st.session_state.df = None
        st.session_state.df_limpio = None

# SECCIÓN 1
if seccion == "📂 Carga y limpieza de datos":
    st.header("📂 Carga y limpieza de datos")
    if archivo_seleccionado:
        st.subheader("Vista previa original")
        st.dataframe(df_cargado.head())
        if st.session_state.df_limpio is not None:
            st.success("✅ Datos limpiados correctamente.")
            st.download_button(
                label="📥 Descargar CSV limpio",
                data=st.session_state.df_limpio.to_csv().encode("utf-8"),
                file_name=f"limpio_{archivo_seleccionado}",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No se pudo limpiar este archivo.")
    else:
        st.info("Por favor selecciona un archivo.")

# SECCIÓN 2
elif seccion == "📊 Visualización de datos":
    st.header("📊 Visualización")
    df = st.session_state.df
    if df is not None:
        st.subheader("Vista previa")
        st.dataframe(df.head())
        st.subheader("Gráfico de precios históricos")
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["price"], label="Precio BTC")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("⚠️ No se ha cargado un dataframe válido.")

# SECCIÓN 3
elif seccion == "🤖 Predicción y resultados":
    st.header("🤖 Predicción")

    df = st.session_state.df
    if df is not None:
        st.markdown("Selecciona un modelo de predicción y presiona **Entrenar modelo**.")

        modelo_tipo = st.selectbox("Modelo a utilizar", ["lstm", "xgboost", "linear"], key="selector_modelo")

        if st.button("🚀 Entrenar modelo"):
            progreso = st.progress(0, text="⏳ Entrenando modelo...")
            modelo, scaler, X_test, y_test = entrenar_modelo(df, modelo_tipo=modelo_tipo)
            progreso.progress(50, text="⏳ Calculando predicción...")

            pred_30 = predecir_precio(df, modelo, scaler, dias=30, modelo_tipo=modelo_tipo)
            pred_90 = predecir_precio(df, modelo, scaler, dias=90, modelo_tipo=modelo_tipo)

            y_pred_test = modelo.predict(X_test)
            if modelo_tipo != "lstm":
                y_pred_test = y_pred_test.reshape(-1, 1)

            y_pred_test_inv = scaler.inverse_transform(y_pred_test).flatten()
            ultimos_reales = df["price"].values[-30:]
            acc = calcular_accuracy(ultimos_reales, y_pred_test_inv)

            st.success(f"✅ Accuracy del modelo ({modelo_tipo}): **{acc:.2f}%**")
            progreso.progress(100, text="✅ Predicción completa.")

            fechas_30 = pd.date_range(df.index[-1], periods=31, freq='D')[1:]
            fechas_90 = pd.date_range(df.index[-1], periods=91, freq='D')[1:]

            st.subheader("📈 Predicción futura")
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df["price"], label="Histórico")
            plt.plot(fechas_30, pred_30, label="Predicción 30 días")
            plt.plot(fechas_90, pred_90, label="Predicción 90 días", linestyle="--")
            plt.legend()
            st.pyplot(plt)

            st.session_state.pred_30 = pd.Series(pred_30, index=fechas_30)
            st.session_state.pred_90 = pd.Series(pred_90, index=fechas_90)
            st.session_state.modelo_tipo = modelo_tipo
            st.session_state.accuracy = acc
    else:
        st.warning("⚠️ Primero debes cargar y limpiar un dataset.")

# SECCIÓN 4
elif seccion == "📋 Comparación y exportación":
    st.header("📋 Comparar resultados y exportar")
    if "pred_30" in st.session_state and "pred_90" in st.session_state:
        st.subheader("Resumen del modelo")
        st.write(f"🧠 Modelo: `{st.session_state.modelo_tipo}`")
        st.write(f"🎯 Accuracy: `{st.session_state.accuracy:.2f}%`")
        st.subheader("📉 Predicción 30 días")
        st.line_chart(st.session_state.pred_30)
        st.download_button(
            "📤 Descargar predicción 30 días",
            data=st.session_state.pred_30.to_csv().encode("utf-8"),
            file_name="prediccion_30_dias.csv"
        )
        st.subheader("📉 Predicción 90 días")
        st.line_chart(st.session_state.pred_90)
        st.download_button(
            "📤 Descargar predicción 90 días",
            data=st.session_state.pred_90.to_csv().encode("utf-8"),
            file_name="prediccion_90_dias.csv"
        )

        # Informe en Markdown en pantalla
        st.subheader("🧾 Informe en Markdown")
        mostrar_reporte_markdown(
            st.session_state.df_limpio,
            st.session_state.pred_30,
            st.session_state.pred_90,
            st.session_state.modelo_tipo,
            st.session_state.accuracy
        )
    else:
        st.info("No hay predicciones generadas aún. Ve a la sección de predicción primero.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.funciones import cargar_datos, limpiar_dataframe, entrenar_modelo, predecir_precio

# Configuración de la página
st.set_page_config(page_title="Predicción BTC-USD", layout="wide")

# Barra lateral de navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Ir a", ["Exploración de Datos", "Predicción de Precios"])

# Sección 1: Exploración de datos
if opcion == "Exploración de Datos":
    st.title("📊 Exploración de Datos BTC-USD")
    archivo = st.file_uploader("Sube tu archivo CSV", type="csv")

    if archivo is not None:
        # Cargar y limpiar datos
        df = cargar_datos(archivo)
        df = limpiar_dataframe(df)

        # Mostrar nombres de columnas para depuración
        st.write("Columnas del DataFrame:", df.columns.tolist())

        # Vista previa de datos
        st.subheader("Vista Previa de los Datos")
        st.dataframe(df.head())

        # Gráfico de líneas
        st.subheader("Gráfico de Precio a lo Largo del Tiempo")
        plt.figure(figsize=(10, 4))
        sns.lineplot(data=df, x=df.index, y="price")
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Sección 2: Predicción de precios
elif opcion == "Predicción de Precios":
    st.title("🤖 Predicción de Precio BTC-USD")
    archivo = st.file_uploader("Sube tu archivo CSV", type="csv")

    if archivo is not None:
        df = cargar_datos(archivo)
        df = limpiar_dataframe(df)

        # Entrenamiento del modelo y predicción
        modelo, scaler = entrenar_modelo(df)
        precio_actual, precio_predicho, cambio_pct = predecir_precio(df, modelo, scaler)

        st.subheader("Resultado de la Predicción")
        st.write(f"📉 Precio actual: ${precio_actual:,.2f}")
        st.write(f"📈 Precio predicho: ${precio_predicho:,.2f}")
        st.write(f"📊 Cambio porcentual esperado: {cambio_pct:.2f}%")

        # Gráfico comparativo
        st.subheader("Gráfico de Predicción")
        fechas = df.index
        precios = df["price"]
        plt.figure(figsize=(10, 4))
        plt.plot(fechas, precios, label="Histórico")
        plt.plot(fechas.iloc[-1] + pd.Timedelta(days=1), precio_predicho, 'ro', label="Predicción")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

import streamlit as st

# ✅ Set page config as the first line
st.set_page_config(
    page_title="Trading Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from data_loader import (
    create_features,
    prepare_ml_data,
    train_random_forest,
    train_xgboost,
    train_linear_regression,
    predict_price_movement,
    generate_synthetic_data
)
import time
from datetime import datetime, timedelta

# Sidebar for configuration
st.sidebar.header("Configuración")

# Data loading
data_option = st.sidebar.radio("Seleccione fuente de datos", ["Datos de ejemplo", "Subir archivo CSV"])
df = pd.DataFrame()

if data_option == "Datos de ejemplo":
    df = generate_synthetic_data()
else:
    uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                st.error("El archivo debe contener las columnas: Open, High, Low, Close")
                df = pd.DataFrame()
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
            df = pd.DataFrame()

# Main app logic
if not df.empty:
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # Default value if missing

    st.subheader("Datos cargados")
    st.write(df.head())

    # Tabs for visualization and predictions
    tab1, tab2, tab3 = st.tabs(["Precios", "Análisis Técnico", "Predicciones"])

    with tab1:
        st.line_chart(df['Close'])

    with tab2:
        if len(df) > 50:
            df_features = create_features(df)
            indicator = st.selectbox("Seleccionar indicador", ["SMA", "Volatilidad", "ROC", "Williams %R"])
            if indicator == "SMA":
                chart_data = df_features[['Close', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50']].dropna()
                st.line_chart(chart_data)
            elif indicator == "Volatilidad":
                chart_data = df_features[['Volatility_5d', 'Volatility_10d', 'Volatility_20d']].dropna()
                st.line_chart(chart_data)
            elif indicator == "ROC":
                chart_data = df_features[['ROC_5', 'ROC_10', 'ROC_20']].dropna()
                st.line_chart(chart_data)
            elif indicator == "Williams %R":
                if "Williams_%R" in df_features.columns:
                    chart_data = df_features['Williams_%R'].dropna()
                    st.line_chart(chart_data)
                else:
                    st.warning("No se pudo calcular Williams %R. Se requieren columnas High y Low.")

    with tab3:
        st.subheader("Predicción de Movimiento de Precios")
        model_name = st.selectbox("Seleccionar modelo", ["XGBoost", "Random Forest", "Linear Regression", "LSTM"])
        if st.button("Realizar predicción"):
            with st.spinner("Entrenando modelo y generando predicción..."):
                prediction_results = predict_price_movement(df, model_name)
                direction = prediction_results["direction"]
                confidence = prediction_results["confidence"]
                factors = prediction_results["factors"]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dirección prevista", direction.upper(), delta="↑" if direction == "alza" else "↓" if direction == "baja" else "→")
                with col2:
                    st.metric("Confianza", f"{confidence:.2f}%")

                st.write(f"**Factores influyentes:** {factors}")

                if model_name != "LSTM":
                    df_features = create_features(df)
                    X_train, X_test, y_train, y_test, _ = prepare_ml_data(
                        df_features,
                        target_column="Target_Direction_5d" if model_name != "Linear Regression" else "Target_5d",
                        train_size=0.8
                    )

                    if model_name == "XGBoost":
                        model = train_xgboost(X_train, y_train, is_classifier=True)
                    elif model_name == "Random Forest":
                        model = train_random_forest(X_train, y_train, is_classifier=True)
                    else:
                        model = train_linear_regression(X_train, y_train)

                    eval_results = predict_price_movement(df, model_name)
                    st.subheader("Rendimiento del modelo en datos históricos")
                    if model_name != "Linear Regression":
                        st.metric("Precisión en datos de prueba", f"{eval_results['accuracy']:.2%}")
                        y_pred = eval_results["predictions"]
                        true_positives = np.sum((y_test == 1) & (y_pred == 1))
                        true_negatives = np.sum((y_test == 0) & (y_pred == 0))
                        false_positives = np.sum((y_test == 0) & (y_pred == 1))
                        false_negatives = np.sum((y_test == 1) & (y_pred == 0))
                        st.text("Matriz de confusión simplificada:")
                        st.text(f"Aciertos en alza: {true_positives}")
                        st.text(f"Aciertos en baja: {true_negatives}")
                        st.text(f"Falsos positivos: {false_positives}")
                        st.text(f"Falsos negativos: {false_negatives}")
                    else:
                        st.metric("Error cuadrático medio (RMSE)", f"{eval_results['rmse']:.4f}")
                        pred_df = pd.DataFrame({'Real': y_test, 'Predicción': eval_results['predictions']})
                        st.line_chart(pred_df)
else:
    st.warning("Por favor, cargue datos o use datos de ejemplo.")
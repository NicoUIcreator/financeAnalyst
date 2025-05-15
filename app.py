import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split # Opcional para validación, no para predicción de series temporales directas
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
# from joblib import dump, load # Para guardar/cargar modelos localmente, no directamente en Streamlit Cloud sin almacenamiento persistente

# --- Configuración de la Página ---
st.set_page_config(page_title="Predicción BTC-USD con XGBoost", layout="wide")

# --- Título y Descripción ---
st.title("🔮 Predicción de Precio BTC-USD con XGBoost (Ilustrativa)")
st.markdown("""
Sube un archivo CSV con el historial de precios de BTC-USD.
El archivo CSV debe contener al menos: `Date`, `Price`, `Open`, `High`, `Low`.
Esta app entrenará un modelo XGBoost simple para predecir precios futuros.
*(**Nota**: Esta es una demostración con un modelo simplificado.
La ingeniería de características y la validación del modelo son básicas para fines ilustrativos.)*
""")

# --- Función para Crear Características (Feature Engineering) ---
def create_features(df, label_col='Price', window_size=5):
    """
    Crea características de series temporales (lags) para el modelo.
    """
    df_featured = df.copy()
    for i in range(1, window_size + 1):
        df_featured[f'lag_{i}'] = df_featured[label_col].shift(i)
    df_featured = df_featured.dropna()
    return df_featured

# --- Carga de Archivo ---
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV de BTC-USD aquí", type="csv")

if uploaded_file is not None:
    try:
        data_orig = pd.read_csv(uploaded_file)
        data = data_orig.copy()

        required_columns = ["Date", "Price", "Open", "High", "Low"]
        if not all(col in data.columns for col in required_columns):
            st.error(f"El archivo CSV debe contener las columnas: {', '.join(required_columns)}")
        else:
            st.success("¡Archivo CSV cargado y validado exitosamente!")

            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
            data = data.set_index('Date') # Importante para el manejo de series temporales

            # --- Preparación de Datos y Feature Engineering ---
            st.subheader("🛠️ Preparación de Datos y Modelo")
            window_size = st.slider("Tamaño de ventana para lags (días previos como características):", 1, 30, 10)
            
            # Usaremos 'Price' para predecir 'Price'
            price_data = data[['Price']].copy()

            # Escalar los datos (XGBoost puede beneficiarse de esto)
            scaler = MinMaxScaler()
            price_data['Price_Scaled'] = scaler.fit_transform(price_data[['Price']])

            # Crear características
            df_featured = create_features(price_data, label_col='Price_Scaled', window_size=window_size)

            if df_featured.empty or len(df_featured) < window_size + 1: # Asegurar suficientes datos para entrenar
                st.warning(f"No hay suficientes datos para el tamaño de ventana seleccionado ({window_size}). Necesitas al menos {window_size*2 +1} filas después de eliminar NaNs.")
            else:
                # Separar características (X) y objetivo (y)
                X = df_featured.drop(['Price', 'Price_Scaled'], axis=1)
                y = df_featured['Price_Scaled']

                # --- Entrenamiento del Modelo XGBoost ---
                if st.button("🚀 Entrenar Modelo XGBoost y Predecir"):
                    with st.spinner("Entrenando modelo XGBoost... Esto puede tardar un momento."):
                        model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=100,      # Número de árboles (ajustable)
                            learning_rate=0.05,    # Tasa de aprendizaje (ajustable)
                            max_depth=5,           # Profundidad máxima del árbol (ajustable)
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                        )
                        model.fit(X, y)
                        st.session_state.model = model # Guardar modelo en estado de sesión
                        st.session_state.scaler = scaler # Guardar scaler
                        st.session_state.window_size = window_size # Guardar window_size
                        st.session_state.last_known_scaled_values = X.iloc[-1].values.reshape(1, -1) # Últimos lags escalados conocidos
                        st.session_state.last_known_actual_price = price_data['Price'].iloc[-1] # Último precio real conocido
                        st.session_state.last_known_date = price_data.index[-1] # Última fecha conocida
                        
                        st.success("¡Modelo XGBoost entrenado exitosamente!")

                # --- Predicción y Visualización (si el modelo está entrenado) ---
                if 'model' in st.session_state:
                    st.subheader("📈 Gráfico de Precios Históricos y Predicción con XGBoost")
                    
                    model = st.session_state.model
                    scaler = st.session_state.scaler
                    window_size = st.session_state.window_size
                    last_known_scaled_lags = price_data['Price_Scaled'].tail(window_size).values # Usar los últimos 'window_size' valores escalados reales
                    
                    # Número de días a predecir en el futuro
                    n_future_days = st.slider("Número de días a predecir en el futuro:", 1, 30, 7, key="future_days_slider")

                    future_predictions_scaled = []
                    current_lags = last_known_scaled_lags.copy() # Empezar con los últimos lags reales conocidos

                    for _ in range(n_future_days):
                        # Preparar la entrada para el modelo (debe tener 'window_size' características)
                        input_features = current_lags.reshape(1, -1)
                        
                        # Predecir el siguiente paso (escalado)
                        next_pred_scaled = model.predict(input_features)[0]
                        future_predictions_scaled.append(next_pred_scaled)
                        
                        # Actualizar los lags: eliminar el más antiguo y añadir la nueva predicción
                        current_lags = np.roll(current_lags, -1)
                        current_lags[-1] = next_pred_scaled
                    
                    # Invertir la escala de las predicciones
                    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()
                    
                    # Crear fechas futuras
                    last_date = st.session_state.last_known_date
                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_future_days + 1)]
                    
                    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

                    # Gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Precio Histórico BTC-USD'))
                    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Price'], mode='lines+markers', name='Predicción XGBoost', line=dict(dash='dash')))
                    
                    fig.update_layout(
                        title='Historial y Predicción de Precio BTC-USD con XGBoost',
                        xaxis_title='Fecha',
                        yaxis_title='Precio (USD)',
                        legend_title='Leyenda'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Mostrar Porcentaje de Cambio Predicho ---
                    st.subheader("📊 Cambio Porcentual Estimado (Futuro con XGBoost)")
                    if not prediction_df.empty:
                        last_actual_price = st.session_state.last_known_actual_price
                        predicted_final_price = prediction_df['Predicted Price'].iloc[-1]
                        
                        percentage_change = ((predicted_final_price - last_actual_price) / last_actual_price) * 100
                        
                        color = "green" if percentage_change > 0 else "red" if percentage_change < 0 else "grey"
                        direction = "SUBIDA" if percentage_change > 0 else "BAJADA" if percentage_change < 0 else "CAMBIO MÍNIMO"

                        st.markdown(f"Con base en el modelo XGBoost, se estima una **{direction}** del **<span style='color:{color};'>{percentage_change:.2f}%</span>** en los próximos {n_future_days} días (comparado con el último precio real).", unsafe_allow_html=True)
                    else:
                        st.info("No hay datos de predicción para mostrar el cambio porcentual.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo o el modelo: {e}")
        st.exception(e) # Muestra el traceback completo para depuración

else:
    st.info("Aún no has subido ningún archivo CSV.")

# --- Pie de Página (Opcional) ---
st.markdown("---")
st.markdown("Aplicación demostrativa creada con Streamlit y XGBoost. Las predicciones son ilustrativas.")
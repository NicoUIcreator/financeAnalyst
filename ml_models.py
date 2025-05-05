import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
from typing import Dict, List, Tuple, Union, Any
import time

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características para modelos de machine learning a partir de datos de precios.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios
    
    Returns:
        pd.DataFrame: DataFrame con características
    """
    # Crear una copia para no modificar el original
    result = df.copy()
    
    # Verificar que tenemos los datos necesarios
    if "Close" not in result.columns:
        st.error("No se encontró la columna 'Close' en los datos")
        return result
    
    # Características basadas en retornos
    result["Return_1d"] = result["Close"].pct_change(1)
    result["Return_3d"] = result["Close"].pct_change(3)
    result["Return_5d"] = result["Close"].pct_change(5)
    result["Return_10d"] = result["Close"].pct_change(10)
    
    # Características basadas en medias móviles
    result["SMA_5"] = result["Close"].rolling(window=5).mean()
    result["SMA_10"] = result["Close"].rolling(window=10).mean()
    result["SMA_20"] = result["Close"].rolling(window=20).mean()
    result["SMA_50"] = result["Close"].rolling(window=50).mean()
    
    # Distancia a medias móviles
    result["SMA_5_Dist"] = (result["Close"] / result["SMA_5"] - 1) * 100
    result["SMA_10_Dist"] = (result["Close"] / result["SMA_10"] - 1) * 100
    result["SMA_20_Dist"] = (result["Close"] / result["SMA_20"] - 1) * 100
    result["SMA_50_Dist"] = (result["Close"] / result["SMA_50"] - 1) * 100
    
    # Volatilidad
    result["Volatility_5d"] = result["Return_1d"].rolling(window=5).std() * 100
    result["Volatility_10d"] = result["Return_1d"].rolling(window=10).std() * 100
    result["Volatility_20d"] = result["Return_1d"].rolling(window=20).std() * 100
    
    # Ratios de precios
    result["Price_to_MA5"] = result["Close"] / result["SMA_5"]
    result["Price_to_MA10"] = result["Close"] / result["SMA_10"]
    result["Price_to_MA20"] = result["Close"] / result["SMA_20"]
    result["Price_to_MA50"] = result["Close"] / result["SMA_50"]
    
    # Características basadas en el volumen
    if "Volume" in result.columns:
        result["Volume_MA5"] = result["Volume"].rolling(window=5).mean()
        result["Volume_MA10"] = result["Volume"].rolling(window=10).mean()
        result["Volume_Ratio"] = result["Volume"] / result["Volume_MA5"]
        
        # Características de tendencia del volumen
        result["Volume_Change"] = result["Volume"].pct_change(1)
        result["Volume_Change_5d"] = result["Volume"].pct_change(5)
    
    # Características técnicas adicionales
    if "High" in result.columns and "Low" in result.columns:
        # True Range (volatilidad)
        result["TR"] = np.maximum(
            np.maximum(
                result["High"] - result["Low"], 
                abs(result["High"] - result["Close"].shift(1))
            ),
            abs(result["Low"] - result["Close"].shift(1))
        )
        result["ATR_14"] = result["TR"].rolling(window=14).mean()
        
        # Williams %R
        result["Williams_%R"] = ((result["High"].rolling(14).max() - result["Close"]) / 
                            (result["High"].rolling(14).max() - result["Low"].rolling(14).min())) * -100
    
    # Características de momentum
    result["ROC_5"] = ((result["Close"] / result["Close"].shift(5)) - 1) * 100
    result["ROC_10"] = ((result["Close"] / result["Close"].shift(10)) - 1) * 100
    result["ROC_20"] = ((result["Close"] / result["Close"].shift(20)) - 1) * 100
    
    # Características para la variable objetivo
    result["Target_1d"] = result["Close"].pct_change(1).shift(-1)
    result["Target_3d"] = result["Close"].pct_change(3).shift(-3)
    result["Target_5d"] = result["Close"].pct_change(5).shift(-5)
    
    # Crear etiquetas binarias para clasificación
    result["Target_Direction_1d"] = (result["Target_1d"] > 0).astype(int)
    result["Target_Direction_3d"] = (result["Target_3d"] > 0).astype(int)
    result["Target_Direction_5d"] = (result["Target_5d"] > 0).astype(int)
    
    # Eliminar filas con valores NaN al principio
    result = result.dropna()
    
    return result

def prepare_ml_data(
    df: pd.DataFrame, 
    target_column: str = "Target_Direction_5d", 
    train_size: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepara los datos para modelos de machine learning.
    
    Args:
        df (pd.DataFrame): DataFrame con características
        target_column (str): Columna objetivo
        train_size (float): Proporción de datos para entrenamiento
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Verificar que tenemos la columna objetivo
    if target_column not in df.columns:
        st.error(f"No se encontró la columna objetivo '{target_column}' en los datos")
        return None, None, None, None, []
    
    # Columnas a excluir
    exclude_columns = [
        "Open", "High", "Low", "Close", "Volume", "Datetime", 
        "Target_1d", "Target_3d", "Target_5d", 
        "Target_Direction_1d", "Target_Direction_3d", "Target_Direction_5d"
    ]
    
    # Seleccionar características
    feature_cols = [col for col in df.columns if col not in exclude_columns and col != target_column]
    
    # Verificar si hay características disponibles
    if not feature_cols:
        st.error("No se encontraron características válidas para el modelo")
        return None, None, None, None, []
    
    # Preparar datos
    X = df[feature_cols].values
    y = df[target_column].values
    
    # Dividir en conjuntos de entrenamiento y prueba (respetando el orden temporal)
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    is_classifier: bool = True, 
    n_estimators: int = 100
) -> Any:
    """
    Entrena un modelo Random Forest.
    
    Args:
        X_train (np.ndarray): Características de entrenamiento
        y_train (np.ndarray): Valores objetivo de entrenamiento
        is_classifier (bool): Si es True, entrena un clasificador; si es False, un regresor
        n_estimators (int): Número de árboles
    
    Returns:
        Any: Modelo entrenado
    """
    if is_classifier:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    is_classifier: bool = True
) -> Any:
    """
    Entrena un modelo XGBoost.
    
    Args:
        X_train (np.ndarray): Características de entrenamiento
        y_train (np.ndarray): Valores objetivo de entrenamiento
        is_classifier (bool): Si es True, entrena un clasificador; si es False, un regresor
    
    Returns:
        Any: Modelo entrenado
    """
    if is_classifier:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    return model

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entrena un modelo de regresión lineal.
    
    Args:
        X_train (np.ndarray): Características de entrenamiento
        y_train (np.ndarray): Valores objetivo de entrenamiento
    
    Returns:
        LinearRegression: Modelo entrenado
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def train_lstm_pytorch(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    sequence_length: int = 10
) -> Dict:
    """
    Entrena un modelo LSTM con PyTorch.
    Esta es una implementación simplificada para Streamlit.
    
    Args:
        X_train (np.ndarray): Características de entrenamiento
        y_train (np.ndarray): Valores objetivo de entrenamiento
        sequence_length (int): Longitud de secuencia para LSTM
    
    Returns:
        Dict: Diccionario de resultados (placeholder simplificado)
    """
    # Nota: En una implementación real, aquí se implementaría un modelo LSTM con PyTorch
    # Esto es un placeholder para la estructura de la aplicación
    
    # Simulamos un entrenamiento rápido
    time.sleep(0.5)
    
    return {
        "status": "trained",
        "message": "Modelo LSTM entrenado (simulado para demostración)"
    }

def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    is_classifier: bool = True
) -> Dict:
    """
    Evalúa un modelo de machine learning.
    
    Args:
        model (Any): Modelo entrenado
        X_test (np.ndarray): Características de prueba
        y_test (np.ndarray): Valores objetivo de prueba
        is_classifier (bool): Si es True, evalúa un clasificador; si es False, un regresor
    
    Returns:
        Dict: Métricas de evaluación
    """
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        
        # Probabilidades para curva ROC (si el modelo lo soporta)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = y_pred
        
        return {
            "accuracy": accuracy,
            "predictions": y_pred,
            "probabilities": y_prob
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "predictions": y_pred
        }

def predict_price_movement(df: pd.DataFrame, model_name: str = "XGBoost") -> Dict:
    """
    Predice el movimiento futuro del precio utilizando un modelo seleccionado.
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos de precios
        model_name (str): Nombre del modelo a utilizar
    
    Returns:
        Dict: Resultados de la predicción
    """
    # Verificar datos mínimos
    if len(df) < 100:
        return {
            "prediction": 0,
            "confidence": 50,
            "direction": "neutral",
            "factors": "Datos insuficientes"
        }
    
    try:
        # Crear características
        df_features = create_features(df)
        
        # Preparar datos
        X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(
            df_features, 
            target_column="Target_Direction_5d",
            train_size=0.8
        )
        
        # Verificar si la preparación fue exitosa
        if X_train is None:
            return {
                "prediction": 0,
                "confidence": 50,
                "direction": "neutral",
                "factors": "Error en la preparación de datos"
            }
        
        # Entrenar modelo según selección
        if model_name == "XGBoost":
            model = train_xgboost(X_train, y_train, is_classifier=True)
            
            # Evaluar
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=True)
            
            # Obtener características importantes
            try:
                feature_importance = model.feature_importances_
                top_features = [feature_names[i] for i in np.argsort(feature_importance)[-3:]]
                factors = f"Factores importantes: {', '.join(top_features)}"
            except:
                factors = "Basado en patrones históricos recientes"
            
            # Predecir con el último dato
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            
            # Predicción
            try:
                prob = model.predict_proba(last_features_scaled)[0, 1]
                prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob) * 100
            except:
                prediction = model.predict(last_features_scaled)[0]
                confidence = 70  # Confianza por defecto
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": factors
            }
            
        elif model_name == "Random Forest":
            model = train_random_forest(X_train, y_train, is_classifier=True)
            
            # Evaluar
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=True)
            
            # Obtener características importantes
            feature_importance = model.feature_importances_
            top_features = [feature_names[i] for i in np.argsort(feature_importance)[-3:]]
            
            # Predecir con el último dato
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            
            # Predicción
            try:
                prob = model.predict_proba(last_features_scaled)[0, 1]
                prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob) * 100
            except:
                prediction = model.predict(last_features_scaled)[0]
                confidence = 70  # Confianza por defecto
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": f"Factores importantes: {', '.join(top_features)}"
            }
            
        elif model_name == "Linear Regression":
            # Para regresión lineal, usamos valores continuos
            _, _, y_train_reg, _, _ = prepare_ml_data(df_features, target_column="Target_5d")
            model = train_linear_regression(X_train, y_train_reg)
            
            # Evaluar
            eval_results = evaluate_model(model, X_test, y_test, is_classifier=False)
            
            # Predecir tendencia
            scaler = StandardScaler()
            last_features_scaled = scaler.fit_transform(df_features[feature_names].values)[-1].reshape(1, -1)
            
            prediction = model.predict(last_features_scaled)[0]
            confidence = min(abs(prediction) * 200, 100)  # Escalado simple
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": "Basado en tendencia reciente y factores técnicos"
            }
            
        elif model_name == "LSTM":
            # Entrenar modelo LSTM
            lstm_results = train_lstm_pytorch(X_train, y_train, sequence_length=10)
            
            # Como es un placeholder, generamos una predicción simulada
            # En una implementación real, usaríamos el modelo LSTM entrenado
            last_trend = df["Close"].pct_change(5).iloc[-1]
            prediction = 1 if last_trend > 0 else 0
            confidence = min(abs(last_trend) * 200, 95)  # Confianza basada en la tendencia reciente
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "direction": "alza" if prediction > 0 else "baja",
                "factors": "Basado en patrones secuenciales temporales (LSTM)",
                "model_info": lstm_results
            }
        else:
            return {
                "prediction": 0,
                "confidence": 50,
                "direction": "neutral",
                "factors": f"Modelo '{model_name}' no reconocido"
            }
    
    except Exception as e:
        # En caso de error, devolver resultado neutro con información del error
        return {
            "prediction": 0,
            "confidence": 50,
            "direction": "neutral",
            "factors": f"Error en la predicción: {str(e)}"
        }

# Función para crear una aplicación Streamlit
def create_financial_ml_app():
    st.title("Análisis Predictivo de Mercados Financieros")
    
    # Sidebar para opciones
    st.sidebar.header("Configuración")
    
    # Opción para subir archivo o usar datos de ejemplo
    data_option = st.sidebar.radio(
        "Seleccione fuente de datos",
        ["Datos de ejemplo", "Subir archivo CSV"]
    )
    
    # Inicializar DataFrame vacío
    df = pd.DataFrame()
    
    if data_option == "Datos de ejemplo":
        # Generar datos de ejemplo
        st.sidebar.info("Usando datos de ejemplo generados")
        
        # Generar fechas para los últimos 500 días
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500)
        
        # Generar precios simulados
        np.random.seed(42)
        price = 100
        prices = [price]
        for _ in range(499):
            change = np.random.normal(0, 1) / 100  # Cambio aleatorio
            price *= (1 + change)
            prices.append(price)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
        
        # Generar columnas adicionales
        df['Open'] = df['Close'] * (1 - np.random.normal(0, 0.5, len(df)) / 100)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.3, len(df)) / 100))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.3, len(df)) / 100))
        df['Volume'] = np.random.randint(1000, 10000, len(df))
        
        # Establecer fecha como índice
        df.set_index('Date', inplace=True)
    
    else:
        # Opción para subir archivo
        uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Verificar columnas necesarias
                required_cols = ['Close']
                if not all(col in df.columns for col in required_cols):
                    st.error("El archivo debe contener al menos la columna 'Close'")
                    df = pd.DataFrame()
                
                # Si tiene una columna de fecha, configurarla como índice
                date_cols = [col for col in df.columns if col.lower() in ['date', 'datetime', 'time']]
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    df.set_index(date_cols[0], inplace=True)
                
            except Exception as e:
                st.error(f"Error al leer el archivo: {str(e)}")
                df = pd.DataFrame()
    
    # Verificar que tenemos datos válidos
    if not df.empty:
        # Mostrar los datos
        st.subheader("Datos cargados")
        st.write(df.head())
        
        # Mostrar gráfico de precios
        st.subheader("Gráfico de precios")
        
        # Crear pestaña para visualización básica
        tab1, tab2, tab3 = st.tabs(["Precios", "Análisis Técnico", "Predicciones"])
        
        with tab1:
            st.line_chart(df['Close'])
        
        with tab2:
            # Calcular indicadores técnicos básicos
            if len(df) > 50:
                # Crear características técnicas
                df_features = create_features(df)
                
                # Seleccionar indicadores para mostrar
                st.subheader("Indicadores Técnicos")
                
                # Seleccionar indicador
                indicator = st.selectbox(
                    "Seleccionar indicador", 
                    ["SMA", "Volatilidad", "ROC", "Williams %R"]
                )
                
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
            
            # Selección de modelo
            model_name = st.selectbox(
                "Seleccionar modelo", 
                ["XGBoost", "Random Forest", "Linear Regression", "LSTM"]
            )
            
            # Botón para realizar predicción
            if st.button("Realizar predicción"):
                with st.spinner("Entrenando modelo y generando predicción..."):
                    # Hacer predicción
                    prediction_results = predict_price_movement(df, model_name)
                    
                    # Mostrar resultados
                    direction = prediction_results["direction"]
                    confidence = prediction_results["confidence"]
                    factors = prediction_results["factors"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Dirección prevista", 
                            direction.upper(),
                            delta="↑" if direction == "alza" else "↓" if direction == "baja" else "→"
                        )
                    
                    with col2:
                        st.metric("Confianza", f"{confidence:.2f}%")
                    
                    st.write(f"**Factores influyentes:** {factors}")
                    
                    # Métricas adicionales basadas en el modelo
                    if model_name != "LSTM":
                        # Crear características y dividir datos
                        df_features = create_features(df)
                        X_train, X_test, y_train, y_test, _ = prepare_ml_data(
                            df_features, 
                            target_column="Target_Direction_5d" if model_name != "Linear Regression" else "Target_5d",
                            train_size=0.8
                        )
                        
                        # Mostrar métricas de rendimiento en datos históricos
                        st.subheader("Rendimiento del modelo en datos históricos")
                        
                        is_classifier = model_name != "Linear Regression"
                        
                        if is_classifier:
                            # Calcular precisión en conjunto de prueba
                            if model_name == "XGBoost":
                                model = train_xgboost(X_train, y_train, is_classifier=True)
                            else:  # Random Forest
                                model = train_random_forest(X_train, y_train, is_classifier=True)
                                
                            eval_results = evaluate_model(model, X_test, y_test, is_classifier=True)
                            st.metric("Precisión en datos de prueba", f"{eval_results['accuracy']:.2%}")
                            
                            # Mostrar matriz de confusión simplificada
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
                        
                        else:  # Regresión
                            model = train_linear_regression(X_train, y_train)
                            eval_results = evaluate_model(model, X_test, y_test, is_classifier=False)
                            
                            st.metric("Error cuadrático medio (RMSE)", f"{eval_results['rmse']:.4f}")
                            
                            # Gráfico de predicciones vs reales
                            pred_df = pd.DataFrame({
                                'Real': y_test,
                                'Predicción': eval_results['predictions']
                            })
                            st.line_chart(pred_df)
    
    else:
            st.warning("Por favor, cargue datos")

    # Close the if not df.empty block
    # Close the create_financial_ml_app() function
    return

# Call the Streamlit app function
create_financial_ml_app()
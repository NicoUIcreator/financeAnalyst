import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Importar módulos personalizados
from data_loader import load_stock_data, get_available_tickers
from technical_analysis import calculate_indicators, generate_signals
from ml_models import predict_price_movement
from dashboard import create_price_chart, create_indicator_charts, create_signal_dashboard
from utils import format_number, calculate_returns

# Configurar la página
st.set_page_config(
    page_title="Trading Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 class='main-header'>Trading Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Lista de tickers disponibles para seleccionar
    available_tickers = get_available_tickers()
    ticker = st.sidebar.selectbox("Seleccionar Activo", available_tickers)
    
    # Periodo de tiempo
    time_periods = {
        "1 Semana": 7,
        "1 Mes": 30,
        "3 Meses": 90,
        "6 Meses": 180,
        "1 Año": 365,
        "2 Años": 730,
        "5 Años": 1825
    }
    selected_period = st.sidebar.selectbox("Periodo de Tiempo", list(time_periods.keys()))
    
    # Intervalo de tiempo
    intervals = ["1d", "1h", "15m", "5m", "1m"]
    interval = st.sidebar.selectbox("Intervalo", intervals)
    
    # Indicadores técnicos a mostrar
    indicators = ["MACD", "RSI", "Bollinger Bands", "Stochastic", "Moving Averages"]
    selected_indicators = st.sidebar.multiselect(
        "Indicadores Técnicos", 
        indicators, 
        default=["MACD", "RSI"]
    )
    
    # Modelos de ML para predicción
    ml_models = ["XGBoost", "LSTM", "Random Forest", "Linear Regression"]
    selected_model = st.sidebar.selectbox("Modelo de Predicción", ml_models)
    
    # Cargar datos basados en las selecciones
    days = time_periods[selected_period]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    with st.spinner(f"Cargando datos para {ticker}..."):
        try:
            df = load_stock_data(ticker, start_date, end_date, interval)
            
            # Calcular indicadores técnicos
            df = calculate_indicators(df, selected_indicators)
            
            # Generar señales de trading
            df, signals = generate_signals(df, selected_indicators)
            
            # Predicción del modelo ML
            prediction_results = predict_price_movement(df, selected_model)
            
            # Renderizar el dashboard
            st.markdown("<h2 class='sub-header'>Análisis de Mercado</h2>", unsafe_allow_html=True)
            
            # Datos básicos del activo
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            col1.metric(
                "Precio Actual", 
                f"${format_number(current_price)}", 
                f"{format_number(price_change_pct)}%"
            )
            
            col2.metric(
                "Volumen (24h)", 
                format_number(df['Volume'].iloc[-1], precision=0)
            )
            
            # Calcular retornos
            daily_return = calculate_returns(df['Close'], period=1).iloc[-1] * 100
            monthly_return = calculate_returns(df['Close'], period=30).iloc[-1] * 100
            
            col3.metric(
                "Retorno Diario", 
                f"{format_number(daily_return)}%"
            )
            
            col4.metric(
                "Retorno Mensual", 
                f"{format_number(monthly_return)}%"
            )
            
            # Gráfico de precios
            st.markdown("<h3>Precio y Volumen</h3>", unsafe_allow_html=True)
            price_chart = create_price_chart(df, ticker)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Gráficos de indicadores
            if selected_indicators:
                st.markdown("<h3>Indicadores Técnicos</h3>", unsafe_allow_html=True)
                indicator_charts = create_indicator_charts(df, selected_indicators)
                for chart in indicator_charts:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Dashboard de señales
            st.markdown("<h3>Señales de Trading</h3>", unsafe_allow_html=True)
            signal_dashboard = create_signal_dashboard(signals, prediction_results)
            st.markdown(signal_dashboard, unsafe_allow_html=True)
            
            # Resultados de predicción
            st.markdown("<h3>Predicción de Precios</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            # Mostrar predicción
            prediction_direction = "alza" if prediction_results['prediction'] > 0 else "baja"
            prediction_confidence = prediction_results['confidence']
            
            col1.markdown(f"""
            <div class='card'>
                <h4>Predicción ({selected_model})</h4>
                <p>Tendencia: <span class='{"positive" if prediction_direction == "alza" else "negative"}'>{prediction_direction.upper()}</span></p>
                <p>Confianza: {prediction_confidence:.2f}%</p>
                <p>Horizonte: 5 días</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar factores relevantes
            col2.markdown(f"""
            <div class='card'>
                <h4>Factores Relevantes</h4>
                <ul>
                    <li>RSI {'sobrecomprado' if 'RSI' in df.columns and df['RSI'].iloc[-1] > 70 else 'sobrevendido' if 'RSI' in df.columns and df['RSI'].iloc[-1] < 30 else 'neutral'}</li>
                    <li>MACD {'positivo' if 'MACD' in df.columns and df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'negativo'}</li>
                    <li>Tendencia: {prediction_results['factors']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            st.error("Por favor verifica tu conexión a internet o elige otro activo.")

if __name__ == "__main__":
    main()
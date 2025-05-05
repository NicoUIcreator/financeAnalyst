import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import time
from typing import List, Dict, Union, Tuple

# Cache para almacenar datos y reducir llamadas a la API
DATA_CACHE = {}
CACHE_EXPIRY = 300  # 5 minutos en segundos

def get_available_tickers() -> List[str]:
    """
    Devuelve una lista de tickers populares para que el usuario seleccione.
    
    Returns:
        List[str]: Lista de tickers disponibles.
    """
    popular_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
        "WMT", "PG", "XOM", "BAC", "DIS", "NFLX", "INTC", "VZ", "KO", "PEP",
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225"
    ]
    return popular_tickers

@st.cache_data(ttl=CACHE_EXPIRY)
def load_stock_data(
    ticker: str, 
    start_date: datetime, 
    end_date: datetime, 
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Carga datos históricos de un activo desde Yahoo Finance.
    
    Args:
        ticker (str): Símbolo del activo.
        start_date (datetime): Fecha de inicio.
        end_date (datetime): Fecha de fin.
        interval (str): Intervalo de tiempo (1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo).
    
    Returns:
        pd.DataFrame: DataFrame con los datos históricos.
    """
    try:
        # Verificar si necesitamos datos intradía
        if interval in ["1m", "5m", "15m", "30m", "60m", "1h"]:
            # Para datos intradía, limitamos a los últimos 60 días (límite de Yahoo Finance)
            if (end_date - start_date).days > 60:
                start_date = end_date - timedelta(days=60)
                st.warning(f"Yahoo Finance limita los datos intradía a 60 días. Ajustando la fecha de inicio.")
                
            # Los datos de 1m solo están disponibles para los últimos 7 días
            if interval == "1m" and (end_date - start_date).days > 7:
                start_date = end_date - timedelta(days=7)
                st.warning(f"Los datos de 1 minuto están limitados a los últimos 7 días. Ajustando la fecha de inicio.")
        
        # Obtener datos
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        # Si no hay datos, mostrar mensaje de error
        if data.empty:
            st.error(f"No se encontraron datos para {ticker} en el periodo seleccionado.")
            return pd.DataFrame()
        
        # Procesar los datos
        data = data.reset_index()
        
        # Renombrar la columna 'Date' o 'Datetime'
        if "Date" in data.columns:
            data = data.rename(columns={"Date": "Datetime"})
        elif "Datetime" not in data.columns:
            data["Datetime"] = data.index
            
        # Asegurarse de que todos los valores son numéricos
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        
        # Calcular campos adicionales
        data["HL2"] = (data["High"] + data["Low"]) / 2
        data["HLC3"] = (data["High"] + data["Low"] + data["Close"]) / 3
        data["OHLC4"] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4

        # Calcular retornos
        data["Return"] = data["Close"].pct_change() * 100
        
        # Eliminar filas con NaN
        data = data.dropna(subset=["Close"])
        
        return data
        
    except Exception as e:
        st.error(f"Error al cargar datos de {ticker}: {str(e)}")
        return pd.DataFrame()

def get_ticker_info(ticker: str) -> Dict:
    """
    Obtiene información general sobre un activo.
    
    Args:
        ticker (str): Símbolo del activo.
    
    Returns:
        Dict: Diccionario con información del activo.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extraer información relevante
        ticker_info = {
            "name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "eps": info.get("trailingEps", None),
            "dividend_yield": info.get("dividendYield", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "description": info.get("longBusinessSummary", "")
        }
        
        return ticker_info
        
    except Exception as e:
        st.error(f"Error al obtener información de {ticker}: {str(e)}")
        return {}

def get_market_news(tickers: List[str] = None) -> List[Dict]:
    """
    Obtiene noticias recientes del mercado o relacionadas con tickers específicos.
    
    Args:
        tickers (List[str], optional): Lista de tickers para filtrar noticias.
    
    Returns:
        List[Dict]: Lista de noticias con título, fuente, enlace y fecha.
    """
    # Esta función es un placeholder - Yahoo Finance no proporciona fácilmente
    # acceso a noticias a través de su API. Podrías implementarla usando otro servicio.
    return []

def get_realtime_quote(ticker: str) -> Dict:
    """
    Obtiene cotización en tiempo real para un activo.
    
    Args:
        ticker (str): Símbolo del activo.
    
    Returns:
        Dict: Cotización en tiempo real.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Obtener último precio
        latest_data = stock.history(period="1d")
        
        if latest_data.empty:
            return {}
        
        # Crear diccionario con información
        quote = {
            "price": latest_data["Close"].iloc[-1],
            "change": latest_data["Close"].iloc[-1] - latest_data["Open"].iloc[-1],
            "change_percent": ((latest_data["Close"].iloc[-1] / latest_data["Open"].iloc[-1]) - 1) * 100,
            "volume": latest_data["Volume"].iloc[-1],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return quote
        
    except Exception as e:
        st.error(f"Error al obtener cotización de {ticker}: {str(e)}")
        return {}

def load_multiple_stocks(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime, 
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Carga datos para múltiples activos.
    
    Args:
        tickers (List[str]): Lista de símbolos de activos.
        start_date (datetime): Fecha de inicio.
        end_date (datetime): Fecha de fin.
        interval (str): Intervalo de tiempo.
    
    Returns:
        Dict[str, pd.DataFrame]: Diccionario de DataFrames por activo.
    """
    result = {}
    
    for ticker in tickers:
        result[ticker] = load_stock_data(ticker, start_date, end_date, interval)
        
    return result
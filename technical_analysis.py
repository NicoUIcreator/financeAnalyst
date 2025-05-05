import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Tuple, Union
import streamlit as st

def calculate_indicators(df: pd.DataFrame, selected_indicators: List[str]) -> pd.DataFrame:
    """
    Calcula indicadores técnicos seleccionados para un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de precios
        selected_indicators (List[str]): Lista de indicadores a calcular
    
    Returns:
        pd.DataFrame: DataFrame con indicadores calculados
    """
    # Crear una copia para no modificar el original
    result = df.copy()
    
    # Verificar que tenemos los datos necesarios
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col not in result.columns:
            st.warning(f"Columna {col} no encontrada. Algunos indicadores pueden no calcularse correctamente.")
    
    # Calcular indicadores según la selección
    for indicator in selected_indicators:
        if indicator == "MACD":
            # MACD (Moving Average Convergence Divergence)
            macd = ta.macd(result["Close"])
            result = pd.concat([result, macd], axis=1)
            
        elif indicator == "RSI":
            # RSI (Relative Strength Index)
            result["RSI"] = ta.rsi(result["Close"], length=14)
            
        elif indicator == "Bollinger Bands":
            # Bandas de Bollinger
            bbands = ta.bbands(result["Close"], length=20)
            result = pd.concat([result, bbands], axis=1)
            
        elif indicator == "Stochastic":
            # Oscilador Estocástico
            stoch = ta.stoch(result["High"], result["Low"], result["Close"])
            result = pd.concat([result, stoch], axis=1)
            
        elif indicator == "Moving Averages":
            # Medias Móviles
            result["SMA_20"] = ta.sma(result["Close"], length=20)
            result["SMA_50"] = ta.sma(result["Close"], length=50)
            result["SMA_200"] = ta.sma(result["Close"], length=200)
            result["EMA_12"] = ta.ema(result["Close"], length=12)
            result["EMA_26"] = ta.ema(result["Close"], length=26)
    
    return result

def generate_signals(df: pd.DataFrame, selected_indicators: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Genera señales de trading basadas en indicadores.
    
    Args:
        df (pd.DataFrame): DataFrame con indicadores calculados
        selected_indicators (List[str]): Lista de indicadores utilizados
    
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con señales y diccionario de resumen
    """
    # Crear una copia para añadir señales
    result = df.copy()
    signals = {
        "buy": [],
        "sell": [],
        "summary": {},
        "strength": 0  # De -100 a 100, donde -100 es fuerte venta y 100, fuerte compra
    }
    
    # Inicializar columnas de señales
    result["Signal_Buy"] = 0
    result["Signal_Sell"] = 0
    result["Signal_Strength"] = 0
    
    # Verificar indicadores disponibles y generar señales
    signal_count = 0
    signal_value = 0
    
    # 1. Señales de MACD
    if "MACD" in selected_indicators and "MACD" in result.columns and "MACD_signal" in result.columns:
        signal_count += 1
        
        # Cruce MACD por encima de la línea de señal (compra)
        result.loc[(result["MACD"] > result["MACD_signal"]) & 
                  (result["MACD"].shift(1) <= result["MACD_signal"].shift(1)), "Signal_Buy"] += 1
        
        # Cruce MACD por debajo de la línea de señal (venta)
        result.loc[(result["MACD"] < result["MACD_signal"]) & 
                  (result["MACD"].shift(1) >= result["MACD_signal"].shift(1)), "Signal_Sell"] += 1
        
        # Señal actual para el resumen
        if result["MACD"].iloc[-1] > result["MACD_signal"].iloc[-1]:
            signals["summary"]["MACD"] = "Compra"
            signal_value += 1
        else:
            signals["summary"]["MACD"] = "Venta"
            signal_value -= 1
    
    # 2. Señales de RSI
    if "RSI" in selected_indicators and "RSI" in result.columns:
        signal_count += 1
        
        # RSI sobrevendido (compra)
        result.loc[result["RSI"] < 30, "Signal_Buy"] += 1
        
        # RSI sobrecomprado (venta)
        result.loc[result["RSI"] > 70, "Signal_Sell"] += 1
        
        # Señal actual para el resumen
        rsi_value = result["RSI"].iloc[-1]
        if rsi_value < 30:
            signals["summary"]["RSI"] = "Fuerte Compra (Sobrevendido)"
            signal_value += 2
        elif rsi_value < 45:
            signals["summary"]["RSI"] = "Compra"
            signal_value += 1
        elif rsi_value > 70:
            signals["summary"]["RSI"] = "Fuerte Venta (Sobrecomprado)"
            signal_value -= 2
        elif rsi_value > 55:
            signals["summary"]["RSI"] = "Venta"
            signal_value -= 1
        else:
            signals["summary"]["RSI"] = "Neutral"
    
    # 3. Señales de Bollinger Bands
    if "Bollinger Bands" in selected_indicators and "BBL_20_2.0" in result.columns and "BBU_20_2.0" in result.columns:
        signal_count += 1
        
        # Precio toca la banda inferior (compra)
        result.loc[result["Close"] <= result["BBL_20_2.0"], "Signal_Buy"] += 1
        
        # Precio toca la banda superior (venta)
        result.loc[result["Close"] >= result["BBU_20_2.0"], "Signal_Sell"] += 1
        
        # Señal actual para el resumen
        if result["Close"].iloc[-1] <= result["BBL_20_2.0"].iloc[-1]:
            signals["summary"]["Bollinger"] = "Compra (Banda Inferior)"
            signal_value += 1
        elif result["Close"].iloc[-1] >= result["BBU_20_2.0"].iloc[-1]:
            signals["summary"]["Bollinger"] = "Venta (Banda Superior)"
            signal_value -= 1
        else:
            signals["summary"]["Bollinger"] = "Neutral (Dentro de Bandas)"
    
    # 4. Señales de Stochastic
    if "Stochastic" in selected_indicators and "STOCHk_14_3_3" in result.columns and "STOCHd_14_3_3" in result.columns:
        signal_count += 1
        
        # Cruce hacia arriba en zona sobrevendida (compra)
        result.loc[(result["STOCHk_14_3_3"] > result["STOCHd_14_3_3"]) & 
                  (result["STOCHk_14_3_3"].shift(1) <= result["STOCHd_14_3_3"].shift(1)) & 
                  (result["STOCHk_14_3_3"] < 20), "Signal_Buy"] += 1
        
        # Cruce hacia abajo en zona sobrecomprada (venta)
        result.loc[(result["STOCHk_14_3_3"] < result["STOCHd_14_3_3"]) & 
                  (result["STOCHk_14_3_3"].shift(1) >= result["STOCHd_14_3_3"].shift(1)) & 
                  (result["STOCHk_14_3_3"] > 80), "Signal_Sell"] += 1
        
        # Señal actual para el resumen
        stoch_k = result["STOCHk_14_3_3"].iloc[-1]
        stoch_d = result["STOCHd_14_3_3"].iloc[-1]
        
        if stoch_k < 20 and stoch_k > stoch_d:
            signals["summary"]["Stochastic"] = "Fuerte Compra (Sobrevendido)"
            signal_value += 2
        elif stoch_k > 80 and stoch_k < stoch_d:
            signals["summary"]["Stochastic"] = "Fuerte Venta (Sobrecomprado)"
            signal_value -= 2
        elif stoch_k > stoch_d:
            signals["summary"]["Stochastic"] = "Compra"
            signal_value += 1
        else:
            signals["summary"]["Stochastic"] = "Venta"
            signal_value -= 1
    
    # 5. Señales de Moving Averages
    if "Moving Averages" in selected_indicators:
        has_ma = False
        
        # Verificar si tenemos las medias móviles calculadas
        if "SMA_20" in result.columns and "SMA_50" in result.columns:
            signal_count += 1
            has_ma = True
            
            # Golden Cross (SMA 20 cruza por encima de SMA 50)
            result.loc[(result["SMA_20"] > result["SMA_50"]) & 
                      (result["SMA_20"].shift(1) <= result["SMA_50"].shift(1)), "Signal_Buy"] += 1
            
            # Death Cross (SMA 20 cruza por debajo de SMA 50)
            result.loc[(result["SMA_20"] < result["SMA_50"]) & 
                      (result["SMA_20"].shift(1) >= result["SMA_50"].shift(1)), "Signal_Sell"] += 1
            
            # Precio por encima/debajo de las medias
            if result["Close"].iloc[-1] > result["SMA_20"].iloc[-1]:
                result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5
            else:
                result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
                
            if result["Close"].iloc[-1] > result["SMA_50"].iloc[-1]:
                result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5
            else:
                result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
        
        # Señal actual para el resumen
        if has_ma:
            ma_signal = ""
            ma_value = 0
            
            # SMA 20 vs SMA 50
            if result["SMA_20"].iloc[-1] > result["SMA_50"].iloc[-1]:
                ma_signal += "SMA 20 por encima de SMA 50 (Alcista). "
                ma_value += 1
            else:
                ma_signal += "SMA 20 por debajo de SMA 50 (Bajista). "
                ma_value -= 1
            
            # Precio vs SMA 200
            if "SMA_200" in result.columns:
                if result["Close"].iloc[-1] > result["SMA_200"].iloc[-1]:
                    ma_signal += "Precio por encima de SMA 200 (Alcista a largo plazo)."
                    ma_value += 1
                else:
                    ma_signal += "Precio por debajo de SMA 200 (Bajista a largo plazo)."
                    ma_value -= 1
            
            signals["summary"]["Moving Averages"] = ma_signal
            signal_value += ma_value
    
    # Calcular la fuerza de la señal global
    if signal_count > 0:
        # Normalizar entre -100 y 100
        signals["strength"] = int((signal_value / (signal_count * 2)) * 100)
    
    # Identificar puntos de señal para el gráfico
    for i in range(len(result)):
        # Si hay suficientes señales de compra
        if result["Signal_Buy"].iloc[i] >= 2:
            signals["buy"].append({
                "index": i,
                "datetime": result["Datetime"].iloc[i],
                "price": result["Close"].iloc[i],
                "strength": result["Signal_Buy"].iloc[i]
            })
        
        # Si hay suficientes señales de venta
        if result["Signal_Sell"].iloc[i] >= 2:
            signals["sell"].append({
                "index": i,
                "datetime": result["Datetime"].iloc[i],
                "price": result["Close"].iloc[i],
                "strength": result["Signal_Sell"].iloc[i]
            })
    
    # Calcular la fuerza de la señal en cada punto
    result["Signal_Strength"] = result["Signal_Buy"] - result["Signal_Sell"]
    
    return result, signals

def backtesting_simple(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """
    Realiza un backtest simple basado en las señales generadas.
    
    Args:
        df (pd.DataFrame): DataFrame con señales calculadas
        initial_capital (float): Capital inicial para el backtest
    
    Returns:
        Dict: Resultados del backtest
    """
    # Verificar si tenemos las columnas necesarias
    if "Signal_Buy" not in df.columns or "Signal_Sell" not in df.columns:
        return {
            "error": "No se encontraron señales para el backtest",
            "final_capital": initial_capital,
            "return_pct": 0.0,
            "trades": 0
        }
    
    # Crear una copia del DataFrame
    backtest = df.copy()
    
    # Inicializar variables
    capital = initial_capital
    position = 0  # 0: sin posición, 1: comprado
    entry_price = 0.0
    trades = []
    
    # Recorrer los datos
    for i in range(1, len(backtest)):
        # Señal de compra (cuando no tenemos posición)
        if position == 0 and backtest["Signal_Buy"].iloc[i] >= 2:
            # Comprar
            entry_price = backtest["Close"].iloc[i]
            position = 1
            shares = capital / entry_price
            
            trades.append({
                "type": "buy",
                "datetime": backtest["Datetime"].iloc[i],
                "price": entry_price,
                "capital": capital,
                "shares": shares
            })
        
        # Señal de venta (cuando tenemos posición)
        elif position == 1 and backtest["Signal_Sell"].iloc[i] >= 2:
            # Vender
            exit_price = backtest["Close"].iloc[i]
            shares = capital / entry_price
            capital = shares * exit_price
            position = 0
            
            trades.append({
                "type": "sell",
                "datetime": backtest["Datetime"].iloc[i],
                "price": exit_price,
                "capital": capital,
                "pnl": ((exit_price / entry_price) - 1) * 100
            })
    
    # Si terminamos con una posición abierta, cerrarla al último precio
    if position == 1:
        exit_price = backtest["Close"].iloc[-1]
        shares = capital / entry_price
        capital = shares * exit_price
        
        trades.append({
            "type": "sell (final)",
            "datetime": backtest["Datetime"].iloc[-1],
            "price": exit_price,
            "capital": capital,
            "pnl": ((exit_price / entry_price) - 1) * 100
        })
    
    # Calcular métricas
    return_pct = ((capital / initial_capital) - 1) * 100
    n_trades = len([t for t in trades if t["type"] == "buy"])
    
    # Calcular PnL de cada operación
    pnl_list = [t["pnl"] for t in trades if "pnl" in t]
    avg_pnl = np.mean(pnl_list) if pnl_list else 0
    win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) if pnl_list else 0
    
    return {
        "final_capital": capital,
        "return_pct": return_pct,
        "trades": n_trades,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "trade_history": trades
    }

def calculate_support_resistance(df: pd.DataFrame, n_levels: int = 3, window: int = 20) -> Dict[str, List[float]]:
    """
    Calcula niveles de soporte y resistencia basados en máximos y mínimos locales.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de precios
        n_levels (int): Número de niveles a identificar
        window (int): Tamaño de la ventana para buscar máximos y mínimos locales
    
    Returns:
        Dict[str, List[float]]: Diccionario con niveles de soporte y resistencia
    """
    result = {
        "support": [],
        "resistance": []
    }
    
    # Verificar que tenemos datos suficientes
    if len(df) < window * 2:
        return result
    
    # Encontrar mínimos locales (soportes)
    min_idx = df["Low"].rolling(window=window, center=True).apply(lambda x: x.argmin() == len(x)//2)
    support_levels = df.loc[min_idx, "Low"].dropna().tolist()
    
    # Encontrar máximos locales (resistencias)
    max_idx = df["High"].rolling(window=window, center=True).apply(lambda x: x.argmax() == len(x)//2)
    resistance_levels = df.loc[max_idx, "High"].dropna().tolist()
    
    # Agrupar niveles cercanos
    support_levels = cluster_levels(support_levels)
    resistance_levels = cluster_levels(resistance_levels)
    
    # Ordenar por frecuencia y tomar los n_levels más frecuentes
    result["support"] = sorted(support_levels, key=lambda x: -support_levels.count(x))[:n_levels]
    result["resistance"] = sorted(resistance_levels, key=lambda x: -resistance_levels.count(x))[:n_levels]
    
    return result

def cluster_levels(levels: List[float], threshold: float = 0.01) -> List[float]:
    """
    Agrupa niveles cercanos utilizando un umbral relativo.
    
    Args:
        levels (List[float]): Lista de niveles
        threshold (float): Umbral para considerar niveles como cercanos (%)
    
    Returns:
        List[float]: Lista de niveles agrupados
    """
    if not levels:
        return []
    
    # Ordenar niveles
    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    # Agrupar niveles cercanos
    for i in range(1, len(sorted_levels)):
        current_level = sorted_levels[i]
        prev_level = current_cluster[-1]
        
        # Si el nivel está cerca del anterior, agregarlo al cluster actual
        if abs(current_level - prev_level) / prev_level <= threshold:
            current_cluster.append(current_level)
        else:
            # Calcular el promedio del cluster actual y comenzar uno nuevo
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [current_level]
    
    # Agregar el último cluster
    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))
    
    return clusters

def identify_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Identifica patrones de velas japonesas.
    
    Args:
        df (pd.DataFrame): DataFrame con datos OHLC
    
    Returns:
        Dict[str, List[Dict]]: Diccionario con patrones identificados
    """
    # Esta es una implementación básica que podría expandirse
    patterns = {
        "bullish": [],
        "bearish": []
    }
    
    # Verificar datos mínimos
    if len(df) < 3:
        return patterns
    
    # Recorrer los datos para encontrar patrones (excluyendo la última fila)
    for i in range(1, len(df) - 1):
        # Datos de las velas
        prev_candle = {
            "open": df["Open"].iloc[i-1],
            "high": df["High"].iloc[i-1],
            "low": df["Low"].iloc[i-1],
            "close": df["Close"].iloc[i-1]
        }
        
        current_candle = {
            "open": df["Open"].iloc[i],
            "high": df["High"].iloc[i],
            "low": df["Low"].iloc[i],
            "close": df["Close"].iloc[i],
            "datetime": df["Datetime"].iloc[i]
        }
        
        next_candle = {
            "open": df["Open"].iloc[i+1],
            "high": df["High"].iloc[i+1],
            "low": df["Low"].iloc[i+1],
            "close": df["Close"].iloc[i+1]
        }
        
        # 1. Patrón de martillo (bullish reversal)
        if (current_candle["close"] > current_candle["open"] and
            (current_candle["high"] - current_candle["close"]) < (current_candle["close"] - current_candle["open"]) * 0.1 and
            (current_candle["open"] - current_candle["low"]) > (current_candle["close"] - current_candle["open"]) * 2 and
            prev_candle["close"] < prev_candle["open"]):
            
            patterns["bullish"].append({
                "pattern": "Hammer",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })
        
        # 2. Patrón de estrella fugaz (bearish reversal)
        if (current_candle["close"] < current_candle["open"] and
            (current_candle["high"] - current_candle["open"]) > (current_candle["open"] - current_candle["close"]) * 2 and
            (current_candle["close"] - current_candle["low"]) < (current_candle["open"] - current_candle["close"]) * 0.1 and
            prev_candle["close"] > prev_candle["open"]):
            
            patterns["bearish"].append({
                "pattern": "Shooting Star",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })
        
        # 3. Patrón de envolvente alcista (bullish engulfing)
        if (current_candle["close"] > current_candle["open"] and
            prev_candle["close"] < prev_candle["open"] and
            current_candle["close"] > prev_candle["open"] and
            current_candle["open"] < prev_candle["close"]):
            
            patterns["bullish"].append({
                "pattern": "Bullish Engulfing",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })
        
        # 4. Patrón de envolvente bajista (bearish engulfing)
        if (current_candle["close"] < current_candle["open"] and
            prev_candle["close"] > prev_candle["open"] and
            current_candle["close"] < prev_candle["open"] and
            current_candle["open"] > prev_candle["close"]):
            
            patterns["bearish"].append({
                "pattern": "Bearish Engulfing",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })
    
    return patterns
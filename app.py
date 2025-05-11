import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.express as px
# import os # No usado, se puede quitar
import yfinance as yf
from typing import List, Dict, Tuple, Union # Mantener esto

st.set_page_config(page_title="Financial Analyst App", layout="wide")

st.title("Financial Analyst App")
st.write("Analiza datos de acciones y genera información.")

# --- Inicializar st.session_state ---
if 'df_data' not in st.session_state:
    st.session_state.df_data = None # DataFrame original cargado/descargado
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None # DataFrame con columnas estandarizadas e indicadores
if 'stock_symbol_display' not in st.session_state:
    st.session_state.stock_symbol_display = "" # Para mostrar en títulos de gráficos, etc.
if 'indicators_calculated' not in st.session_state:
    st.session_state.indicators_calculated = False
if 'signals_generated' not in st.session_state:
    st.session_state.signals_generated = False

# --- Funciones (las adaptaremos progresivamente) ---

def calculate_indicators(df: pd.DataFrame, selected_indicators: List[str]) -> pd.DataFrame:
    result = df.copy()
    required_columns = ["Open", "High", "Low", "Price", "Volume"] # 'Price' es nuestra columna estandarizada
    for col in required_columns:
        if col not in result.columns:
            # Intentar encontrar versiones en minúscula si las mayúsculas no están
            col_lower = col.lower()
            if col_lower in result.columns:
                result.rename(columns={col_lower: col}, inplace=True)
            # else: # Comentado para reducir verbosidad inicial, se puede reactivar
                # st.warning(f"Columna {col} (o {col_lower}) no encontrada. Algunos indicadores pueden no calcularse correctamente.")
                # Si una columna crítica como 'Price' falta, el indicador fallará de todas formas.
                # Pandas TA a menudo puede encontrar 'open', 'high', 'low', 'close', 'volume' automáticamente.

    # Asegurarse de que 'Price' exista antes de intentar usarla masivamente
    if "Price" not in result.columns:
        st.error("La columna 'Price' es esencial y no se encontró o no se pudo mapear. No se pueden calcular indicadores.")
        return result # Devolver el dataframe sin cambios o lanzar una excepción más específica

    for indicator in selected_indicators:
        try:
            if indicator == "MACD":
                # pandas_ta usa 'close' por defecto, así que si tenemos 'Price', podemos pasarla explícitamente.
                # O, si 'close' existe y es lo que queremos usar, pandas_ta lo tomará.
                # Aquí asumimos que 'Price' es la columna que queremos usar como 'close' para los indicadores.
                macd_df = ta.macd(result["Price"], fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty:
                    result = pd.concat([result, macd_df], axis=1)
                else:
                    st.warning(f"No se pudo calcular MACD. Verifique los datos de 'Price'.")
            elif indicator == "RSI":
                result["RSI"] = ta.rsi(result["Price"], length=14)
            elif indicator == "Bollinger Bands":
                bbands_df = ta.bbands(result["Price"], length=20, std=2)
                if bbands_df is not None and not bbands_df.empty:
                    result = pd.concat([result, bbands_df], axis=1)
                else:
                    st.warning(f"No se pudieron calcular Bollinger Bands.")
            elif indicator == "Stochastic":
                # Stochastic necesita High, Low, Price (Close)
                if all(col in result.columns for col in ["High", "Low", "Price"]):
                    stoch_df = ta.stoch(result["High"], result["Low"], result["Price"], k=14, d=3, smooth_k=3)
                    if stoch_df is not None and not stoch_df.empty:
                        result = pd.concat([result, stoch_df], axis=1)
                    else:
                        st.warning(f"No se pudo calcular Stochastic. Verifique columnas High, Low, Price.")
                else:
                    st.warning("Faltan columnas High, Low o Price para Stochastic.")
            elif indicator == "Moving Averages":
                result["SMA_20"] = ta.sma(result["Price"], length=20)
                result["SMA_50"] = ta.sma(result["Price"], length=50)
                result["SMA_200"] = ta.sma(result["Price"], length=200)
                result["EMA_12"] = ta.ema(result["Price"], length=12)
                result["EMA_26"] = ta.ema(result["Price"], length=26)
        except Exception as e:
            st.error(f"Error calculando el indicador {indicator}: {e}")
    return result

def identify_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    patterns = {"bullish": [], "bearish": []}
    required_cols = ["Open", "High", "Low", "Price"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Identify Patterns: Faltan una o más columnas requeridas: {', '.join(required_cols)}")
        return patterns
    if len(df) < 3: # Necesitamos al menos 3 velas para algunas lógicas que miran prev y current
        st.warning("Identify Patterns: No hay suficientes datos (se necesitan al menos 3 filas).")
        return patterns

    for i in range(1, len(df) - 1): # Ajustado para evitar ir más allá de los límites con next_candle si se usara
        current_datetime = df.index[i] # Usar el índice para la fecha/hora
        prev_candle = {
            "open": df["Open"].iloc[i-1], "high": df["High"].iloc[i-1],
            "low": df["Low"].iloc[i-1], "close": df["Price"].iloc[i-1]
        }
        current_candle = {
            "open": df["Open"].iloc[i], "high": df["High"].iloc[i],
            "low": df["Low"].iloc[i], "close": df["Price"].iloc[i],
            "datetime": current_datetime # Usar el índice (que es un Timestamp)
        }
        # next_candle no se usa en los patrones actuales, pero si se usara:
        # next_candle = {
        #     "open": df["Open"].iloc[i+1], "high": df["High"].iloc[i+1],
        #     "low": df["Low"].iloc[i+1], "close": df["Price"].iloc[i+1]
        # }

        # Patrón de martillo (bullish)
        body_size = abs(current_candle["close"] - current_candle["open"])
        if body_size == 0: body_size = 0.00001 # Evitar división por cero si es un doji

        is_bullish_candle = current_candle["close"] > current_candle["open"]
        upper_shadow = current_candle["high"] - current_candle["close"] if is_bullish_candle else current_candle["high"] - current_candle["open"]
        lower_shadow = current_candle["open"] - current_candle["low"] if is_bullish_candle else current_candle["close"] - current_candle["low"]

        # Hammer: cuerpo pequeño, sombra inferior larga (>2x cuerpo), sombra superior pequeña (<0.1-0.3x cuerpo), tendencia previa bajista
        if (is_bullish_candle and
            upper_shadow < body_size * 0.3 and # Sombra superior pequeña
            lower_shadow > body_size * 2 and   # Sombra inferior larga
            prev_candle["close"] < prev_candle["open"]): # Tendencia previa bajista (vela anterior roja)
            patterns["bullish"].append({
                "pattern": "Hammer", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de estrella fugaz (bearish)
        is_bearish_candle = current_candle["close"] < current_candle["open"]
        # Shooting Star: cuerpo pequeño, sombra superior larga (>2x cuerpo), sombra inferior pequeña (<0.1-0.3x cuerpo), tendencia previa alcista
        if (is_bearish_candle and # Puede ser cuerpo pequeño alcista o bajista, pero la señal es bajista
            upper_shadow > body_size * 2 and   # Sombra superior larga
            lower_shadow < body_size * 0.3 and # Sombra inferior pequeña
            prev_candle["close"] > prev_candle["open"]): # Tendencia previa alcista (vela anterior verde)
             patterns["bearish"].append({
                "pattern": "Shooting Star", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de envolvente alcista (bullish engulfing)
        if (is_bullish_candle and # Vela actual es alcista
            prev_candle["close"] < prev_candle["open"] and # Vela anterior es bajista
            current_candle["close"] > prev_candle["open"] and # Cierre actual envuelve apertura anterior
            current_candle["open"] < prev_candle["close"]):   # Apertura actual envuelve cierre anterior
            patterns["bullish"].append({
                "pattern": "Bullish Engulfing", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de envolvente bajista (bearish engulfing)
        if (is_bearish_candle and # Vela actual es bajista
            prev_candle["close"] > prev_candle["open"] and # Vela anterior es alcista
            current_candle["close"] < prev_candle["open"] and # Cierre actual envuelve apertura anterior (error en original, debe ser prev_candle["close"])
                                                               # No, el original estaba bien: el cuerpo de la actual envuelve el cuerpo de la anterior
            current_candle["open"] > prev_candle["close"]):   # Apertura actual envuelve cierre anterior
            patterns["bearish"].append({
                "pattern": "Bearish Engulfing", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })
    return patterns


def generate_signals(df: pd.DataFrame, selected_indicators: List[str]) -> Tuple[pd.DataFrame, Dict]:
    result = df.copy()
    signals = {"buy": [], "sell": [], "summary": {}, "strength": 0}
    result["Signal_Buy"] = 0
    result["Signal_Sell"] = 0
    result["Signal_Strength"] = 0 # No usado actualmente para generar Signal_Buy/Sell, pero calculado al final
    signal_count = 0 # Número de indicadores que contribuyen a la señal
    signal_value = 0 # Suma ponderada de señales (+ para compra, - para venta)

    # Nombres de columna de MACD pueden variar (ej. MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
    # Intentamos encontrar las columnas correctas
    macd_line_col = next((col for col in result.columns if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col), None)
    macd_signal_col = next((col for col in result.columns if 'MACDs_' in col), None) # MACD Signal line

    if "MACD" in selected_indicators and macd_line_col and macd_signal_col:
        signal_count += 1
        # Cruce alcista de MACD
        result.loc[(result[macd_line_col] > result[macd_signal_col]) &
                   (result[macd_line_col].shift(1) <= result[macd_signal_col].shift(1)), "Signal_Buy"] += 1
        # Cruce bajista de MACD
        result.loc[(result[macd_line_col] < result[macd_signal_col]) &
                   (result[macd_line_col].shift(1) >= result[macd_signal_col].shift(1)), "Signal_Sell"] += 1

        if not result.empty:
            if result[macd_line_col].iloc[-1] > result[macd_signal_col].iloc[-1]:
                signals["summary"]["MACD"] = "Compra"
                signal_value += 1
            elif result[macd_line_col].iloc[-1] < result[macd_signal_col].iloc[-1]: # Añadido elif para claridad
                signals["summary"]["MACD"] = "Venta"
                signal_value -= 1
            else:
                signals["summary"]["MACD"] = "Neutral"


    if "RSI" in selected_indicators and "RSI" in result.columns: # Asumiendo que la columna se llama 'RSI'
        signal_count += 1
        result.loc[result["RSI"] < 30, "Signal_Buy"] += 1
        result.loc[result["RSI"] > 70, "Signal_Sell"] += 1
        if not result.empty:
            rsi_value = result["RSI"].iloc[-1]
            if rsi_value < 30:
                signals["summary"]["RSI"] = "Fuerte Compra (Sobrevendido < 30)"
                signal_value += 2
            elif rsi_value < 45: # Ajustado el umbral para "Compra"
                signals["summary"]["RSI"] = "Compra (< 45)"
                signal_value += 1
            elif rsi_value > 70:
                signals["summary"]["RSI"] = "Fuerte Venta (Sobrecomprado > 70)"
                signal_value -= 2
            elif rsi_value > 55: # Ajustado el umbral para "Venta"
                signals["summary"]["RSI"] = "Venta (> 55)"
                signal_value -= 1
            else:
                signals["summary"]["RSI"] = "Neutral (45-55)"

    # Nombres de columna de Bollinger Bands (ej. BBL_20_2.0, BBU_20_2.0)
    bbl_col = next((col for col in result.columns if 'BBL_' in col), None)
    bbu_col = next((col for col in result.columns if 'BBU_' in col), None)

    if "Bollinger Bands" in selected_indicators and bbl_col and bbu_col and "Price" in result.columns:
        signal_count += 1
        result.loc[result["Price"] <= result[bbl_col], "Signal_Buy"] += 1
        result.loc[result["Price"] >= result[bbu_col], "Signal_Sell"] += 1
        if not result.empty:
            if result["Price"].iloc[-1] <= result[bbl_col].iloc[-1]:
                signals["summary"]["Bollinger"] = "Compra (Tocando/Cruzando Banda Inferior)"
                signal_value += 1
            elif result["Price"].iloc[-1] >= result[bbu_col].iloc[-1]:
                signals["summary"]["Bollinger"] = "Venta (Tocando/Cruzando Banda Superior)"
                signal_value -= 1
            else:
                signals["summary"]["Bollinger"] = "Neutral (Dentro de Bandas)"

    # Nombres de columna de Stochastic (ej. STOCHk_14_3_3, STOCHd_14_3_3)
    stoch_k_col = next((col for col in result.columns if 'STOCHk_' in col), None)
    stoch_d_col = next((col for col in result.columns if 'STOCHd_' in col), None)

    if "Stochastic" in selected_indicators and stoch_k_col and stoch_d_col:
        signal_count += 1
        # Cruce alcista de Estocástico en zona de sobreventa
        result.loc[(result[stoch_k_col] > result[stoch_d_col]) &
                   (result[stoch_k_col].shift(1) <= result[stoch_d_col].shift(1)) &
                   (result[stoch_k_col] < 20), "Signal_Buy"] += 1 # Algunos usan <20 o <30 para sobreventa
        # Cruce bajista de Estocástico en zona de sobrecompra
        result.loc[(result[stoch_k_col] < result[stoch_d_col]) &
                   (result[stoch_k_col].shift(1) >= result[stoch_d_col].shift(1)) &
                   (result[stoch_k_col] > 80), "Signal_Sell"] += 1 # Algunos usan >80 o >70 para sobrecompra

        if not result.empty:
            stoch_k_val = result[stoch_k_col].iloc[-1]
            stoch_d_val = result[stoch_d_col].iloc[-1]
            if stoch_k_val < 20 and stoch_k_val > stoch_d_val: # Cruce alcista reciente en sobreventa
                signals["summary"]["Stochastic"] = "Fuerte Compra (Cruce Alcista en Sobrevendido < 20)"
                signal_value += 2
            elif stoch_k_val > 80 and stoch_k_val < stoch_d_val: # Cruce bajista reciente en sobrecompra
                signals["summary"]["Stochastic"] = "Fuerte Venta (Cruce Bajista en Sobrecomprado > 80)"
                signal_value -= 2
            elif stoch_k_val > stoch_d_val: # Tendencia general del estocástico
                signals["summary"]["Stochastic"] = "Compra (K% > D%)"
                signal_value += 1
            elif stoch_k_val < stoch_d_val: # Añadido elif para claridad
                signals["summary"]["Stochastic"] = "Venta (K% < D%)"
                signal_value -= 1
            else:
                signals["summary"]["Stochastic"] = "Neutral"


    if "Moving Averages" in selected_indicators:
        has_ma_signal = False # Para el resumen
        # Cruce Dorado (SMA50 sobre SMA200 es a más largo plazo, SMA20 sobre SMA50 es más a corto/medio)
        if "SMA_20" in result.columns and "SMA_50" in result.columns:
            signal_count += 1 # Consideramos este cruce como una fuente de señal
            has_ma_signal = True
            # Cruce alcista de SMA20 sobre SMA50
            result.loc[(result["SMA_20"] > result["SMA_50"]) &
                       (result["SMA_20"].shift(1) <= result["SMA_50"].shift(1)), "Signal_Buy"] += 1
            # Cruce bajista de SMA20 sobre SMA50 (Cruce de la Muerte a corto plazo)
            result.loc[(result["SMA_20"] < result["SMA_50"]) &
                       (result["SMA_20"].shift(1) >= result["SMA_50"].shift(1)), "Signal_Sell"] += 1

            ma_summary_parts = []
            current_ma_value = 0

            if not result.empty:
                if result["SMA_20"].iloc[-1] > result["SMA_50"].iloc[-1]:
                    ma_summary_parts.append("SMA20 > SMA50 (Positivo Corto Plazo)")
                    current_ma_value += 1
                else:
                    ma_summary_parts.append("SMA20 < SMA50 (Negativo Corto Plazo)")
                    current_ma_value -= 1
                
                # Considerar precio relativo a SMAs para la señal actual
                if "Price" in result.columns:
                    if result["Price"].iloc[-1] > result["SMA_20"].iloc[-1]:
                        # No añadir a Signal_Buy aquí directamente, sino al resumen de fuerza
                        # result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5 # Original, pero mejor para resumen
                        pass # Se considera en signal_value
                    # else:
                        # result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
                    if result["Price"].iloc[-1] > result["SMA_50"].iloc[-1]:
                        # result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5
                        pass
                    # else:
                        # result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
            
            # Lógica para SMA200 si existe
            if "SMA_200" in result.columns and "Price" in result.columns and not result.empty:
                if result["Price"].iloc[-1] > result["SMA_200"].iloc[-1]:
                    ma_summary_parts.append("Precio > SMA200 (Positivo Largo Plazo)")
                    current_ma_value += 1.5 # Más peso a la tendencia a largo plazo
                else:
                    ma_summary_parts.append("Precio < SMA200 (Negativo Largo Plazo)")
                    current_ma_value -= 1.5
            
            if has_ma_signal : # Solo si se calcularon las MAs para el cruce
                signals["summary"]["Moving Averages"] = ". ".join(ma_summary_parts) if ma_summary_parts else "Neutral"
                signal_value += current_ma_value


    if signal_count > 0:
        # Normalizar la fuerza de la señal entre -100 y 100
        # El máximo valor posible para signal_value es signal_count * 2 (si todos dan Fuerte Compra)
        # El mínimo es signal_count * -2 (si todos dan Fuerte Venta)
        max_abs_value = signal_count * 2 if signal_count * 2 != 0 else 1 # Evitar división por cero
        signals["strength"] = int((signal_value / max_abs_value) * 100)


    # Llenar las listas de señales 'buy' y 'sell' para el historial
    # Usar el índice para datetime
    if not result.empty:
        for i in range(len(result)):
            if result["Signal_Buy"].iloc[i] >= 2: # Umbral para considerar una señal de compra fuerte
                signals["buy"].append({
                    "index": result.index[i], # Guardar el índice (timestamp)
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i] if "Price" in result.columns else None,
                    "strength": result["Signal_Buy"].iloc[i]
                })
            if result["Signal_Sell"].iloc[i] >= 2: # Umbral para considerar una señal de venta fuerte
                signals["sell"].append({
                    "index": result.index[i],
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i] if "Price" in result.columns else None,
                    "strength": result["Signal_Sell"].iloc[i]
                })

    result["Signal_Strength_Value"] = result["Signal_Buy"] - result["Signal_Sell"] # Diferencia neta de señales
    return result, signals


def backtesting_simple(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    if "Signal_Buy" not in df.columns or "Signal_Sell" not in df.columns or "Price" not in df.columns:
        st.error("Faltan columnas 'Signal_Buy', 'Signal_Sell', o 'Price' para el backtest.")
        return {
            "error": "No se encontraron señales o precios para el backtest",
            "final_capital": initial_capital, "return_pct": 0.0, "trades": 0,
            "avg_pnl": 0.0, "win_rate": 0.0, "trade_history": []
        }
    
    backtest_df = df.copy()
    capital = initial_capital
    position = 0  # 0: sin posición, >0: número de acciones compradas
    entry_price = 0.0
    trades = []
    n_trades = 0 # Contador de ciclos de compra-venta

    for i in range(1, len(backtest_df)): # Empezar desde 1 para poder mirar shift(1) si fuera necesario
        current_datetime = backtest_df.index[i] # Usar el índice para la fecha/hora
        current_price = backtest_df["Price"].iloc[i]

        # Decisión de Compra
        if position == 0 and backtest_df["Signal_Buy"].iloc[i] >= 2: # Umbral de compra
            if capital > 0 and current_price > 0: # Asegurarse de que hay capital y el precio es válido
                shares_to_buy = capital / current_price
                position = shares_to_buy
                entry_price = current_price
                capital = 0 # Todo el capital invertido
                trades.append({
                    "type": "buy",
                    "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
                    "price": entry_price,
                    "shares": shares_to_buy,
                    "capital_before": initial_capital if not trades else trades[-1].get("capital_after_trade", initial_capital) # Aprox
                })
        # Decisión de Venta
        elif position > 0 and backtest_df["Signal_Sell"].iloc[i] >= 2: # Umbral de venta
            exit_price = current_price
            capital_after_sell = position * exit_price
            pnl_trade = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            trades.append({
                "type": "sell",
                "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
                "price": exit_price,
                "shares": position,
                "capital_after_trade": capital_after_sell,
                "pnl_pct": pnl_trade
            })
            capital = capital_after_sell
            position = 0
            entry_price = 0
            n_trades += 1

    # Si al final del periodo hay una posición abierta, se cierra al último precio
    if position > 0 and not backtest_df.empty:
        exit_price = backtest_df["Price"].iloc[-1]
        current_datetime = backtest_df.index[-1]
        capital_after_sell = position * exit_price
        pnl_trade = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        trades.append({
            "type": "sell (cierre final)",
            "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
            "price": exit_price,
            "shares": position,
            "capital_after_trade": capital_after_sell,
            "pnl_pct": pnl_trade
        })
        capital = capital_after_sell
        n_trades +=1 # Cuenta como una operación si se cierra al final

    return_pct = ((capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0
    pnl_list = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    avg_pnl = np.mean(pnl_list) if pnl_list else 0
    win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) if pnl_list else 0

    return {
        "final_capital": capital,
        "return_pct": return_pct,
        "trades": n_trades, # Número de ciclos de compra-venta completados
        "avg_pnl": avg_pnl,
        "win_rate": win_rate * 100, # En porcentaje
        "trade_history": trades
    }

# --- Barra Lateral para Carga de Datos ---
st.sidebar.header("Configuración de Datos")
data_source = st.sidebar.radio("Fuente de Datos:", ("Subir CSV", "Descargar de Yahoo Finance"))

if data_source == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_temp = pd.read_csv(uploaded_file)
            st.session_state.df_data = df_temp.copy()
            st.session_state.stock_symbol_display = uploaded_file.name.split('.')[0] # Nombre base del archivo como símbolo
            st.sidebar.success("¡CSV cargado con éxito!")
            # Resetear estados dependientes
            st.session_state.df_processed = None
            st.session_state.indicators_calculated = False
            st.session_state.signals_generated = False
        except Exception as e:
            st.sidebar.error(f"Error al leer el CSV: {e}")
            st.session_state.df_data = None
else: # Descargar de Yahoo Finance
    default_symbol = "AAPL"
    user_symbol = st.sidebar.text_input("Símbolo de Acción (ej: AAPL, MSFT, BTC-USD)", default_symbol)
    
    today = datetime.now()
    default_start_date = today - timedelta(days=365*2) # Por defecto 2 años atrás

    start_date = st.sidebar.date_input("Fecha de Inicio", default_start_date)
    end_date = st.sidebar.date_input("Fecha de Fin", today)

    if st.sidebar.button("Obtener Datos de yfinance"):
        if user_symbol:
            if start_date >= end_date:
                st.sidebar.error("La fecha de inicio debe ser anterior a la fecha de fin.")
            else:
                try:
                    with st.spinner(f"Descargando datos para {user_symbol}..."):
                        df_temp = yf.download(user_symbol, start=start_date, end=end_date)
                    if df_temp.empty:
                        st.sidebar.warning("No se encontraron datos para el símbolo o rango de fechas especificado.")
                        st.session_state.df_data = None
                    else:
                        st.session_state.df_data = df_temp.copy()
                        st.session_state.stock_symbol_display = user_symbol
                        st.sidebar.success(f"¡Datos para {user_symbol} descargados!")
                        # Resetear estados dependientes
                        st.session_state.df_processed = None
                        st.session_state.indicators_calculated = False
                        st.session_state.signals_generated = False
                except Exception as e:
                    st.sidebar.error(f"Error al descargar datos de yfinance: {e}")
                    st.session_state.df_data = None
        else:
            st.sidebar.warning("Por favor, introduce un símbolo de acción.")


# --- Procesamiento y Visualización de Datos ---
if st.session_state.df_data is not None and st.session_state.df_processed is None:
    st.subheader("Preprocesamiento de Datos")
    with st.spinner("Procesando datos..."):
        df_work = st.session_state.df_data.copy()

        # 1. Manejo de Fechas y Establecer Índice
        date_col_found = False
        if isinstance(df_work.index, pd.DatetimeIndex): # Si yfinance ya puso DatetimeIndex
            df_work.index.name = "Timestamp" # Estandarizar nombre
            date_col_found = True
        else: # Para CSVs
            # Intentar encontrar columnas de fecha comunes
            date_candidates = [col for col in df_work.columns if col.lower() in ['date', 'datetime', 'timestamp', 'fecha']]
            if date_candidates:
                date_col_to_use = date_candidates[0]
                try:
                    df_work[date_col_to_use] = pd.to_datetime(df_work[date_col_to_use])
                    df_work.set_index(date_col_to_use, inplace=True)
                    df_work.index.name = "Timestamp"
                    date_col_found = True
                    st.write(f"Columna '{date_col_to_use}' establecida como índice de tiempo.")
                except Exception as e:
                    st.warning(f"No se pudo convertir la columna '{date_col_to_use}' a datetime o establecerla como índice: {e}")
            else: # Si no hay columna de fecha obvia y el índice no es datetime
                 st.warning("No se encontró una columna de fecha/datetime obvia ('Date', 'Datetime', 'Timestamp', 'Fecha'). "
                           "Asegúrate de que tu CSV tenga una o que el índice sea de tipo datetime.")

        if not date_col_found and not isinstance(df_work.index, pd.DatetimeIndex):
            st.error("No se pudo establecer un índice de tiempo. El análisis no puede continuar.")
            st.stop()
        
        # Ordenar por índice de tiempo por si acaso
        df_work.sort_index(inplace=True)

        # 2. Estandarizar Columnas OHLCV y 'Price'
        # Mapeo de nombres comunes (minúsculas) a estándar (Capitalizadas)
        rename_map = {}
        potential_price_cols = ['close', 'adj close', 'price', 'precio', 'valor'] # 'adj close' tiene prioridad
        ohlcv_map = {
            'Open': ['open', 'apertura'],
            'High': ['high', 'alto', 'maximo'],
            'Low': ['low', 'bajo', 'minimo'],
            'Volume': ['volume', 'volumen']
        }

        # Primero, intentar encontrar 'Adj Close' para 'Price'
        adj_close_col = next((col for col in df_work.columns if col.lower() == 'adj close'), None)
        if adj_close_col:
            df_work['Price'] = df_work[adj_close_col]
            if adj_close_col.lower() != 'price': # Evitar renombrar si ya se llama 'Price' (aunque sea en minúsculas)
                 rename_map[adj_close_col] = 'Price' # Marcar para renombrar o simplemente usar la nueva 'Price'
        else: # Si no hay 'Adj Close', buscar otras candidatas para 'Price'
            price_col_found = False
            for p_col_name in potential_price_cols:
                actual_price_col = next((col for col in df_work.columns if col.lower() == p_col_name), None)
                if actual_price_col:
                    df_work['Price'] = df_work[actual_price_col] # Crear la columna 'Price'
                    if actual_price_col.lower() != 'price':
                         rename_map[actual_price_col] = 'Price' # Marcar para renombrar si es diferente
                    price_col_found = True
                    break
            if not price_col_found and 'Price' not in df_work.columns: # Si después de todo, 'Price' no existe
                st.error("No se pudo encontrar una columna de precio ('Close', 'Adj Close', 'Price'). El análisis no puede continuar.")
                st.stop()
        
        # Renombrar otras columnas OHLCV
        for standard_name, common_names in ohlcv_map.items():
            if standard_name not in df_work.columns and standard_name not in rename_map.values(): # No sobrescribir si ya se mapeó 'Price' desde 'Close'
                for common_name in common_names:
                    actual_col = next((col for col in df_work.columns if col.lower() == common_name), None)
                    if actual_col:
                        # df_work.rename(columns={actual_col: standard_name}, inplace=True) # Renombrar directamente
                        # O mejor, crear las nuevas y luego seleccionar
                        if standard_name != 'Price': # 'Price' ya se manejó
                             df_work[standard_name] = df_work[actual_col]
                        break
        
        # Seleccionar solo las columnas estándar + 'Price' (si se creó) + otras que puedan existir
        standard_cols_present = [col for col in ['Timestamp', 'Open', 'High', 'Low', 'Price', 'Volume'] if col in df_work.columns or col == 'Timestamp'] # Timestamp es el índice
        
        # Mantener otras columnas que no sean las estandarizadas ni las originales que se renombraron
        other_cols = [col for col in df_work.columns if col not in standard_cols_present and col.lower() not in [p.lower() for p in potential_price_cols + sum(ohlcv_map.values(), [])]]
        
        df_final_cols = [col for col in ['Open', 'High', 'Low', 'Price', 'Volume'] if col in df_work.columns] # Columnas OHLCV+Price que realmente existen
        
        # df_work = df_work[df_final_cols + other_cols] # Esto podría perder el índice si no se maneja con cuidado
        # Es mejor asegurarse de que las columnas 'Open', 'High', 'Low', 'Price', 'Volume' existan con esos nombres.
        # Las funciones de indicadores las buscarán así.

        # 3. Manejo de Nulos (después de reestructurar columnas)
        if df_work[df_final_cols].isnull().values.any():
            st.write("Valores nulos encontrados. Aplicando forward fill (ffill) y luego backward fill (bfill).")
            for col in df_final_cols: # Aplicar solo a las columnas relevantes
                if df_work[col].isnull().any():
                    df_work[col].ffill(inplace=True)
                    df_work[col].bfill(inplace=True)
        
        # Verificar si aún hay nulos después del llenado (podría indicar columnas enteras de NaN)
        if df_work[df_final_cols].isnull().values.any():
            st.warning("Aún existen valores nulos después del llenado. Considere revisar sus datos. Se eliminarán filas con NaNs en columnas OHLCV/Price.")
            df_work.dropna(subset=df_final_cols, inplace=True)

        if df_work.empty:
            st.error("El DataFrame está vacío después del preprocesamiento. No se puede continuar.")
            st.stop()

        st.session_state.df_processed = df_work.copy()
        st.success("¡Datos preprocesados con éxito!")

# --- Mostrar Datos y Gráfico Principal si están procesados ---
if st.session_state.df_processed is not None:
    st.subheader(f"Datos para: {st.session_state.stock_symbol_display}")
    st.dataframe(st.session_state.df_processed.head())

    if "Price" in st.session_state.df_processed.columns:
        try:
            fig = px.line(st.session_state.df_processed, x=st.session_state.df_processed.index, y="Price",
                          title=f"Precio de Cierre para {st.session_state.stock_symbol_display}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar el gráfico de precios: {e}")
    else:
        st.warning("No se encontró la columna 'Price' en los datos procesados para graficar.")

    # --- Sección de Análisis Técnico ---
    st.header("Análisis Técnico")

    expander_indicators = st.expander("Calcular Indicadores Técnicos", expanded=False)
    with expander_indicators:
        available_indicators = ["MACD", "RSI", "Bollinger Bands", "Stochastic", "Moving Averages"]
        # Usar una clave única para el multiselect si hay varios
        selected_indicators = st.multiselect(
            "Elige los indicadores a calcular:",
            available_indicators,
            default=["RSI", "MACD"], # Algunos por defecto
            key="indicator_selector"
        )

        if st.button("1. Calcular Indicadores Seleccionados", key="calc_ind_btn"):
            if not selected_indicators:
                st.warning("Por favor, selecciona al menos un indicador.")
            else:
                with st.spinner("Calculando indicadores..."):
                    st.session_state.df_processed = calculate_indicators(st.session_state.df_processed, selected_indicators)
                    st.session_state.indicators_calculated = True
                    st.session_state.signals_generated = False # Resetear si se recalculan indicadores
                st.success("¡Indicadores calculados!")
                st.dataframe(st.session_state.df_processed.tail()) # Mostrar cola para ver indicadores recientes

    if st.session_state.indicators_calculated:
        expander_signals = st.expander("Generar Señales de Trading", expanded=False)
        with expander_signals:
            if st.button("2. Generar Señales", key="gen_sig_btn"):
                with st.spinner("Generando señales..."):
                    # Asegurarse de que selected_indicators esté disponible aquí, si es necesario para generate_signals
                    # Si generate_signals depende de los indicadores que ACABAN de ser seleccionados y calculados:
                    current_selected_indicators = st.session_state.get("indicator_selector", []) # Obtener la selección actual
                    
                    df_with_signals, signals_summary = generate_signals(st.session_state.df_processed, current_selected_indicators)
                    st.session_state.df_processed = df_with_signals # Actualizar el df con las señales
                    st.session_state.signals_generated = True
                st.success("¡Señales generadas!")
                st.subheader("Resumen de Señales (Último Dato):")
                
                # Formatear el resumen de señales para mejor visualización
                if signals_summary.get("summary"):
                    for k, v in signals_summary["summary"].items():
                        st.markdown(f"**{k}:** {v}")
                st.markdown(f"**Fuerza General de la Señal (último dato):** `{signals_summary.get('strength', 'N/A')}%`")
                
                st.subheader("Datos con Columnas de Señal (Últimos 5 Días):")
                st.dataframe(st.session_state.df_processed[['Price', 'Signal_Buy', 'Signal_Sell', 'Signal_Strength_Value']].tail())


    expander_patterns = st.expander("Identificar Patrones de Velas", expanded=False)
    with expander_patterns:
        if st.button("Identificar Patrones de Velas Japonesas", key="identify_patterns_btn"):
            if st.session_state.df_processed is not None and \
               all(col in st.session_state.df_processed.columns for col in ["Open", "High", "Low", "Price"]):
                with st.spinner("Identificando patrones..."):
                    patterns = identify_patterns(st.session_state.df_processed)
                st.success("¡Patrones identificados!")
                if patterns["bullish"] or patterns["bearish"]:
                    st.write("Patrones Alcistas Recientes:")
                    st.json(patterns["bullish"][-5:]) # Mostrar los últimos 5 encontrados
                    st.write("Patrones Bajistas Recientes:")
                    st.json(patterns["bearish"][-5:]) # Mostrar los últimos 5 encontrados
                else:
                    st.info("No se identificaron patrones con la lógica actual en los datos recientes.")
            else:
                st.warning("Se necesitan columnas 'Open', 'High', 'Low', 'Price' para identificar patrones. "
                           "Asegúrate de que los datos estén cargados y procesados.")


    if st.session_state.signals_generated:
        expander_backtest = st.expander("Realizar Backtesting Simple", expanded=False)
        with expander_backtest:
            initial_capital_bt = st.number_input("Capital Inicial para Backtesting:", min_value=100.0, value=10000.0, step=1000.0)
            if st.button("3. Ejecutar Backtesting", key="run_bt_btn"):
                with st.spinner("Ejecutando backtest..."):
                    backtest_results = backtesting_simple(st.session_state.df_processed, initial_capital_bt)
                
                st.subheader("Resultados del Backtesting")
                if "error" in backtest_results:
                    st.error(backtest_results["error"])
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Capital Final", f"${backtest_results['final_capital']:.2f}")
                    col2.metric("Retorno Total", f"{backtest_results['return_pct']:.2f}%")
                    col3.metric("Nº de Operaciones", backtest_results['trades'])
                    
                    col4, col5 = st.columns(2)
                    col4.metric("Rentabilidad Media / Op.", f"{backtest_results['avg_pnl']:.2f}%")
                    col5.metric("Tasa de Acierto (Win Rate)", f"{backtest_results['win_rate']:.2f}%")

                    st.subheader("Historial de Operaciones")
                    # Convertir historial a DataFrame para mejor visualización
                    if backtest_results['trade_history']:
                        df_history = pd.DataFrame(backtest_results['trade_history'])
                        # Formatear columnas si es necesario, por ejemplo, datetime
                        df_history['datetime'] = pd.to_datetime(df_history['datetime'])
                        st.dataframe(df_history)
                        
                        # Gráfico de evolución del capital (simple)
                        capital_over_time = [initial_capital_bt] + [trade['capital_after_trade'] for trade in backtest_results['trade_history'] if 'capital_after_trade' in trade and trade['type'] != 'buy']
                        trade_dates = [st.session_state.df_processed.index[0]] + [pd.to_datetime(trade['datetime']) for trade in backtest_results['trade_history'] if 'capital_after_trade' in trade and trade['type'] != 'buy']
                        
                        if len(trade_dates) == len(capital_over_time) and len(trade_dates) > 1:
                            df_capital_plot = pd.DataFrame({'Timestamp': trade_dates, 'Capital': capital_over_time}).set_index('Timestamp')
                            fig_capital = px.line(df_capital_plot, y="Capital", title="Evolución del Capital (Estimado Post-Venta)")
                            st.plotly_chart(fig_capital, use_container_width=True)
                        else:
                             st.info("No hay suficientes datos de operaciones de venta para graficar la evolución del capital.")

                    else:
                        st.info("No se realizaron operaciones en el backtest.")

else:
    st.info("⬅️ Por favor, carga un archivo CSV o descarga datos usando la barra lateral para comenzar.")

st.sidebar.markdown("---")
st.sidebar.markdown("Construido con Streamlit y cariño por la IA y tú.")
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

# --- DEBUGGING PRINTS ---
print("DEBUG: Script A - Antes de inicializar session_state.")
print(f"DEBUG: Tipo de st: {type(st)}")
if hasattr(st, 'session_state'):
    print(f"DEBUG: Tipo de st.session_state: {type(st.session_state)}")
    try:
        # Intentar listar claves si session_state ya es un objeto similar a un dict
        keys = list(st.session_state.keys()) # Esto podría fallar si session_state no está completamente listo
        print(f"DEBUG: Claves existentes en st.session_state (si las hay antes de init): {keys}")
    except Exception as e:
        print(f"DEBUG: Error al intentar acceder a las claves de st.session_state (antes de init): {e}")
else:
    print("DEBUG: st.session_state no existe como atributo de st todavía (antes de init).")
# --- FIN DEBUGGING PRINTS ---


# --- Inicializar st.session_state ---
if 'df_data' not in st.session_state:
    st.session_state.df_data = None # DataFrame original cargado/descargado
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None # DataFrame con columnas estandarizadas e indicadores
if 'stock_symbol_display' not in st.session_state:
    st.session_state.stock_symbol_display = "" # Para mostrar en títulos de gráficos, etc.
if 'indicators_calculated' not in st.session_state: # Esta es la línea 23 según el traceback del usuario
    st.session_state.indicators_calculated = False
if 'signals_generated' not in st.session_state:
    st.session_state.signals_generated = False

# --- Funciones (las adaptaremos progresivamente) ---

def calculate_indicators(df: pd.DataFrame, selected_indicators: List[str]) -> pd.DataFrame:
    result = df.copy()
    required_columns = ["Open", "High", "Low", "Price", "Volume"] # 'Price' es nuestra columna estandarizada
    for col in required_columns:
        if col not in result.columns:
            col_lower = col.lower()
            if col_lower in result.columns:
                result.rename(columns={col_lower: col}, inplace=True)

    if "Price" not in result.columns:
        st.error("La columna 'Price' es esencial y no se encontró o no se pudo mapear. No se pueden calcular indicadores.")
        return result

    for indicator in selected_indicators:
        try:
            if indicator == "MACD":
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
            print(f"ERROR_CALC_INDICATOR ({indicator}): {e}") # DEBUG PRINT
    return result

def identify_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    patterns = {"bullish": [], "bearish": []}
    required_cols = ["Open", "High", "Low", "Price"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Identify Patterns: Faltan una o más columnas requeridas: {', '.join(required_cols)}")
        return patterns
    if len(df) < 3:
        st.warning("Identify Patterns: No hay suficientes datos (se necesitan al menos 3 filas).")
        return patterns

    for i in range(1, len(df) - 1):
        current_datetime = df.index[i]
        prev_candle = {
            "open": df["Open"].iloc[i-1], "high": df["High"].iloc[i-1],
            "low": df["Low"].iloc[i-1], "close": df["Price"].iloc[i-1]
        }
        current_candle = {
            "open": df["Open"].iloc[i], "high": df["High"].iloc[i],
            "low": df["Low"].iloc[i], "close": df["Price"].iloc[i],
            "datetime": current_datetime
        }

        body_size = abs(current_candle["close"] - current_candle["open"])
        if body_size == 0: body_size = 0.00001

        is_bullish_candle = current_candle["close"] > current_candle["open"]
        is_bearish_candle = current_candle["close"] < current_candle["open"] # Definido para claridad

        # Sombra superior e inferior calculadas correctamente independientemente de si es alcista o bajista
        upper_shadow = current_candle["high"] - max(current_candle["open"], current_candle["close"])
        lower_shadow = min(current_candle["open"], current_candle["close"]) - current_candle["low"]


        if (is_bullish_candle and # Hammer es usualmente verde, pero algunos lo aceptan rojo si cumple las proporciones.
            upper_shadow < body_size * 0.3 and
            lower_shadow > body_size * 2 and
            prev_candle["close"] < prev_candle["open"]):
            patterns["bullish"].append({
                "pattern": "Hammer", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Shooting Star: cuerpo pequeño, sombra superior larga, sombra inferior pequeña, tendencia previa alcista
        # El cuerpo puede ser rojo o verde, pero la implicación es bajista.
        if (upper_shadow > body_size * 2 and
            lower_shadow < body_size * 0.3 and
            prev_candle["close"] > prev_candle["open"]):
             patterns["bearish"].append({
                "pattern": "Shooting Star", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        if (is_bullish_candle and
            prev_candle["close"] < prev_candle["open"] and
            current_candle["close"] > prev_candle["open"] and
            current_candle["open"] < prev_candle["close"]):
            patterns["bullish"].append({
                "pattern": "Bullish Engulfing", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        if (is_bearish_candle and
            prev_candle["close"] > prev_candle["open"] and
            current_candle["close"] < prev_candle["open"] and # Error original aquí: debe ser prev_candle['close'] para cuerpo real
            current_candle["open"] > prev_candle["close"]):   # Error original aquí: debe ser prev_candle['open'] para cuerpo real
                                                              # Corregido: current_candle.open > prev_candle.open AND current_candle.close < prev_candle.close
                                                              # La lógica original de envolver el cuerpo es:
                                                              # Bearish Engulfing: prev is green, current is red. current.open > prev.open, current.close < prev.close
                                                              #  Y current.open > prev.close, current.close < prev.open
            # Lógica original del usuario para envolvente bajista:
            # current_candle["close"] < current_candle["open"] (vela actual roja)
            # prev_candle["close"] > prev_candle["open"] (vela anterior verde)
            # current_candle["close"] < prev_candle["open"] (cierre actual por debajo de apertura anterior)
            # current_candle["open"] > prev_candle["close"] (apertura actual por encima de cierre anterior)
            # Esta lógica parece correcta para "outside bar" que envuelve el rango total de la anterior.
            # Para "body engulfing", sería current_close < prev_close y current_open > prev_open
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
    result["Signal_Strength"] = 0
    signal_count = 0
    signal_value = 0

    macd_line_col = next((col for col in result.columns if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col), None)
    macd_signal_col = next((col for col in result.columns if 'MACDs_' in col), None)

    if "MACD" in selected_indicators and macd_line_col and macd_signal_col:
        signal_count += 1
        result.loc[(result[macd_line_col] > result[macd_signal_col]) &
                   (result[macd_line_col].shift(1) <= result[macd_signal_col].shift(1)), "Signal_Buy"] += 1
        result.loc[(result[macd_line_col] < result[macd_signal_col]) &
                   (result[macd_line_col].shift(1) >= result[macd_signal_col].shift(1)), "Signal_Sell"] += 1

        if not result.empty:
            if result[macd_line_col].iloc[-1] > result[macd_signal_col].iloc[-1]:
                signals["summary"]["MACD"] = "Compra"
                signal_value += 1
            elif result[macd_line_col].iloc[-1] < result[macd_signal_col].iloc[-1]:
                signals["summary"]["MACD"] = "Venta"
                signal_value -= 1
            else:
                signals["summary"]["MACD"] = "Neutral"


    if "RSI" in selected_indicators and "RSI" in result.columns:
        signal_count += 1
        result.loc[result["RSI"] < 30, "Signal_Buy"] += 1
        result.loc[result["RSI"] > 70, "Signal_Sell"] += 1
        if not result.empty:
            rsi_value = result["RSI"].iloc[-1]
            if rsi_value < 30:
                signals["summary"]["RSI"] = "Fuerte Compra (Sobrevendido < 30)"
                signal_value += 2
            elif rsi_value < 45:
                signals["summary"]["RSI"] = "Compra (< 45)"
                signal_value += 1
            elif rsi_value > 70:
                signals["summary"]["RSI"] = "Fuerte Venta (Sobrecomprado > 70)"
                signal_value -= 2
            elif rsi_value > 55:
                signals["summary"]["RSI"] = "Venta (> 55)"
                signal_value -= 1
            else:
                signals["summary"]["RSI"] = "Neutral (45-55)"

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

    stoch_k_col = next((col for col in result.columns if 'STOCHk_' in col), None)
    stoch_d_col = next((col for col in result.columns if 'STOCHd_' in col), None)

    if "Stochastic" in selected_indicators and stoch_k_col and stoch_d_col:
        signal_count += 1
        result.loc[(result[stoch_k_col] > result[stoch_d_col]) &
                   (result[stoch_k_col].shift(1) <= result[stoch_d_col].shift(1)) &
                   (result[stoch_k_col] < 20), "Signal_Buy"] += 1
        result.loc[(result[stoch_k_col] < result[stoch_d_col]) &
                   (result[stoch_k_col].shift(1) >= result[stoch_d_col].shift(1)) &
                   (result[stoch_k_col] > 80), "Signal_Sell"] += 1

        if not result.empty:
            stoch_k_val = result[stoch_k_col].iloc[-1]
            stoch_d_val = result[stoch_d_col].iloc[-1]
            if stoch_k_val < 20 and stoch_k_val > stoch_d_val:
                signals["summary"]["Stochastic"] = "Fuerte Compra (Cruce Alcista en Sobrevendido < 20)"
                signal_value += 2
            elif stoch_k_val > 80 and stoch_k_val < stoch_d_val:
                signals["summary"]["Stochastic"] = "Fuerte Venta (Cruce Bajista en Sobrecomprado > 80)"
                signal_value -= 2
            elif stoch_k_val > stoch_d_val:
                signals["summary"]["Stochastic"] = "Compra (K% > D%)"
                signal_value += 1
            elif stoch_k_val < stoch_d_val:
                signals["summary"]["Stochastic"] = "Venta (K% < D%)"
                signal_value -= 1
            else:
                signals["summary"]["Stochastic"] = "Neutral"


    if "Moving Averages" in selected_indicators:
        has_ma_signal = False
        current_ma_value = 0 # Para el resumen de signal_value
        ma_summary_parts = []

        if "SMA_20" in result.columns and "SMA_50" in result.columns:
            # Solo contar una vez para signal_count si se usa cualquier cruce de MA
            if not has_ma_signal: signal_count += 1
            has_ma_signal = True
            
            result.loc[(result["SMA_20"] > result["SMA_50"]) &
                       (result["SMA_20"].shift(1) <= result["SMA_50"].shift(1)), "Signal_Buy"] += 1
            result.loc[(result["SMA_20"] < result["SMA_50"]) &
                       (result["SMA_20"].shift(1) >= result["SMA_50"].shift(1)), "Signal_Sell"] += 1

            if not result.empty:
                if result["SMA_20"].iloc[-1] > result["SMA_50"].iloc[-1]:
                    ma_summary_parts.append("SMA20 > SMA50 (Positivo Corto Plazo)")
                    current_ma_value += 1
                else:
                    ma_summary_parts.append("SMA20 < SMA50 (Negativo Corto Plazo)")
                    current_ma_value -= 1
            
        if "SMA_200" in result.columns and "Price" in result.columns and not result.empty:
            if not has_ma_signal: signal_count += 1 # Contar si este es el único indicador MA usado
            has_ma_signal = True
            if result["Price"].iloc[-1] > result["SMA_200"].iloc[-1]:
                ma_summary_parts.append("Precio > SMA200 (Positivo Largo Plazo)")
                current_ma_value += 1.5
            else:
                ma_summary_parts.append("Precio < SMA200 (Negativo Largo Plazo)")
                current_ma_value -= 1.5
            
        if has_ma_signal:
            signals["summary"]["Moving Averages"] = ". ".join(ma_summary_parts) if ma_summary_parts else "Neutral"
            signal_value += current_ma_value


    if signal_count > 0:
        max_abs_value = signal_count * 2 if signal_count * 2 != 0 else 1
        signals["strength"] = int((signal_value / max_abs_value) * 100)


    if not result.empty:
        for i in range(len(result)):
            if result["Signal_Buy"].iloc[i] >= 2:
                signals["buy"].append({
                    "index": result.index[i],
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i] if "Price" in result.columns else None,
                    "strength": result["Signal_Buy"].iloc[i]
                })
            if result["Signal_Sell"].iloc[i] >= 2:
                signals["sell"].append({
                    "index": result.index[i],
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i] if "Price" in result.columns else None,
                    "strength": result["Signal_Sell"].iloc[i]
                })

    result["Signal_Strength_Value"] = result["Signal_Buy"] - result["Signal_Sell"]
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
    position_shares = 0
    entry_price = 0.0
    trades = []
    n_completed_trades = 0

    for i in range(1, len(backtest_df)):
        current_datetime = backtest_df.index[i]
        current_price = backtest_df["Price"].iloc[i]

        if position_shares == 0 and backtest_df["Signal_Buy"].iloc[i] >= 2:
            if capital > 0 and current_price > 0:
                shares_to_buy = capital / current_price
                position_shares = shares_to_buy
                entry_price = current_price
                capital_before_buy = capital # Guardar capital antes de la compra
                capital = 0 
                trades.append({
                    "type": "buy",
                    "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
                    "price": entry_price,
                    "shares": shares_to_buy,
                    "capital_involved": capital_before_buy 
                })
        elif position_shares > 0 and backtest_df["Signal_Sell"].iloc[i] >= 2:
            exit_price = current_price
            capital_after_sell = position_shares * exit_price
            pnl_trade = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            trades.append({
                "type": "sell",
                "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
                "price": exit_price,
                "shares": position_shares,
                "capital_after_trade": capital_after_sell,
                "pnl_pct": pnl_trade
            })
            capital = capital_after_sell
            position_shares = 0
            entry_price = 0
            n_completed_trades += 1

    if position_shares > 0 and not backtest_df.empty:
        exit_price = backtest_df["Price"].iloc[-1]
        current_datetime = backtest_df.index[-1]
        capital_after_sell = position_shares * exit_price
        pnl_trade = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        trades.append({
            "type": "sell (cierre final)",
            "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
            "price": exit_price,
            "shares": position_shares,
            "capital_after_trade": capital_after_sell,
            "pnl_pct": pnl_trade
        })
        capital = capital_after_sell
        n_completed_trades +=1

    return_pct = ((capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0
    pnl_list = [t["pnl_pct"] for t in trades if t["type"] != "buy" and "pnl_pct" in t] # Solo pnl de ventas
    avg_pnl = np.mean(pnl_list) if pnl_list else 0
    win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) if pnl_list else 0

    return {
        "final_capital": capital,
        "return_pct": return_pct,
        "trades": n_completed_trades,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate * 100,
        "trade_history": trades
    }

# --- Barra Lateral para Carga de Datos ---
st.sidebar.header("Configuración de Datos")
data_source = st.sidebar.radio("Fuente de Datos:", ("Subir CSV", "Descargar de Yahoo Finance"))
uploaded_file = None # Inicializar para evitar UnboundLocalError si no se entra en la rama CSV

if data_source == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")
    print(f"DEBUG: En Subir CSV - uploaded_file (después de file_uploader): {uploaded_file}, Tipo: {type(uploaded_file)}") # DEBUG PRINT
    if uploaded_file is not None:
        print(f"DEBUG: En Subir CSV - uploaded_file TIENE VALOR. Nombre: {getattr(uploaded_file, 'name', 'SIN ATRIBUTO NAME')}") # DEBUG PRINT
        try:
            df_temp = pd.read_csv(uploaded_file)
            st.session_state.df_data = df_temp.copy()
            # Usar getattr para acceder a .name de forma segura
            file_name_for_display = getattr(uploaded_file, 'name', 'archivo_csv')
            st.session_state.stock_symbol_display = file_name_for_display.split('.')[0]
            st.sidebar.success("¡CSV cargado con éxito!")
            st.session_state.df_processed = None
            st.session_state.indicators_calculated = False
            st.session_state.signals_generated = False
        except Exception as e:
            print(f"ERROR DENTRO DEL BLOQUE 'uploaded_file is not None' (Subir CSV): {e}") # DEBUG PRINT
            st.sidebar.error(f"Error al leer el CSV: {e}")
            st.session_state.df_data = None
    else:
        print("DEBUG: En Subir CSV - uploaded_file es None, no se procesa.") # DEBUG PRINT
else: # Descargar de Yahoo Finance
    default_symbol = "AAPL"
    user_symbol = st.sidebar.text_input("Símbolo de Acción (ej: AAPL, MSFT, BTC-USD)", default_symbol)
    
    today = datetime.now()
    default_start_date = today - timedelta(days=365*2)

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
                        st.session_state.df_processed = None
                        st.session_state.indicators_calculated = False
                        st.session_state.signals_generated = False
                except Exception as e:
                    print(f"ERROR al descargar datos de yfinance: {e}") # DEBUG PRINT
                    st.sidebar.error(f"Error al descargar datos de yfinance: {e}")
                    st.session_state.df_data = None
        else:
            st.sidebar.warning("Por favor, introduce un símbolo de acción.")


# --- Procesamiento y Visualización de Datos ---
if st.session_state.df_data is not None and st.session_state.df_processed is None:
    st.subheader("Preprocesamiento de Datos")
    with st.spinner("Procesando datos..."):
        print("DEBUG: Iniciando bloque de preprocesamiento de datos.") # DEBUG PRINT
        df_work = st.session_state.df_data.copy()

        date_col_found = False
        if isinstance(df_work.index, pd.DatetimeIndex):
            df_work.index.name = "Timestamp"
            date_col_found = True
            print("DEBUG: Índice ya es DatetimeIndex, renombrado a Timestamp.") # DEBUG PRINT
        else:
            date_candidates = [col for col in df_work.columns if col.lower() in ['date', 'datetime', 'timestamp', 'fecha']]
            if date_candidates:
                date_col_to_use = date_candidates[0]
                print(f"DEBUG: Usando columna de fecha candidata: {date_col_to_use}") # DEBUG PRINT
                try:
                    df_work[date_col_to_use] = pd.to_datetime(df_work[date_col_to_use])
                    df_work.set_index(date_col_to_use, inplace=True)
                    df_work.index.name = "Timestamp"
                    date_col_found = True
                    st.write(f"Columna '{date_col_to_use}' establecida como índice de tiempo.")
                except Exception as e:
                    st.warning(f"No se pudo convertir la columna '{date_col_to_use}' a datetime o establecerla como índice: {e}")
                    print(f"ERROR_CONVERT_DATE_COL ({date_col_to_use}): {e}") # DEBUG PRINT
            else:
                 st.warning("No se encontró una columna de fecha/datetime obvia.")
                 print("DEBUG: No se encontró columna de fecha obvia en CSV.") # DEBUG PRINT

        if not date_col_found and not isinstance(df_work.index, pd.DatetimeIndex):
            st.error("No se pudo establecer un índice de tiempo. El análisis no puede continuar.")
            print("ERROR: No se pudo establecer índice de tiempo.") # DEBUG PRINT
            st.stop()
        
        df_work.sort_index(inplace=True)

        # Estandarizar Columnas OHLCV y 'Price'
        rename_map = {}
        potential_price_cols = ['adj close', 'close', 'price', 'precio', 'valor'] # adj close primero
        ohlcv_map = {
            'Open': ['open', 'apertura'], 'High': ['high', 'alto', 'maximo'],
            'Low': ['low', 'bajo', 'minimo'], 'Volume': ['volume', 'volumen']
        }
        
        # Crear columna 'Price'
        price_col_source = None
        for p_col_name in potential_price_cols:
            actual_price_col = next((col for col in df_work.columns if col.lower() == p_col_name), None)
            if actual_price_col:
                df_work['Price'] = df_work[actual_price_col]
                price_col_source = actual_price_col
                print(f"DEBUG: Columna 'Price' creada desde '{actual_price_col}'.") # DEBUG PRINT
                break
        if 'Price' not in df_work.columns:
            st.error("No se pudo encontrar/crear una columna de 'Price'. El análisis no puede continuar.")
            print("ERROR: No se pudo crear la columna 'Price'.") # DEBUG PRINT
            st.stop()

        # Estandarizar otras columnas OHLCV si existen y no son la fuente de 'Price'
        for standard_name, common_names in ohlcv_map.items():
            if standard_name not in df_work.columns: # Si no existe ya con el nombre estándar
                for common_name in common_names:
                    actual_col = next((col for col in df_work.columns if col.lower() == common_name), None)
                    if actual_col and (price_col_source is None or actual_col.lower() != price_col_source.lower()):
                        df_work[standard_name] = df_work[actual_col]
                        print(f"DEBUG: Columna '{standard_name}' creada desde '{actual_col}'.") # DEBUG PRINT
                        break
        
        final_cols_to_keep = [col for col in ['Open', 'High', 'Low', 'Price', 'Volume'] if col in df_work.columns]
        
        # Manejo de Nulos
        if df_work[final_cols_to_keep].isnull().values.any():
            st.write("Valores nulos encontrados. Aplicando forward fill (ffill) y luego backward fill (bfill).")
            print("DEBUG: Llenando NaNs.") # DEBUG PRINT
            for col in final_cols_to_keep:
                if df_work[col].isnull().any():
                    df_work[col].ffill(inplace=True)
                    df_work[col].bfill(inplace=True)
        
        if df_work[final_cols_to_keep].isnull().values.any():
            st.warning("Aún existen valores nulos después del llenado. Se eliminarán filas con NaNs en columnas OHLCV/Price.")
            print("DEBUG: Eliminando filas con NaNs restantes.") # DEBUG PRINT
            df_work.dropna(subset=final_cols_to_keep, inplace=True)

        if df_work.empty:
            st.error("El DataFrame está vacío después del preprocesamiento. No se puede continuar.")
            print("ERROR: DataFrame vacío post-preprocesamiento.") # DEBUG PRINT
            st.stop()

        st.session_state.df_processed = df_work.copy() # Guardar solo el df procesado
        st.success("¡Datos preprocesados con éxito!")
        print("DEBUG: Preprocesamiento de datos completado.")


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
            print(f"ERROR_PLOT_PRICE: {e}") # DEBUG PRINT
    else:
        st.warning("No se encontró la columna 'Price' en los datos procesados para graficar.")
        print("WARNING: No se encontró 'Price' para graficar.") # DEBUG PRINT

    # --- Sección de Análisis Técnico ---
    st.header("Análisis Técnico")

    expander_indicators = st.expander("Calcular Indicadores Técnicos", expanded=False)
    with expander_indicators:
        available_indicators = ["MACD", "RSI", "Bollinger Bands", "Stochastic", "Moving Averages"]
        selected_indicators = st.multiselect(
            "Elige los indicadores a calcular:",
            available_indicators,
            default=["RSI", "MACD"],
            key="indicator_selector"
        )

        if st.button("1. Calcular Indicadores Seleccionados", key="calc_ind_btn"):
            if not selected_indicators:
                st.warning("Por favor, selecciona al menos un indicador.")
            else:
                with st.spinner("Calculando indicadores..."):
                    # Usar una copia para la función, luego reasignar a session_state
                    df_temp_for_calc = st.session_state.df_processed.copy()
                    df_with_indicators = calculate_indicators(df_temp_for_calc, selected_indicators)
                    st.session_state.df_processed = df_with_indicators # Actualizar el df en session_state
                    st.session_state.indicators_calculated = True
                    st.session_state.signals_generated = False 
                st.success("¡Indicadores calculados!")
                st.dataframe(st.session_state.df_processed.tail())

    if st.session_state.indicators_calculated:
        expander_signals = st.expander("Generar Señales de Trading", expanded=False)
        with expander_signals:
            if st.button("2. Generar Señales", key="gen_sig_btn"):
                with st.spinner("Generando señales..."):
                    current_selected_indicators = st.session_state.get("indicator_selector", [])
                    
                    # Usar una copia para la función, luego reasignar a session_state
                    df_temp_for_signals = st.session_state.df_processed.copy()
                    df_with_signals, signals_summary = generate_signals(df_temp_for_signals, current_selected_indicators)
                    st.session_state.df_processed = df_with_signals # Actualizar el df en session_state
                    st.session_state.signals_generated = True
                st.success("¡Señales generadas!")
                st.subheader("Resumen de Señales (Último Dato):")
                
                if signals_summary.get("summary"):
                    for k, v in signals_summary["summary"].items():
                        st.markdown(f"**{k}:** {v}")
                st.markdown(f"**Fuerza General de la Señal (último dato):** `{signals_summary.get('strength', 'N/A')}%`")
                
                st.subheader("Datos con Columnas de Señal (Últimos 5 Días):")
                cols_to_show_signals = ['Price'] + [col for col in ['Signal_Buy', 'Signal_Sell', 'Signal_Strength_Value'] if col in st.session_state.df_processed.columns]
                st.dataframe(st.session_state.df_processed[cols_to_show_signals].tail())


    expander_patterns = st.expander("Identificar Patrones de Velas", expanded=False)
    with expander_patterns:
        if st.button("Identificar Patrones de Velas Japonesas", key="identify_patterns_btn"):
            if st.session_state.df_processed is not None and \
               all(col in st.session_state.df_processed.columns for col in ["Open", "High", "Low", "Price"]):
                with st.spinner("Identificando patrones..."):
                    patterns = identify_patterns(st.session_state.df_processed) # Usa el df de session_state directamente (es una copia)
                st.success("¡Patrones identificados!")
                if patterns["bullish"] or patterns["bearish"]:
                    st.write("Patrones Alcistas Recientes:")
                    st.json(patterns["bullish"][-5:])
                    st.write("Patrones Bajistas Recientes:")
                    st.json(patterns["bearish"][-5:])
                else:
                    st.info("No se identificaron patrones con la lógica actual en los datos recientes.")
            else:
                st.warning("Se necesitan columnas 'Open', 'High', 'Low', 'Price' para identificar patrones.")


    if st.session_state.signals_generated:
        expander_backtest = st.expander("Realizar Backtesting Simple", expanded=False)
        with expander_backtest:
            initial_capital_bt = st.number_input("Capital Inicial para Backtesting:", min_value=100.0, value=10000.0, step=1000.0)
            if st.button("3. Ejecutar Backtesting", key="run_bt_btn"):
                with st.spinner("Ejecutando backtest..."):
                    # Usa el df de session_state directamente (es una copia)
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
                    if backtest_results['trade_history']:
                        df_history = pd.DataFrame(backtest_results['trade_history'])
                        if 'datetime' in df_history.columns:
                             df_history['datetime'] = pd.to_datetime(df_history['datetime'])
                        st.dataframe(df_history)
                        
                        capital_over_time = [initial_capital_bt] + [trade['capital_after_trade'] for trade in backtest_results['trade_history'] if 'capital_after_trade' in trade and trade.get('type','').lower() != 'buy']
                        trade_dates_valid = [st.session_state.df_processed.index[0]] + \
                                            [pd.to_datetime(trade['datetime']) for trade in backtest_results['trade_history'] if 'capital_after_trade' in trade and trade.get('type','').lower() != 'buy' and 'datetime' in trade]
                        
                        if len(trade_dates_valid) == len(capital_over_time) and len(trade_dates_valid) > 1:
                            df_capital_plot = pd.DataFrame({'Timestamp': trade_dates_valid, 'Capital': capital_over_time}).set_index('Timestamp')
                            fig_capital = px.line(df_capital_plot, y="Capital", title="Evolución del Capital (Estimado Post-Venta)")
                            st.plotly_chart(fig_capital, use_container_width=True)
                        else:
                             st.info("No hay suficientes datos de operaciones de venta para graficar la evolución del capital.")
                    else:
                        st.info("No se realizaron operaciones en el backtest.")
else:
    print("DEBUG: df_processed es None, mostrando mensaje para cargar datos.") # DEBUG PRINT
    st.info("⬅️ Por favor, carga un archivo CSV o descarga datos usando la barra lateral para comenzar.")

st.sidebar.markdown("---")
st.sidebar.markdown("App mejorada con la ayuda de IA.")
print("DEBUG: Fin del script.") # DEBUG PRINT
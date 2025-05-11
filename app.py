import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.express as px
# import os # No usado
import yfinance as yf
from typing import List, Dict, Tuple, Union

st.set_page_config(page_title="Financial Analyst App", layout="wide")

st.title("Financial Analyst App")
st.write("Analiza datos de acciones y genera información.")

# --- DEBUGGING PRINTS ---
print("DEBUG: Script A - Antes de inicializar session_state.")
print(f"DEBUG: Tipo de st: {type(st)}")
if hasattr(st, 'session_state'):
    print(f"DEBUG: Tipo de st.session_state: {type(st.session_state)}")
    try:
        keys = list(st.session_state.keys())
        print(f"DEBUG: Claves existentes en st.session_state (antes de init): {keys}")
    except Exception as e:
        print(f"DEBUG: Error al intentar acceder a las claves de st.session_state (antes de init): {e}")
else:
    print("DEBUG: st.session_state no existe como atributo de st todavía (antes de init).")
# --- FIN DEBUGGING PRINTS ---


# --- Inicializar st.session_state ---
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'stock_symbol_display' not in st.session_state:
    st.session_state.stock_symbol_display = ""
if 'indicators_calculated' not in st.session_state:
    st.session_state.indicators_calculated = False
if 'signals_generated' not in st.session_state:
    st.session_state.signals_generated = False

# --- Funciones ---

def calculate_indicators(df: pd.DataFrame, selected_indicators: List[str]) -> pd.DataFrame:
    result = df.copy()
    # Las columnas ya deberían ser numéricas y con nombres estándar desde el preprocesamiento
    if "Price" not in result.columns or not pd.api.types.is_numeric_dtype(result["Price"]):
        st.error("La columna 'Price' no es numérica o no existe. No se pueden calcular indicadores.")
        print("ERROR_CALC_INDICATOR: 'Price' no numérica o ausente.")
        return df # Devolver el df original si Price no es válida

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
                required_stoch_cols = ["High", "Low", "Price"]
                if all(col in result.columns and pd.api.types.is_numeric_dtype(result[col]) for col in required_stoch_cols):
                    stoch_df = ta.stoch(result["High"], result["Low"], result["Price"], k=14, d=3, smooth_k=3)
                    if stoch_df is not None and not stoch_df.empty:
                        result = pd.concat([result, stoch_df], axis=1)
                    else:
                        st.warning(f"No se pudo calcular Stochastic.")
                else:
                    st.warning(f"Faltan columnas numéricas High, Low o Price para Stochastic. Columnas presentes: {result.columns.tolist()}")
                    for col_s in required_stoch_cols:
                        if col_s in result.columns:
                            print(f"DEBUG_STOCH: Col {col_s} tipo: {result[col_s].dtype}")
                        else:
                             print(f"DEBUG_STOCH: Col {col_s} AUSENTE")
            elif indicator == "Moving Averages":
                result["SMA_20"] = ta.sma(result["Price"], length=20)
                result["SMA_50"] = ta.sma(result["Price"], length=50)
                result["SMA_200"] = ta.sma(result["Price"], length=200)
                result["EMA_12"] = ta.ema(result["Price"], length=12)
                result["EMA_26"] = ta.ema(result["Price"], length=26)
        except Exception as e:
            st.error(f"Error calculando el indicador {indicator}: {e}")
            print(f"ERROR_CALC_INDICATOR ({indicator}): {e}")
    return result

def identify_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    patterns = {"bullish": [], "bearish": []}
    required_cols = ["Open", "High", "Low", "Price"]
    if not all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in required_cols):
        st.warning(f"Identify Patterns: Faltan una o más columnas numéricas requeridas: {', '.join(required_cols)}")
        print(f"ERROR_ID_PATTERNS: Faltan columnas numéricas. Columnas: {df.columns.tolist()}")
        for col_p in required_cols:
            if col_p in df.columns: print(f"DEBUG_ID_PATTERNS: Col {col_p} tipo: {df[col_p].dtype}")
            else: print(f"DEBUG_ID_PATTERNS: Col {col_p} AUSENTE")
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
        if body_size == 0: body_size = 0.00001 # Evitar división por cero si es un doji

        is_bullish_candle = current_candle["close"] > current_candle["open"]
        is_bearish_candle = current_candle["close"] < current_candle["open"]

        upper_shadow = current_candle["high"] - max(current_candle["open"], current_candle["close"])
        lower_shadow = min(current_candle["open"], current_candle["close"]) - current_candle["low"]

        if (is_bullish_candle and
            upper_shadow < body_size * 0.3 and
            lower_shadow > body_size * 2 and
            prev_candle["close"] < prev_candle["open"]):
            patterns["bullish"].append({
                "pattern": "Hammer", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        if (upper_shadow > body_size * 2 and
            lower_shadow < body_size * 0.3 and
            prev_candle["close"] > prev_candle["open"]):
             patterns["bearish"].append({
                "pattern": "Shooting Star", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        if (is_bullish_candle and
            prev_candle["close"] < prev_candle["open"] and # Vela anterior es bajista
            current_candle["close"] > prev_candle["open"] and # Cierre actual > Apertura anterior
            current_candle["open"] < prev_candle["close"]):   # Apertura actual < Cierre anterior
            patterns["bullish"].append({
                "pattern": "Bullish Engulfing", "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        if (is_bearish_candle and
            prev_candle["close"] > prev_candle["open"] and # Vela anterior es alcista
            current_candle["open"] > prev_candle["close"] and # Apertura actual > Cierre anterior
            current_candle["close"] < prev_candle["open"]):   # Cierre actual < Apertura anterior
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
    # result["Signal_Strength"] = 0 # Ya no se usa así directamente, se calcula al final
    signal_count = 0
    signal_value = 0

    macd_line_col = next((col for col in result.columns if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col), None)
    macd_signal_col = next((col for col in result.columns if 'MACDs_' in col), None)

    if "MACD" in selected_indicators and macd_line_col and macd_signal_col:
        if pd.api.types.is_numeric_dtype(result[macd_line_col]) and pd.api.types.is_numeric_dtype(result[macd_signal_col]):
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
        else: st.warning("Columnas MACD no son numéricas.")


    if "RSI" in selected_indicators and "RSI" in result.columns:
        if pd.api.types.is_numeric_dtype(result["RSI"]):
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
        else: st.warning("Columna RSI no es numérica.")

    bbl_col = next((col for col in result.columns if 'BBL_' in col), None)
    bbu_col = next((col for col in result.columns if 'BBU_' in col), None)

    if "Bollinger Bands" in selected_indicators and bbl_col and bbu_col and "Price" in result.columns:
        if pd.api.types.is_numeric_dtype(result[bbl_col]) and pd.api.types.is_numeric_dtype(result[bbu_col]) and pd.api.types.is_numeric_dtype(result["Price"]):
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
        else: st.warning("Columnas de Bollinger Bands o Price no son numéricas.")

    stoch_k_col = next((col for col in result.columns if 'STOCHk_' in col), None)
    stoch_d_col = next((col for col in result.columns if 'STOCHd_' in col), None)

    if "Stochastic" in selected_indicators and stoch_k_col and stoch_d_col:
        if pd.api.types.is_numeric_dtype(result[stoch_k_col]) and pd.api.types.is_numeric_dtype(result[stoch_d_col]):
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
        else: st.warning("Columnas de Stochastic no son numéricas.")


    if "Moving Averages" in selected_indicators:
        has_ma_signal = False
        current_ma_value = 0
        ma_summary_parts = []
        sma20_ok = "SMA_20" in result.columns and pd.api.types.is_numeric_dtype(result["SMA_20"])
        sma50_ok = "SMA_50" in result.columns and pd.api.types.is_numeric_dtype(result["SMA_50"])
        sma200_ok = "SMA_200" in result.columns and pd.api.types.is_numeric_dtype(result["SMA_200"])
        price_ok = "Price" in result.columns and pd.api.types.is_numeric_dtype(result["Price"])


        if sma20_ok and sma50_ok:
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
            
        if sma200_ok and price_ok and not result.empty:
            if not has_ma_signal: signal_count += 1
            has_ma_signal = True # Aunque ya sea true, está bien
            if result["Price"].iloc[-1] > result["SMA_200"].iloc[-1]:
                ma_summary_parts.append("Precio > SMA200 (Positivo Largo Plazo)")
                current_ma_value += 1.5
            else:
                ma_summary_parts.append("Precio < SMA200 (Negativo Largo Plazo)")
                current_ma_value -= 1.5
            
        if has_ma_signal:
            signals["summary"]["Moving Averages"] = ". ".join(ma_summary_parts) if ma_summary_parts else "Neutral"
            signal_value += current_ma_value
        elif "Moving Averages" in selected_indicators: # Si se seleccionó pero no se pudo calcular nada
             st.warning("No se pudieron usar las Medias Móviles para señales (columnas ausentes o no numéricas).")


    if signal_count > 0:
        max_abs_value = signal_count * 2 if signal_count * 2 != 0 else 1
        signals["strength"] = int((signal_value / max_abs_value) * 100)


    if not result.empty and "Price" in result.columns and pd.api.types.is_numeric_dtype(result["Price"]):
        for i in range(len(result)):
            if result["Signal_Buy"].iloc[i] >= 2:
                signals["buy"].append({
                    "index": result.index[i],
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i],
                    "strength": result["Signal_Buy"].iloc[i]
                })
            if result["Signal_Sell"].iloc[i] >= 2:
                signals["sell"].append({
                    "index": result.index[i],
                    "datetime": result.index[i].strftime('%Y-%m-%d %H:%M:%S') if isinstance(result.index[i], pd.Timestamp) else str(result.index[i]),
                    "price": result["Price"].iloc[i],
                    "strength": result["Signal_Sell"].iloc[i]
                })

    result["Signal_Strength_Value"] = result["Signal_Buy"] - result["Signal_Sell"]
    return result, signals


def backtesting_simple(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    if not all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in ["Signal_Buy", "Signal_Sell", "Price"]):
        st.error("Faltan columnas numéricas 'Signal_Buy', 'Signal_Sell', o 'Price' para el backtest.")
        return {
            "error": "No se encontraron señales o precios numéricos para el backtest",
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
                capital_before_buy = capital
                capital = 0 
                trades.append({
                    "type": "buy",
                    "datetime": current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, pd.Timestamp) else str(current_datetime),
                    "price": entry_price,
                    "shares": shares_to_buy,
                    "capital_involved": capital_before_buy 
                })
        elif position_shares > 0 and backtest_df["Signal_Sell"].iloc[i] >= 2:
            if current_price > 0 : # Vender solo si el precio es válido
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
        if backtest_df["Price"].iloc[-1] > 0: # Cerrar solo si el precio es válido
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
    pnl_list = [t["pnl_pct"] for t in trades if t["type"] != "buy" and "pnl_pct" in t]
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
uploaded_file = None 

if data_source == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type="csv")
    print(f"DEBUG: En Subir CSV - uploaded_file (después de file_uploader): {uploaded_file}, Tipo: {type(uploaded_file)}")
    if uploaded_file is not None:
        print(f"DEBUG: En Subir CSV - uploaded_file TIENE VALOR. Nombre: {getattr(uploaded_file, 'name', 'SIN ATRIBUTO NAME')}")
        try:
            df_temp = pd.read_csv(uploaded_file)
            st.session_state.df_data = df_temp.copy()
            file_name_for_display = getattr(uploaded_file, 'name', 'archivo_csv')
            st.session_state.stock_symbol_display = file_name_for_display.split('.')[0]
            st.sidebar.success("¡CSV cargado con éxito!")
            st.session_state.df_processed = None
            st.session_state.indicators_calculated = False
            st.session_state.signals_generated = False
        except Exception as e:
            print(f"ERROR DENTRO DEL BLOQUE 'uploaded_file is not None' (Subir CSV): {e}")
            st.sidebar.error(f"Error al leer el CSV: {e}")
            st.session_state.df_data = None
    else:
        print("DEBUG: En Subir CSV - uploaded_file es None, no se procesa.")
else: 
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
                    print(f"ERROR al descargar datos de yfinance: {e}")
                    st.sidebar.error(f"Error al descargar datos de yfinance: {e}")
                    st.session_state.df_data = None
        else:
            st.sidebar.warning("Por favor, introduce un símbolo de acción.")


# --- Procesamiento y Visualización de Datos ---
if st.session_state.df_data is not None and st.session_state.df_processed is None:
    st.subheader("Preprocesamiento de Datos")
    with st.spinner("Procesando datos..."):
        print("DEBUG: Iniciando bloque de preprocesamiento de datos.")
        df_work = st.session_state.df_data.copy()

        date_col_found = False
        if isinstance(df_work.index, pd.DatetimeIndex):
            df_work.index.name = "Timestamp"
            date_col_found = True
            print("DEBUG: Índice ya es DatetimeIndex, renombrado a Timestamp.")
        else:
            date_candidates = [col for col in df_work.columns if col.lower() in ['date', 'datetime', 'timestamp', 'fecha']]
            if date_candidates:
                date_col_to_use = date_candidates[0]
                print(f"DEBUG: Usando columna de fecha candidata: {date_col_to_use}")
                try:
                    df_work[date_col_to_use] = pd.to_datetime(df_work[date_col_to_use])
                    df_work.set_index(date_col_to_use, inplace=True)
                    df_work.index.name = "Timestamp"
                    date_col_found = True
                    st.write(f"Columna '{date_col_to_use}' establecida como índice de tiempo.")
                except Exception as e:
                    st.warning(f"No se pudo convertir la columna '{date_col_to_use}' a datetime o establecerla como índice: {e}")
                    print(f"ERROR_CONVERT_DATE_COL ({date_col_to_use}): {e}")
            else:
                 st.warning("No se encontró una columna de fecha/datetime obvia.")
                 print("DEBUG: No se encontró columna de fecha obvia en CSV.")

        if not date_col_found and not isinstance(df_work.index, pd.DatetimeIndex):
            st.error("No se pudo establecer un índice de tiempo. El análisis no puede continuar.")
            print("ERROR: No se pudo establecer índice de tiempo.")
            st.stop()
        
        df_work.sort_index(inplace=True)

        # --- CONVERSIÓN NUMÉRICA Y ESTANDARIZACIÓN DE COLUMNAS ---
        print("DEBUG: Iniciando estandarización de columnas y conversión numérica.")
        price_col_source_name = None # Para saber de dónde vino 'Price' y no intentar renombrarla dos veces

        # 1. Crear columna 'Price' (priorizando 'Adj Close', luego 'Close', etc.)
        potential_price_cols_map = {'Adj Close': 'adj close', 'Close': 'close', 'Price': 'price'} # Mantener capitalización original si es posible
        for display_name, lower_name in potential_price_cols_map.items():
            actual_col_name = next((col for col in df_work.columns if col.lower() == lower_name), None)
            if actual_col_name:
                df_work['Price'] = pd.to_numeric(df_work[actual_col_name], errors='coerce')
                price_col_source_name = actual_col_name # Guardar el nombre original de la fuente de Price
                print(f"DEBUG: Columna 'Price' creada desde '{actual_col_name}' y convertida a numérico. Tipo: {df_work['Price'].dtype}")
                break
        
        if 'Price' not in df_work.columns: # Si después de los intentos, 'Price' no existe
            st.error("No se pudo encontrar o crear una columna 'Price' numérica válida. El análisis no puede continuar.")
            print("ERROR: No se pudo crear la columna 'Price' numérica.")
            st.stop()

        # 2. Estandarizar y convertir otras columnas OHLC
        ohl_map = {'Open': 'open', 'High': 'high', 'Low': 'low'}
        for standard_name, lower_name in ohl_map.items():
            if standard_name not in df_work.columns: # Si no existe ya con el nombre estándar
                actual_col_name = next((col for col in df_work.columns if col.lower() == lower_name), None)
                if actual_col_name:
                    # Solo convertir si no fue la fuente de 'Price' (ej. si 'Price' vino de 'Close', no procesar 'Close' aquí de nuevo)
                    if price_col_source_name is None or actual_col_name.lower() != price_col_source_name.lower():
                        df_work[standard_name] = pd.to_numeric(df_work[actual_col_name], errors='coerce')
                        print(f"DEBUG: Columna '{standard_name}' creada desde '{actual_col_name}' y convertida a numérico. Tipo: {df_work[standard_name].dtype if standard_name in df_work.columns else 'Error'}")
                    # Si actual_col_name fue la fuente de Price, y queremos una columna Open/High/Low con el mismo nombre original
                    # (ej. Price vino de 'Close', pero también tenemos 'Open'), esto ya se manejó si el nombre es diferente de 'Price'.
                    # Si el nombre es igual (ej. CSV tiene solo 'Price' y no 'Open', 'High', 'Low'), no se crearán.
                else:
                     print(f"DEBUG: Columna estándar '{standard_name}' (o su variante '{lower_name}') no encontrada en el CSV.")


        # 3. Manejar columna de Volumen (ej. 'Vol.' del CSV)
        # La imagen del usuario muestra 'Vol.'
        volume_col_original_name = next((col for col in df_work.columns if col.lower() in ['vol.', 'volume', 'volumen']), None)
        if volume_col_original_name:
            print(f"DEBUG: Procesando columna de volumen original: '{volume_col_original_name}'")
            vol_col_data = df_work[volume_col_original_name].astype(str).str.upper()
            
            def convert_volume_value(value):
                value = str(value).replace(',', '') # Eliminar comas si las hay
                if 'M' in value: return pd.to_numeric(value.replace('M', ''), errors='coerce') * 1_000_000
                elif 'K' in value: return pd.to_numeric(value.replace('K', ''), errors='coerce') * 1_000
                elif 'B' in value: return pd.to_numeric(value.replace('B', ''), errors='coerce') * 1_000_000_000
                return pd.to_numeric(value, errors='coerce')

            df_work['Volume'] = vol_col_data.apply(convert_volume_value)
            print(f"DEBUG: Columna 'Volume' (desde '{volume_col_original_name}') convertida. Tipo: {df_work['Volume'].dtype if 'Volume' in df_work.columns else 'No procesada'}")
            # Opcional: eliminar la columna de volumen original si el nombre era diferente de 'Volume' y no es la misma que la fuente de precio
            if volume_col_original_name.lower() != 'volume' and \
               (price_col_source_name is None or volume_col_original_name.lower() != price_col_source_name.lower()):
                df_work.drop(columns=[volume_col_original_name], inplace=True, errors='ignore')
                print(f"DEBUG: Columna original '{volume_col_original_name}' eliminada después de crear 'Volume'.")
        else:
            print("DEBUG: No se encontró columna de Volumen ('Vol.', 'volume', 'volumen').")


        # 4. Limpieza de NaNs introducidos por 'coerce' en columnas cruciales
        crucial_numeric_cols = [col for col in ['Open', 'High', 'Low', 'Price'] if col in df_work.columns and pd.api.types.is_numeric_dtype(df_work[col])]
        if crucial_numeric_cols:
            original_len = len(df_work)
            df_work.dropna(subset=crucial_numeric_cols, inplace=True) # Elimina filas si ALGUNA de estas es NaN
            if len(df_work) < original_len:
                print(f"DEBUG: Se eliminaron {original_len - len(df_work)} filas debido a NaNs en {crucial_numeric_cols} post-conversión.")
        
        # 5. Verificación final de columnas requeridas para el análisis
        final_required_cols = ['Price'] # Mínimo absoluto
        # Puedes añadir Open, High, Low aquí si son estrictamente necesarias para *todos* los análisis posteriores
        # y detener si no existen y son numéricas.
        for req_col in final_required_cols:
            if req_col not in df_work.columns or not pd.api.types.is_numeric_dtype(df_work[req_col]):
                st.error(f"La columna requerida y numérica '{req_col}' no está disponible después del preprocesamiento.")
                print(f"ERROR: Columna '{req_col}' no disponible/numérica post-preprocesamiento.")
                st.stop()


        if df_work.empty:
            st.error("El DataFrame está vacío después del preprocesamiento. No se puede continuar.")
            print("ERROR: DataFrame vacío post-preprocesamiento.")
            st.stop()

        st.session_state.df_processed = df_work.copy()
        st.success("¡Datos preprocesados con éxito!")
        print("DEBUG: Preprocesamiento de datos completado.")
        print(f"DEBUG: Tipos de datos finales en df_processed: \n{st.session_state.df_processed.dtypes}")


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
            print(f"ERROR_PLOT_PRICE: {e}")
    else:
        st.warning("No se encontró la columna 'Price' en los datos procesados para graficar.")
        print("WARNING: No se encontró 'Price' para graficar.")

    # --- Sección de Análisis Técnico ---
    st.header("Análisis Técnico")

    expander_indicators = st.expander("Calcular Indicadores Técnicos", expanded=True) # Expandido por defecto ahora
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
                if st.session_state.df_processed is not None:
                    with st.spinner("Calculando indicadores..."):
                        df_temp_for_calc = st.session_state.df_processed.copy() # Trabajar con una copia
                        df_with_indicators = calculate_indicators(df_temp_for_calc, selected_indicators)
                        st.session_state.df_processed = df_with_indicators # Actualizar el estado
                        st.session_state.indicators_calculated = True
                        st.session_state.signals_generated = False 
                    st.success("¡Indicadores calculados!")
                    st.dataframe(st.session_state.df_processed.tail())
                else:
                    st.error("No hay datos procesados para calcular indicadores.")

    if st.session_state.indicators_calculated:
        expander_signals = st.expander("Generar Señales de Trading", expanded=False)
        with expander_signals:
            if st.button("2. Generar Señales", key="gen_sig_btn"):
                if st.session_state.df_processed is not None:
                    with st.spinner("Generando señales..."):
                        current_selected_indicators = st.session_state.get("indicator_selector", [])
                        
                        df_temp_for_signals = st.session_state.df_processed.copy()
                        df_with_signals, signals_summary = generate_signals(df_temp_for_signals, current_selected_indicators)
                        st.session_state.df_processed = df_with_signals
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
                else:
                    st.error("No hay datos con indicadores para generar señales.")


    expander_patterns = st.expander("Identificar Patrones de Velas", expanded=False)
    with expander_patterns:
        if st.button("Identificar Patrones de Velas Japonesas", key="identify_patterns_btn"):
            if st.session_state.df_processed is not None:
                # La verificación de columnas numéricas ahora está dentro de identify_patterns
                with st.spinner("Identificando patrones..."):
                    patterns = identify_patterns(st.session_state.df_processed.copy()) # Pasar una copia por si acaso
                st.success("¡Patrones identificados!")
                if patterns["bullish"] or patterns["bearish"]:
                    st.write("Patrones Alcistas Recientes:")
                    st.json(patterns["bullish"][-5:])
                    st.write("Patrones Bajistas Recientes:")
                    st.json(patterns["bearish"][-5:])
                else:
                    st.info("No se identificaron patrones con la lógica actual en los datos recientes.")
            else:
                st.warning("No hay datos procesados para identificar patrones.")


    if st.session_state.signals_generated:
        expander_backtest = st.expander("Realizar Backtesting Simple", expanded=False)
        with expander_backtest:
            initial_capital_bt = st.number_input("Capital Inicial para Backtesting:", min_value=100.0, value=10000.0, step=1000.0)
            if st.button("3. Ejecutar Backtesting", key="run_bt_btn"):
                if st.session_state.df_processed is not None:
                    with st.spinner("Ejecutando backtest..."):
                        backtest_results = backtesting_simple(st.session_state.df_processed.copy(), initial_capital_bt)
                    
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
                            trade_dates_valid = []
                            if not st.session_state.df_processed.empty: # Asegurarse que el df no esté vacío
                                trade_dates_valid.append(st.session_state.df_processed.index[0])
                            
                            trade_dates_valid.extend([pd.to_datetime(trade['datetime']) for trade in backtest_results['trade_history'] if 'capital_after_trade' in trade and trade.get('type','').lower() != 'buy' and 'datetime' in trade])
                            
                            if len(trade_dates_valid) == len(capital_over_time) and len(trade_dates_valid) > 1:
                                df_capital_plot = pd.DataFrame({'Timestamp': trade_dates_valid, 'Capital': capital_over_time}).set_index('Timestamp')
                                fig_capital = px.line(df_capital_plot, y="Capital", title="Evolución del Capital (Estimado Post-Venta)")
                                st.plotly_chart(fig_capital, use_container_width=True)
                            elif backtest_results['trades'] > 0 : # Si hubo trades pero no se pudo graficar
                                 st.info("No se pudo graficar la evolución del capital (datos insuficientes o inconsistentes para el gráfico).")
                        else:
                            st.info("No se realizaron operaciones en el backtest.")
                else:
                    st.error("No hay datos con señales para ejecutar el backtesting.")
else:
    print("DEBUG: df_processed es None, mostrando mensaje para cargar datos.")
    st.info("⬅️ Por favor, carga un archivo CSV o descarga datos usando la barra lateral para comenzar.")

st.sidebar.markdown("---")
st.sidebar.markdown("App mejorada con la ayuda de IA.")
print("DEBUG: Fin del script.")
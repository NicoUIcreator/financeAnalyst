import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.express as px
import os
import yfinance as yf


st.set_page_config(page_title="Financial Analyst App", layout="wide")

st.title("Financial Analyst App")
st.write("Analyze stock data and generate insights using AI agents.")

# Sección de carga de datos
st.header("Welcome to the Financial Analyst App")
st.write("Please upload a CSV file containing historical price data or select a stock symbol.")

# Opción para cargar un archivo CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

file_name = uploaded_file.name 

stock_symbol = file_name

df = uploaded_file

df["Date"] = pd.to_datetime(df["Date"])
    
    # Establecer la columna de fecha como índice
df.set_index("Date",inplace=True)

# Importar funciones definidas
from typing import List, Dict, Tuple, Union


def calculate_indicators(df: pd.DataFrame, selected_indicators: List[str]) -> pd.DataFrame:
    """
    Calcula indicadores técnicos seleccionados para un DataFrame.
    Args:
        df (pd.DataFrame): DataFrame con datos de precios
        selected_indicators (List[str]): Lista de indicadores a calcular
    Returns:
        pd.DataFrame: DataFrame con indicadores calculados
    """
    result = df.copy()
    required_columns = ["Open", "High", "Low", "Price", "Volume"]
    for col in required_columns:
        if col not in result.columns:
            st.warning(f"Columna {col} no encontrada. Algunos indicadores pueden no calcularse correctamente.")
    for indicator in selected_indicators:
        if indicator == "MACD":
            macd = ta.macd(result["Price"])
            result = pd.concat([result, macd], axis=1)
        elif indicator == "RSI":
            result["RSI"] = ta.rsi(result["Price"], length=14)
        elif indicator == "Bollinger Bands":
            bbands = ta.bbands(result["Price"], length=20)
            result = pd.concat([result, bbands], axis=1)
        elif indicator == "Stochastic":
            stoch = ta.stoch(result["High"], result["Low"], result["Price"])
            result = pd.concat([result, stoch], axis=1)
        elif indicator == "Moving Averages":
            result["SMA_20"] = ta.sma(result["Price"], length=20)
            result["SMA_50"] = ta.sma(result["Price"], length=50)
            result["SMA_200"] = ta.sma(result["Price"], length=200)
            result["EMA_12"] = ta.ema(result["Price"], length=12)
            result["EMA_26"] = ta.ema(result["Price"], length=26)
    return result

def identify_patterns(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Identifica patrones de velas japonesas.
    Args:
        df (pd.DataFrame): DataFrame con datos OHLC
    Returns:
        Dict[str, List[Dict]]: Diccionario con patrones identificados
    """
    patterns = {
        "bullish": [],
        "bearish": []
    }
    if len(df) < 3:
        return patterns
    for i in range(1, len(df) - 1):
        prev_candle = {
            "open": df["Open"].iloc[i-1],
            "high": df["High"].iloc[i-1],
            "low": df["Low"].iloc[i-1],
            "close": df["Price"].iloc[i-1]
        }
        current_candle = {
            "open": df["Open"].iloc[i],
            "high": df["High"].iloc[i],
            "low": df["Low"].iloc[i],
            "close": df["Price"].iloc[i],
            "datetime": df["Datetime"].iloc[i]
        }
        next_candle = {
            "open": df["Open"].iloc[i+1],
            "high": df["High"].iloc[i+1],
            "low": df["Low"].iloc[i+1],
            "close": df["Price"].iloc[i+1]
        }

        # Patrón de martillo (bullish)
        if (current_candle["close"] > current_candle["open"] and
            (current_candle["high"] - current_candle["close"]) < (current_candle["close"] - current_candle["open"]) * 0.1 and
            (current_candle["open"] - current_candle["low"]) > (current_candle["close"] - current_candle["open"]) * 2 and
            prev_candle["close"] < prev_candle["open"]):
            patterns["bullish"].append({
                "pattern": "Hammer",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de estrella fugaz (bearish)
        if (current_candle["close"] < current_candle["open"] and
            (current_candle["high"] - current_candle["open"]) > (current_candle["open"] - current_candle["close"]) * 2 and
            (current_candle["close"] - current_candle["low"]) < (current_candle["open"] - current_candle["close"]) * 0.1 and
            prev_candle["close"] > prev_candle["open"]):
            patterns["bearish"].append({
                "pattern": "Shooting Star",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de envolvente alcista (bullish engulfing)
        if (current_candle["close"] > current_candle["open"] and
            prev_candle["close"] < prev_candle["open"] and
            current_candle["close"] > prev_candle["open"] and
            current_candle["open"] < prev_candle["close"]):
            patterns["bullish"].append({
                "pattern": "Bullish Engulfing",
                "datetime": current_candle["datetime"],
                "price": current_candle["close"]
            })

        # Patrón de envolvente bajista (bearish engulfing)
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

def backtesting_simple(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """
    Realiza un backtest simple basado en las señales generadas.
    Args:
        df (pd.DataFrame): DataFrame con señales calculadas
        initial_capital (float): Capital inicial para el backtest
    Returns:
        Dict: Resultados del backtest
    """
    if "Signal_Buy" not in df.columns or "Signal_Sell" not in df.columns:
        return {
            "error": "No se encontraron señales para el backtest",
            "final_capital": initial_capital,
            "return_pct": 0.0,
            "trades": 0
        }
    backtest = df.copy()
    capital = initial_capital
    position = 0  # 0: sin posición, 1: comprado
    entry_price = 0.0
    trades = []

    for i in range(1, len(backtest)):
        if position == 0 and backtest["Signal_Buy"].iloc[i] >= 2:
            entry_price = backtest["Price"].iloc[i]
            position = 1
            shares = capital / entry_price
            trades.append({
                "type": "buy",
                "datetime": backtest["Datetime"].iloc[i],
                "price": entry_price,
                "capital": capital,
                "shares": shares
            })
        elif position == 1 and backtest["Signal_Sell"].iloc[i] >= 2:
            exit_price = backtest["Price"].iloc[i]
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

    if position == 1:
        exit_price = backtest["Price"].iloc[-1]
        shares = capital / entry_price
        capital = shares * exit_price
        trades.append({
            "type": "sell (final)",
            "datetime": backtest["Datetime"].iloc[-1],
            "price": exit_price,
            "capital": capital,
            "pnl": ((exit_price / entry_price) - 1) * 100
        })

    return_pct = ((capital / initial_capital) - 1) * 100
    n_trades = len([t for t in trades if t["type"] == "buy"])
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

def generate_signals(df: pd.DataFrame, selected_indicators: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Genera señales de trading basadas en indicadores.
    Args:
        df (pd.DataFrame): DataFrame con indicadores calculados
        selected_indicators (List[str]): Lista de indicadores utilizados
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con señales y diccionario de resumen
    """
    result = df.copy()
    signals = {
        "buy": [],
        "sell": [],
        "summary": {},
        "strength": 0
    }
    result["Signal_Buy"] = 0
    result["Signal_Sell"] = 0
    result["Signal_Strength"] = 0
    signal_count = 0
    signal_value = 0

    if "MACD" in selected_indicators and "MACD" in result.columns and "MACD_signal" in result.columns:
        signal_count += 1
        result.loc[(result["MACD"] > result["MACD_signal"]) & 
                  (result["MACD"].shift(1) <= result["MACD_signal"].shift(1)), "Signal_Buy"] += 1
        result.loc[(result["MACD"] < result["MACD_signal"]) & 
                  (result["MACD"].shift(1) >= result["MACD_signal"].shift(1)), "Signal_Sell"] += 1
        if result["MACD"].iloc[-1] > result["MACD_signal"].iloc[-1]:
            signals["summary"]["MACD"] = "Compra"
            signal_value += 1
        else:
            signals["summary"]["MACD"] = "Venta"
            signal_value -= 1

    if "RSI" in selected_indicators and "RSI" in result.columns:
        signal_count += 1
        result.loc[result["RSI"] < 30, "Signal_Buy"] += 1
        result.loc[result["RSI"] > 70, "Signal_Sell"] += 1
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

    if "Bollinger Bands" in selected_indicators and "BBL_20_2.0" in result.columns and "BBU_20_2.0" in result.columns:
        signal_count += 1
        result.loc[result["Price"] <= result["BBL_20_2.0"], "Signal_Buy"] += 1
        result.loc[result["Price"] >= result["BBU_20_2.0"], "Signal_Sell"] += 1
        if result["Price"].iloc[-1] <= result["BBL_20_2.0"].iloc[-1]:
            signals["summary"]["Bollinger"] = "Compra (Banda Inferior)"
            signal_value += 1
        elif result["Price"].iloc[-1] >= result["BBU_20_2.0"].iloc[-1]:
            signals["summary"]["Bollinger"] = "Venta (Banda Superior)"
            signal_value -= 1
        else:
            signals["summary"]["Bollinger"] = "Neutral (Dentro de Bandas)"

    if "Stochastic" in selected_indicators and "STOCHk_14_3_3" in result.columns and "STOCHd_14_3_3" in result.columns:
        signal_count += 1
        result.loc[(result["STOCHk_14_3_3"] > result["STOCHd_14_3_3"]) & 
                  (result["STOCHk_14_3_3"].shift(1) <= result["STOCHd_14_3_3"].shift(1)) & 
                  (result["STOCHk_14_3_3"] < 20), "Signal_Buy"] += 1
        result.loc[(result["STOCHk_14_3_3"] < result["STOCHd_14_3_3"]) & 
                  (result["STOCHk_14_3_3"].shift(1) >= result["STOCHd_14_3_3"].shift(1)) & 
                  (result["STOCHk_14_3_3"] > 80), "Signal_Sell"] += 1
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

    if "Moving Averages" in selected_indicators:
        has_ma = False
        if "SMA_20" in result.columns and "SMA_50" in result.columns:
            signal_count += 1
            has_ma = True
            result.loc[(result["SMA_20"] > result["SMA_50"]) & 
                      (result["SMA_20"].shift(1) <= result["SMA_50"].shift(1)), "Signal_Buy"] += 1
            result.loc[(result["SMA_20"] < result["SMA_50"]) & 
                      (result["SMA_20"].shift(1) >= result["SMA_50"].shift(1)), "Signal_Sell"] += 1
            if result["Price"].iloc[-1] > result["SMA_20"].iloc[-1]:
                result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5
            else:
                result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
            if result["Price"].iloc[-1] > result["SMA_50"].iloc[-1]:
                result.iloc[-1, result.columns.get_loc("Signal_Buy")] += 0.5
            else:
                result.iloc[-1, result.columns.get_loc("Signal_Sell")] += 0.5
        if has_ma:
            ma_signal = ""
            ma_value = 0
            if result["SMA_20"].iloc[-1] > result["SMA_50"].iloc[-1]:
                ma_signal += "SMA 20 por encima de SMA 50 (Alcista). "
                ma_value += 1
            else:
                ma_signal += "SMA 20 por debajo de SMA 50 (Bajista). "
                ma_value -= 1
            if "SMA_200" in result.columns:
                if result["Price"].iloc[-1] > result["SMA_200"].iloc[-1]:
                    ma_signal += "Precio por encima de SMA 200 (Alcista a largo plazo)."
                    ma_value += 1
                else:
                    ma_signal += "Precio por debajo de SMA 200 (Bajista a largo plazo)."
                    ma_value -= 1
            signals["summary"]["Moving Averages"] = ma_signal
            signal_value += ma_value

    if signal_count > 0:
        signals["strength"] = int((signal_value / (signal_count * 2)) * 100)

    for i in range(len(result)):
        if result["Signal_Buy"].iloc[i] >= 2:
            signals["buy"].append({
                "index": i,
                "datetime": result["Datetime"].iloc[i],
                "price": result["Price"].iloc[i],
                "strength": result["Signal_Buy"].iloc[i]
            })
        if result["Signal_Sell"].iloc[i] >= 2:
            signals["sell"].append({
                "index": i,
                "datetime": result["Datetime"].iloc[i],
                "price": result["Price"].iloc[i],
                "strength": result["Signal_Sell"].iloc[i]
            })

    result["Signal_Strength"] = result["Signal_Buy"] - result["Signal_Sell"]
    return result, signals

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    # Opción para seleccionar un símbolo de acción
    if 'df' in locals() and 'stock_symbol' in locals():
        st.subheader("Interactive Chart of Closing Prices")

        fig = px.line(df, x=df.index, y='Price', title=f"{stock_symbol} Closing Price")
        st.plotly_chart(fig)
    else:
        st.warning("Please upload a file or enter a stock symbol first.")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, NVDA, BTCUSD)", "AAPL")
    if st.button("Fetch Data"):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        st.success(f"Data fetched for {stock_symbol}!")

# Función para limpiar los datos
def clean_data(df):
    if df.isnull().values.any():
        st.warning("The dataset contains null values. Filling them with forward fill method.")
        df = df.fillna(method='ffill')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

if 'df' in locals():
    df = clean_data(df)
    st.success("Data cleaned successfully!")
    st.dataframe(df.head())

    # Visualizar gráficos interactivos
    st.subheader("Interactive Chart of Closing Prices")
    fig = px.line(df, x=df.index, y='Price', title=f"{stock_symbol} Closing Price")
    st.plotly_chart(fig)

    # Seleccionar indicadores técnicos
    st.subheader("Select Indicators to Analyze")
    selected_indicators = st.multiselect(
        "Choose indicators",
        ["MACD", "RSI", "Bollinger Bands", "Stochastic", "Moving Averages"],
        default=["MACD", "RSI", "Bollinger Bands"]
    )

    if st.button("Calculate Indicators"):
        df = calculate_indicators(df, selected_indicators)
        st.success("Indicators calculated successfully!")
        st.dataframe(df.head())

    if st.button("Generate Signals"):
        df, signals = generate_signals(df, selected_indicators)
        st.success("Signals generated successfully!")
        st.json(signals)

    if st.button("Backtest Strategy"):
        
        results = backtesting_simple(df)
        st.success("Backtest completed!")
        st.write(f"Final Capital: ${results['final_capital']:.2f}")
        st.write(f"Return Percentage: {results['return_pct']:.2f}%")
        st.write(f"Number of Trades: {results['trades']}")
        st.write(f"Average PnL: {results['avg_pnl']:.2f}%")
        st.write(f"Win Rate: {results['win_rate']:.2f}%")

    if st.button("Identify Candlestick Patterns"):
        patterns = identify_patterns(df)
        st.success("Candlestick patterns identified!")
        st.json(patterns)

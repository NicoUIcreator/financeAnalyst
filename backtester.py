import vectorbt as vbt

def backtest(df):
    """Simula una estrategia de trading usando VectorBT."""
    # Estrategia: Comprar cuando RSI < 30, vender cuando RSI > 70
    strategy = vbt.IndicatorFactory.from_talib('RSI').run(df['Close'], timeperiod=14)
    entries = strategy.rsi < 30
    exits = strategy.rsi > 70
    pf = vbt.Portfolio.from_signals(df['Close'], entries, exits)
    return pf
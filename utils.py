import pandas as pd
import numpy as np

def format_number(number, precision=2):
    """
    Formatea un número para su visualización.
    
    Args:
        number (float): Número a formatear
        precision (int): Cantidad de decimales a mostrar
        
    Returns:
        str: Número formateado
    """
    if number is None:
        return "N/A"
    
    # Para números grandes, usar K, M, B
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{precision}f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{precision}f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def calculate_returns(price_series, period=1):
    """
    Calcula los retornos de una serie de precios para un periodo determinado.
    
    Args:
        price_series (pandas.Series): Serie de precios
        period (int): Periodo para calcular los retornos (1 = diario, etc.)
        
    Returns:
        pandas.Series: Serie con los retornos calculados
    """
    # Calculamos los retornos porcentuales
    returns = price_series.pct_change(periods=period)
    
    return returns
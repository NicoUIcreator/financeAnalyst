import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_price_chart(df, ticker):
    """
    Crea un gráfico interactivo de precios y volumen con Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos de precios
        ticker (str): Símbolo del activo
        
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly con el gráfico de precios y volumen
    """
    # Crear una figura con dos subplots (precio y volumen)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f'Precio de {ticker}', 'Volumen'))
    
    # Añadir gráfico de velas para los precios
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Precio'
        ),
        row=1, col=1
    )
    
    # Añadir gráfico de barras para el volumen
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volumen',
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Actualizar diseño del gráfico
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02),
        template='plotly_white'
    )
    
    # Personalizar eje Y
    fig.update_yaxes(title_text='Precio', row=1, col=1)
    fig.update_yaxes(title_text='Volumen', row=2, col=1)
    
    return fig

def create_indicator_charts(df, selected_indicators):
    """
    Crea gráficos para los indicadores técnicos seleccionados.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos de precios e indicadores
        selected_indicators (list): Lista de indicadores seleccionados
        
    Returns:
        list: Lista de figuras de Plotly con los gráficos de indicadores
    """
    charts = []
    
    # Procesar cada indicador seleccionado
    for indicator in selected_indicators:
        if indicator == "RSI":
            # Crear gráfico RSI
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            
            # Líneas de sobrecompra/sobreventa
            fig.add_shape(
                type="line", line_color="red", line_width=2, opacity=0.5,
                x0=df.index[0], x1=df.index[-1], y0=70, y1=70
            )
            fig.add_shape(
                type="line", line_color="green", line_width=2, opacity=0.5,
                x0=df.index[0], x1=df.index[-1], y0=30, y1=30
            )
            
            fig.update_layout(
                title="RSI (Relative Strength Index)",
                xaxis_title="Fecha",
                yaxis_title="RSI",
                template="plotly_white",
                height=300
            )
            charts.append(fig)
            
        elif indicator == "MACD":
            # Crear gráfico MACD
            fig = make_subplots(rows=1, cols=1)
            
            # Línea MACD
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['MACD'], 
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                )
            )
            
            # Línea de señal
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['MACD_signal'], 
                    name='Señal',
                    line=dict(color='red', width=1.5)
                )
            )
            
            # Histograma (diferencia)
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD'] - df['MACD_signal'],
                    name='Histograma',
                    marker_color=np.where(df['MACD'] >= df['MACD_signal'], 'rgba(0,255,0,0.5)', 'rgba(255,0,0,0.5)')
                )
            )
            
            fig.update_layout(
                title="MACD (Moving Average Convergence Divergence)",
                xaxis_title="Fecha",
                yaxis_title="MACD",
                template="plotly_white",
                height=300
            )
            charts.append(fig)
            
        elif indicator == "Bollinger Bands":
            # Crear gráfico Bollinger Bands
            fig = go.Figure()
            
            # Línea del precio
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Close'], 
                    name='Precio',
                    line=dict(color='blue', width=1)
                )
            )
            
            # Banda superior
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['BB_upper'], 
                    name='Banda Superior',
                    line=dict(color='rgba(0,255,0,0.3)', width=1),
                    fill=None
                )
            )
            
            # Banda inferior (con área sombreada)
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['BB_lower'], 
                    name='Banda Inferior',
                    line=dict(color='rgba(0,255,0,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.1)'
                )
            )
            
            # Media móvil
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['BB_middle'], 
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                )
            )
            
            fig.update_layout(
                title="Bandas de Bollinger",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                template="plotly_white",
                height=300
            )
            charts.append(fig)
            
        elif indicator == "Stochastic":
            # Crear gráfico Stochastic
            fig = go.Figure()
            
            # Línea %K
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Stoch_K'], 
                    name='%K',
                    line=dict(color='blue', width=1.5)
                )
            )
            
            # Línea %D
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Stoch_D'], 
                    name='%D',
                    line=dict(color='red', width=1.5)
                )
            )
            
            # Líneas de sobrecompra/sobreventa
            fig.add_shape(
                type="line", line_color="red", line_width=2, opacity=0.5,
                x0=df.index[0], x1=df.index[-1], y0=80, y1=80
            )
            fig.add_shape(
                type="line", line_color="green", line_width=2, opacity=0.5,
                x0=df.index[0], x1=df.index[-1], y0=20, y1=20
            )
            
            fig.update_layout(
                title="Oscillador Estocástico",
                xaxis_title="Fecha",
                yaxis_title="Valor",
                template="plotly_white",
                height=300
            )
            charts.append(fig)
            
        elif indicator == "Moving Averages":
            # Crear gráfico de Medias Móviles
            fig = go.Figure()
            
            # Precio
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Close'], 
                    name='Precio',
                    line=dict(color='black', width=1)
                )
            )
            
            # SMA 20
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['SMA_20'], 
                    name='SMA 20',
                    line=dict(color='blue', width=1.5)
                )
            )
            
            # SMA 50
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['SMA_50'], 
                    name='SMA 50',
                    line=dict(color='orange', width=1.5)
                )
            )
            
            # SMA 200
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['SMA_200'], 
                    name='SMA 200',
                    line=dict(color='red', width=1.5)
                )
            )
            
            fig.update_layout(
                title="Medias Móviles",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                template="plotly_white",
                height=300
            )
            charts.append(fig)
    
    return charts

def create_signal_dashboard(signals, prediction_results):
    """
    Crea un panel HTML con las señales de trading y predicciones.
    
    Args:
        signals (dict): Diccionario con las señales de trading
        prediction_results (dict): Resultados de la predicción del modelo ML
        
    Returns:
        str: HTML formateado para el dashboard de señales
    """
    # Obtener color según la señal
    def get_signal_color(signal):
        if signal.lower() == "compra":
            return "positive"
        elif signal.lower() == "venta":
            return "negative"
        else:
            return ""
    
    # Construir HTML para las señales
    html = """
    <div class="card">
        <h4>Señales de Trading</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Indicador</th>
                <th style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">Señal</th>
                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Fuerza</th>
            </tr>
    """
    
    # Añadir filas para cada señal
    for indicator, signal_info in signals.items():
        signal = signal_info.get('signal', 'Neutral')
        strength = signal_info.get('strength', 0)
        
        html += f"""
        <tr>
            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{indicator}</td>
            <td style="text-align: center; padding: 8px; border-bottom: 1px solid #ddd;">
                <span class="{get_signal_color(signal)}">{signal}</span>
            </td>
            <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">
                {'▲' * int(abs(strength) / 20) if strength > 0 else '▼' * int(abs(strength) / 20)}
            </td>
        </tr>
        """
    
    # Añadir resumen del consenso de señales
    buy_signals = sum(1 for info in signals.values() if info.get('signal', '').lower() == 'compra')
    sell_signals = sum(1 for info in signals.values() if info.get('signal', '').lower() == 'venta')
    neutral_signals = len(signals) - buy_signals - sell_signals
    
    # Determinar señal de consenso
    if buy_signals > sell_signals and buy_signals > neutral_signals:
        consensus = "COMPRA"
        consensus_class = "positive"
    elif sell_signals > buy_signals and sell_signals > neutral_signals:
        consensus = "VENTA"
        consensus_class = "negative"
    else:
        consensus = "NEUTRAL"
        consensus_class = ""
    
    # Añadir consenso al HTML
    html += f"""
        </table>
        <div style="margin-top: 15px; padding: 10px; background-color: #f2f2f2; border-radius: 5px;">
            <span style="font-weight: bold;">Consenso: </span>
            <span class="{consensus_class}" style="font-weight: bold;">{consensus}</span>
            <div style="margin-top: 5px;">
                <span style="color: #4CAF50;">Compra: {buy_signals}</span> | 
                <span>Neutral: {neutral_signals}</span> | 
                <span style="color: #F44336;">Venta: {sell_signals}</span>
            </div>
        </div>
    </div>
    """
    
    return html
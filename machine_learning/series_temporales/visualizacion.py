# visualizacion.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta


class FinancialCharts:
    """
    Clase para generar visualizaciones financieras interactivas usando Plotly
    """
    
    def __init__(self, data=None, symbol=None):
        """
        Inicializa la clase con datos financieros
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV
            symbol (str): Símbolo de la acción
        """
        self.data = data
        self.symbol = symbol
        
    def set_data(self, data, symbol=None):
        """
        Establece los datos para visualización
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV
            symbol (str): Símbolo de la acción
        """
        self.data = data
        if symbol:
            self.symbol = symbol
    
    def candlestick_chart(self, title=None, show_volume=True):
        """
        Genera un gráfico de velas japonesas con volumen
        
        Args:
            title (str): Título del gráfico
            show_volume (bool): Si mostrar el volumen
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        # Crear subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{self.symbol} - Precio', 'Volumen'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Gráfico de velas
        candlestick = go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name=f'{self.symbol}',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Gráfico de volumen
        if show_volume:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(self.data['close'], self.data['open'])]
            
            volume_bar = go.Bar(
                x=self.data.index,
                y=self.data['volume'],
                name='Volumen',
                marker_color=colors,
                opacity=0.7
            )
            fig.add_trace(volume_bar, row=2, col=1)
        
        # Configuración del layout
        chart_title = title or f'Gráfico de Velas - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            xaxis_title='Fecha',
            yaxis_title='Precio ($)',
            template='plotly_white',
            height=600 if show_volume else 500,
            showlegend=True
        )
        
        # Configurar ejes
        if show_volume:
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_yaxes(title_text="Volumen", row=2, col=1)
        
        return fig
    
    def line_chart(self, column='close', title=None, add_ma=True, ma_periods=[20, 50]):
        """
        Genera un gráfico de líneas con medias móviles opcionales
        
        Args:
            column (str): Columna a graficar ('close', 'open', 'high', 'low')
            title (str): Título del gráfico
            add_ma (bool): Si agregar medias móviles
            ma_periods (list): Períodos para las medias móviles
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        fig = go.Figure()
        
        # Línea principal
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data[column],
            mode='lines',
            name=f'{self.symbol} - {column.upper()}',
            line=dict(color='blue', width=2)
        ))
        
        # Medias móviles
        if add_ma:
            for period in ma_periods:
                ma = self.data[column].rolling(window=period).mean()
                fig.add_trace(go.Scatter(
                    x=self.data.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(dash='dash', width=1)
                ))
        
        # Configuración del layout
        chart_title = title or f'Gráfico de Líneas - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            xaxis_title='Fecha',
            yaxis_title='Precio ($)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def bollinger_bands_chart(self, period=20, std_dev=2, title=None):
        """
        Genera un gráfico con Bandas de Bollinger
        
        Args:
            period (int): Período para la media móvil
            std_dev (float): Desviación estándar para las bandas
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        # Calcular Bandas de Bollinger
        close = self.data['close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        fig = go.Figure()
        
        # Bandas de Bollinger
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=upper_band,
            mode='lines',
            name='Banda Superior',
            line=dict(color='red', dash='dash'),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=lower_band,
            mode='lines',
            name='Banda Inferior',
            line=dict(color='red', dash='dash'),
            opacity=0.7,
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Precio de cierre
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=close,
            mode='lines',
            name=f'{self.symbol} - Close',
            line=dict(color='blue', width=2)
        ))
        
        # Media móvil
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=sma,
            mode='lines',
            name=f'SMA{period}',
            line=dict(color='orange', width=1)
        ))
        
        # Configuración del layout
        chart_title = title or f'Bandas de Bollinger - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            xaxis_title='Fecha',
            yaxis_title='Precio ($)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def rsi_chart(self, period=14, title=None):
        """
        Genera un gráfico del RSI (Relative Strength Index)
        
        Args:
            period (int): Período para el cálculo del RSI
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        # Calcular RSI
        close = self.data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig = go.Figure()
        
        # RSI
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=rsi,
            mode='lines',
            name=f'RSI({period})',
            line=dict(color='purple', width=2)
        ))
        
        # Líneas de referencia
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Sobrecompra (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Sobreventa (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                     annotation_text="Neutral (50)")
        
        # Configuración del layout
        chart_title = title or f'RSI - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            xaxis_title='Fecha',
            yaxis_title='RSI',
            template='plotly_white',
            height=400,
            yaxis=dict(range=[0, 100]),
            showlegend=True
        )
        
        return fig
    
    def macd_chart(self, fast=12, slow=26, signal=9, title=None):
        """
        Genera un gráfico del MACD (Moving Average Convergence Divergence)
        
        Args:
            fast (int): Período rápido
            slow (int): Período lento
            signal (int): Período de la señal
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        # Calcular MACD
        close = self.data['close']
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{self.symbol} - Precio', f'MACD({fast},{slow},{signal})'),
            row_heights=[0.7, 0.3]
        )
        
        # Precio
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=close,
            mode='lines',
            name=f'{self.symbol}',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=macd_line,
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=signal_line,
            mode='lines',
            name='Señal',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        # Histograma
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=histogram,
            name='Histograma',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        # Línea cero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Configuración del layout
        chart_title = title or f'MACD - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        return fig
    
    def comprehensive_analysis(self, title=None):
        """
        Genera un análisis técnico completo con múltiples indicadores
        
        Args:
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{self.symbol} - Precio y Bandas de Bollinger',
                'RSI (14)',
                'MACD',
                'Volumen'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Precio con Bandas de Bollinger
        close = self.data['close']
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        
        # Velas
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name=f'{self.symbol}',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Bandas de Bollinger
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=upper_band,
            mode='lines',
            name='BB Superior',
            line=dict(color='red', dash='dash'),
            opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=lower_band,
            mode='lines',
            name='BB Inferior',
            line=dict(color='red', dash='dash'),
            opacity=0.7,
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ), row=1, col=1)
        
        # 2. RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=rsi,
            mode='lines',
            name='RSI(14)',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=macd_line,
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=signal_line,
            mode='lines',
            name='Señal',
            line=dict(color='red', width=2)
        ), row=3, col=1)
        
        # 4. Volumen
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['volume'],
            name='Volumen',
            marker_color=colors,
            opacity=0.7
        ), row=4, col=1)
        
        # Configuración del layout
        chart_title = title or f'Análisis Técnico Completo - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        # Configurar ejes
        fig.update_xaxes(title_text="Fecha", row=4, col=1)
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Volumen", row=4, col=1)
        
        return fig
    
    def show_chart(self, chart_type='candlestick', **kwargs):
        """
        Método de conveniencia para mostrar diferentes tipos de gráficos
        
        Args:
            chart_type (str): Tipo de gráfico ('candlestick', 'line', 'bollinger', 'rsi', 'macd', 'comprehensive')
            **kwargs: Argumentos adicionales para el gráfico específico
        """
        chart_methods = {
            'candlestick': self.candlestick_chart,
            'line': self.line_chart,
            'bollinger': self.bollinger_bands_chart,
            'rsi': self.rsi_chart,
            'macd': self.macd_chart,
            'comprehensive': self.comprehensive_analysis
        }
        
        if chart_type not in chart_methods:
            print(f"❌ Tipo de gráfico '{chart_type}' no soportado")
            print(f"Tipos disponibles: {list(chart_methods.keys())}")
            return None
        
        fig = chart_methods[chart_type](**kwargs)
        if fig:
            fig.show()
        return fig

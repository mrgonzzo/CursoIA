# indicadores_tecnicos.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


class TechnicalIndicators:
    """
    Clase para calcular indicadores técnicos financieros
    """
    
    def __init__(self, data):
        """
        Inicializa con datos financieros
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV
        """
        self.data = data.copy()
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.volume = data['volume']
    
    def sma(self, period=20):
        """
        Simple Moving Average (Media Móvil Simple)
        
        Args:
            period (int): Período para la media móvil
            
        Returns:
            pd.Series: Serie con la media móvil
        """
        return self.close.rolling(window=period).mean()
    
    def ema(self, period=20):
        """
        Exponential Moving Average (Media Móvil Exponencial)
        
        Args:
            period (int): Período para la media móvil
            
        Returns:
            pd.Series: Serie con la media móvil exponencial
        """
        return self.close.ewm(span=period).mean()
    
    def rsi(self, period=14):
        """
        Relative Strength Index (Índice de Fuerza Relativa)
        
        Args:
            period (int): Período para el cálculo del RSI
            
        Returns:
            pd.Series: Serie con los valores del RSI
        """
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            fast (int): Período rápido
            slow (int): Período lento
            signal (int): Período de la señal
            
        Returns:
            pd.DataFrame: DataFrame con MACD, señal e histograma
        """
        ema_fast = self.close.ewm(span=fast).mean()
        ema_slow = self.close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def bollinger_bands(self, period=20, std_dev=2):
        """
        Bollinger Bands (Bandas de Bollinger)
        
        Args:
            period (int): Período para la media móvil
            std_dev (float): Desviación estándar para las bandas
            
        Returns:
            pd.DataFrame: DataFrame con bandas superior, media e inferior
        """
        sma = self.close.rolling(window=period).mean()
        std = self.close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        percent_b = (self.close - lower_band) / (upper_band - lower_band)
        
        return pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band,
            'percent_b': percent_b
        })
    
    def stochastic(self, k_period=14, d_period=3):
        """
        Stochastic Oscillator
        
        Args:
            k_period (int): Período para %K
            d_period (int): Período para %D
            
        Returns:
            pd.DataFrame: DataFrame con %K y %D
        """
        lowest_low = self.low.rolling(window=k_period).min()
        highest_high = self.high.rolling(window=k_period).max()
        
        k_percent = 100 * ((self.close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def williams_r(self, period=14):
        """
        Williams %R
        
        Args:
            period (int): Período para el cálculo
            
        Returns:
            pd.Series: Serie con Williams %R
        """
        highest_high = self.high.rolling(window=period).max()
        lowest_low = self.low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - self.close) / (highest_high - lowest_low))
        return williams_r
    
    def atr(self, period=14):
        """
        Average True Range (Rango Verdadero Promedio)
        
        Args:
            period (int): Período para el cálculo
            
        Returns:
            pd.Series: Serie con ATR
        """
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift())
        low_close = np.abs(self.low - self.close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def adx(self, period=14):
        """
        Average Directional Index
        
        Args:
            period (int): Período para el cálculo
            
        Returns:
            pd.DataFrame: DataFrame con ADX, +DI y -DI
        """
        # Calcular True Range
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift())
        low_close = np.abs(self.low - self.close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calcular +DM y -DM
        plus_dm = self.high.diff()
        minus_dm = -self.low.diff()
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        # Suavizar
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        # Calcular ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    def obv(self):
        """
        On-Balance Volume
        
        Returns:
            pd.Series: Serie con OBV
        """
        obv = np.where(self.close > self.close.shift(), self.volume,
                      np.where(self.close < self.close.shift(), -self.volume, 0))
        return pd.Series(obv).cumsum()
    
    def vwap(self):
        """
        Volume Weighted Average Price
        
        Returns:
            pd.Series: Serie con VWAP
        """
        typical_price = (self.high + self.low + self.close) / 3
        vwap = (typical_price * self.volume).cumsum() / self.volume.cumsum()
        return vwap
    
    def ichimoku(self, conversion_period=9, base_period=26, leading_span_b_period=52, displacement=26):
        """
        Ichimoku Cloud
        
        Args:
            conversion_period (int): Período para Tenkan-sen
            base_period (int): Período para Kijun-sen
            leading_span_b_period (int): Período para Senkou Span B
            displacement (int): Desplazamiento para Senkou Span
            
        Returns:
            pd.DataFrame: DataFrame con componentes de Ichimoku
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (self.high.rolling(window=conversion_period).max() + 
                     self.low.rolling(window=conversion_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (self.high.rolling(window=base_period).max() + 
                    self.low.rolling(window=base_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((self.high.rolling(window=leading_span_b_period).max() + 
                         self.low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = self.close.shift(-displacement)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })


class TechnicalAnalysisVisualizer:
    """
    Clase para visualizar análisis técnico con múltiples indicadores
    """
    
    def __init__(self, data, symbol=None):
        """
        Inicializa con datos financieros
        
        Args:
            data (pd.DataFrame): DataFrame con datos OHLCV
            symbol (str): Símbolo de la acción
        """
        self.data = data
        self.symbol = symbol
        self.indicators = TechnicalIndicators(data)
    
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
        bb_data = self.indicators.bollinger_bands()
        
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
            y=bb_data['upper_band'],
            mode='lines',
            name='BB Superior',
            line=dict(color='red', dash='dash'),
            opacity=0.7
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=bb_data['lower_band'],
            mode='lines',
            name='BB Inferior',
            line=dict(color='red', dash='dash'),
            opacity=0.7,
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ), row=1, col=1)
        
        # Media móvil
        sma20 = self.indicators.sma(20)
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=sma20,
            mode='lines',
            name='SMA20',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        # 2. RSI
        rsi_data = self.indicators.rsi()
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=rsi_data,
            mode='lines',
            name='RSI(14)',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # 3. MACD
        macd_data = self.indicators.macd()
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=macd_data['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=macd_data['signal'],
            mode='lines',
            name='Señal',
            line=dict(color='red', width=2)
        ), row=3, col=1)
        
        # Histograma MACD
        colors = ['green' if val >= 0 else 'red' for val in macd_data['histogram']]
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=macd_data['histogram'],
            name='Histograma MACD',
            marker_color=colors,
            opacity=0.7
        ), row=3, col=1)
        
        # Línea cero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
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
    
    def price_with_indicators(self, indicators=['sma', 'ema', 'bb'], periods=[20, 50], title=None):
        """
        Gráfico de precio con indicadores seleccionados
        
        Args:
            indicators (list): Lista de indicadores a mostrar
            periods (list): Períodos para las medias móviles
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        fig = go.Figure()
        
        # Precio de cierre
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['close'],
            mode='lines',
            name=f'{self.symbol} - Close',
            line=dict(color='blue', width=2)
        ))
        
        # Indicadores seleccionados
        if 'sma' in indicators:
            for period in periods:
                sma = self.indicators.sma(period)
                fig.add_trace(go.Scatter(
                    x=self.data.index,
                    y=sma,
                    mode='lines',
                    name=f'SMA{period}',
                    line=dict(dash='dash', width=1)
                ))
        
        if 'ema' in indicators:
            for period in periods:
                ema = self.indicators.ema(period)
                fig.add_trace(go.Scatter(
                    x=self.data.index,
                    y=ema,
                    mode='lines',
                    name=f'EMA{period}',
                    line=dict(dash='dot', width=1)
                ))
        
        if 'bb' in indicators:
            bb_data = self.indicators.bollinger_bands()
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=bb_data['upper_band'],
                mode='lines',
                name='BB Superior',
                line=dict(color='red', dash='dash'),
                opacity=0.7
            ))
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=bb_data['lower_band'],
                mode='lines',
                name='BB Inferior',
                line=dict(color='red', dash='dash'),
                opacity=0.7,
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
        
        # Configuración del layout
        chart_title = title or f'Precio con Indicadores - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            xaxis_title='Fecha',
            yaxis_title='Precio ($)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def momentum_indicators(self, title=None):
        """
        Gráfico con indicadores de momentum
        
        Args:
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'Stochastic', 'Williams %R'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # RSI
        rsi_data = self.indicators.rsi()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=rsi_data,
            mode='lines',
            name='RSI(14)',
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # Stochastic
        stoch_data = self.indicators.stochastic()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=stoch_data['k_percent'],
            mode='lines',
            name='%K',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=stoch_data['d_percent'],
            mode='lines',
            name='%D',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
        
        # Williams %R
        williams_data = self.indicators.williams_r()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=williams_data,
            mode='lines',
            name='Williams %R',
            line=dict(color='orange', width=2)
        ), row=3, col=1)
        fig.add_hline(y=-20, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=-80, line_dash="dash", line_color="green", row=3, col=1)
        
        # Configuración del layout
        chart_title = title or f'Indicadores de Momentum - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Fecha", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="Stochastic", row=2, col=1)
        fig.update_yaxes(title_text="Williams %R", row=3, col=1)
        
        return fig
    
    def volume_analysis(self, title=None):
        """
        Análisis de volumen con indicadores
        
        Args:
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        if self.data is None or self.data.empty:
            print("❌ No hay datos para visualizar")
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{self.symbol} - Precio', 'Volumen', 'OBV'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Precio
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['close'],
            mode='lines',
            name=f'{self.symbol}',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # VWAP
        vwap = self.indicators.vwap()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=vwap,
            mode='lines',
            name='VWAP',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
        
        # Volumen
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(self.data['close'], self.data['open'])]
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['volume'],
            name='Volumen',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        # OBV
        obv = self.indicators.obv()
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=obv,
            mode='lines',
            name='OBV',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        # Configuración del layout
        chart_title = title or f'Análisis de Volumen - {self.symbol}'
        fig.update_layout(
            title=chart_title,
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Fecha", row=3, col=1)
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volumen", row=2, col=1)
        fig.update_yaxes(title_text="OBV", row=3, col=1)
        
        return fig

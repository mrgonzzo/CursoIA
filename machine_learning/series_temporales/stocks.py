import pandas as pd
import numpy as np
# ... (otras importaciones)
import warnings

warnings.filterwarnings('ignore')
# ASUMIMOS que estas clases existen en tus módulos:
from descarga_datos import FinancialDataDownloader
from visualizacion import FinancialCharts
from indicadores_tecnicos import TechnicalAnalysisVisualizer, TechnicalIndicators


# Crear instancia y descargar datos
downloader = FinancialDataDownloader()
tsla_data = downloader.get_stock_data('TSLA', period='6mo')

if tsla_data is not None:
    print(tsla_data.head())
    print(f"\nColumnas: {tsla_data.columns.tolist()}")
    print(f"Forma del DataFrame: {tsla_data.shape}")
# No necesitamos los módulos de ARIMA o Backtesting para este análisis.

class SimpleTSLAAnalysis:
    """
    Realiza un análisis descriptivo y técnico básico pero potente para un símbolo.
    """

    def __init__(self, symbol, period='1y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.downloader = FinancialDataDownloader()

    def load_data(self):
        """ Cargar datos financieros """
        print(f"🔄 Descargando datos para {self.symbol}...")
        # Aseguramos que la columna del precio sea 'close'
        self.data = self.downloader.get_stock_data(self.symbol, period=self.period)

        if self.data is not None and not self.data.empty:
            print(f"✅ Datos cargados: {len(self.data)} registros ({self.period})")
            return True
        else:
            print("❌ Error cargando datos o DataFrame vacío.")
            return False

    def basic_descriptive_analysis(self):
        """
        Análisis financiero descriptivo: Retornos, Volatilidad, Máximos/Mínimos.
        """
        if self.data is None:
            print("❌ Datos no cargados.")
            return

        print(f"\n📈 ANÁLISIS DESCRIPTIVO BÁSICO - {self.symbol}")
        print("=" * 60)

        # 1. Cálculo de Retornos
        returns = self.data['close'].pct_change().dropna()

        current_price = self.data['close'].iloc[-1]
        initial_price = self.data['close'].iloc[0]
        total_return = (current_price / initial_price - 1) * 100

        # 2. Volatilidad y Sharpe Ratio
        volatilidad_anualizada = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Asumiendo tasa libre de riesgo de 0

        # 3. Máximos y Mínimos
        max_price = self.data['high'].max()
        min_price = self.data['low'].min()
        max_date = self.data['high'].idxmax().date()
        min_date = self.data['low'].idxmin().date()

        print(f"💰 Precio Actual: ${current_price:.2f}")
        print(f"  > Retorno Total ({self.period}): **{total_return:.2f}%**")
        print(f"  > Volatilidad Anualizada: **{volatilidad_anualizada:.2f}%**")
        print(f"  > **Sharpe Ratio (Anualizado): {sharpe_ratio:.3f}**")
        print(f"🔺 Máximo Histórico (High): ${max_price:.2f} (Fecha: {max_date})")
        print(f"🔻 Mínimo Histórico (Low): ${min_price:.2f} (Fecha: {min_date})")
        print("-" * 60)

    def simple_technical_analysis(self):
        """
        Análisis técnico con visualización (corregida) y resumen de indicadores clave (RSI, MACD, BB).
        """
        if self.data is None:
            print("❌ Datos no cargados.")
            return

        print(f"\n🔧 ANÁLISIS TÉCNICO POTENTE - {self.symbol}")
        print("=" * 60)

        # 1. Visualización (CORREGIDA para usar el método existente)
        print("[INFO] Generando gráfico de Análisis Técnico Integral (Velas, Bandas, RSI, MACD)...")
        tech_analyzer = TechnicalAnalysisVisualizer(self.data, self.symbol)

        # *** CORRECCIÓN DEL ERROR: Cambiamos 'plot_ohlc_with_volume' por 'comprehensive_analysis' ***
        ohlc_fig = tech_analyzer.comprehensive_analysis()
        ohlc_fig.show()

        # 2. Cálculo y Resumen de Indicadores Clave
        indicators = TechnicalIndicators(self.data)

        # RSI (Fuerza Relativa)
        rsi_data = indicators.rsi()
        current_rsi = rsi_data.iloc[-1]

        # MACD (Convergencia/Divergencia de Medias Móviles)
        macd_data = indicators.macd()
        current_macd = macd_data['macd'].iloc[-1]
        current_signal = macd_data['signal'].iloc[-1]

        # Bandas de Bollinger (BB)
        bb_data = indicators.bollinger_bands()
        current_price = self.data['close'].iloc[-1]
        upper_band = bb_data['upper_band'].iloc[-1]
        lower_band = bb_data['lower_band'].iloc[-1]
        percent_b = bb_data['percent_b'].iloc[-1]  # Posición del precio dentro de las bandas (0=inferior, 1=superior)

        # 3. Interpretación de Indicadores
        print("\n📊 INDICADORES ACTUALES Y SU INTERPRETACIÓN:")

        # RSI Interpretation
        rsi_status = "Neutral (30-70)"
        if current_rsi > 70:
            rsi_status = "**¡SOBRECOMPRA! (Posible corrección a la baja)**"
        elif current_rsi < 30:
            rsi_status = "**¡SOBREVENTA! (Posible rebote al alza)**"
        print(f"  **RSI (14):** {current_rsi:.2f} -> {rsi_status}")

        # MACD Interpretation
        macd_status = "Bajista (MACD < Señal)"
        if current_macd > current_signal:
            macd_status = "**Alcista (MACD > Señal) - Fuerte Momentum**"
        print(f"  **MACD:** {current_macd:.4f} vs Señal: {current_signal:.4f} -> {macd_status}")

        # Bollinger Bands Interpretation
        bb_status = "Rango Normal (Entre bandas)"
        if current_price > upper_band:
            bb_status = "**¡EXPANSIÓN! (Precio por encima de Banda Superior)**"
        elif current_price < lower_band:
            bb_status = "**¡CONTRACCIÓN! (Precio por debajo de Banda Inferior)**"

        print(f"  **Bandas de Bollinger (20,2):**")
        print(f"    - Banda Superior: ${upper_band:.2f}")
        print(f"    - Precio Actual: ${current_price:.2f} ({bb_status})")
        print(f"    - Banda Inferior: ${lower_band:.2f}")
        print(f"    - %B (Posición): {percent_b:.2f}")
        print("-" * 60)

    def run_analysis(self):
        """ Ejecuta el análisis completo """
        print(f"🚀 INICIANDO ANÁLISIS SIMPLE DE ALTO IMPACTO PARA {self.symbol}")
        print("=" * 60)

        if self.load_data():
            self.basic_descriptive_analysis()
            self.simple_technical_analysis()

        print(f"\n✅ ANÁLISIS FINALIZADO PARA {self.symbol}.")
        print("💡 Revise el gráfico interactivo generado por Plotly para el contexto visual.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    TICKET_A_ANALIZAR = 'TSLA'
    PERIODO_ANALISIS = '1y'

    tsla_analyzer = SimpleTSLAAnalysis(TICKET_A_ANALIZAR, period=PERIODO_ANALISIS)
    tsla_analyzer.run_analysis()
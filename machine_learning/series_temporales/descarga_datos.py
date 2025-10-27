# descarga_datos.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests


class FinancialDataDownloader:
    """
    Clase para descargar datos financieros de Yahoo Finance usando yfinance
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_stock_data(self, symbol, period='1y', interval='1d'):
        """
        Descarga datos históricos de un símbolo

        Args:
            symbol (str): Símbolo de la acción (ej: 'TSLA', 'AAPL')
            period (str): Período de tiempo ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Intervalo de datos ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            pd.DataFrame: DataFrame con datos OHLCV
        """
        try:
            print(f"📥 Descargando datos para {symbol} - Período: {period}")

            # Descargar datos usando yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                print(f"❌ No se pudieron descargar datos para {symbol}")
                return None

            # Verificar que tenemos las columnas necesarias
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"❌ Columna {col} no encontrada en los datos")
                    return None

            # Renombrar columnas para consistencia (inglés -> español)
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Limpiar datos: eliminar filas con valores NaN
            data = data.dropna()

            print(f"✅ Datos descargados exitosamente: {len(data)} registros")
            print(f"   Rango de fechas: {data.index[0].date()} a {data.index[-1].date()}")

            return data

        except Exception as e:
            print(f"❌ Error descargando datos para {symbol}: {str(e)}")
            return None

    def get_multiple_stocks(self, symbols, period='1y'):
        """
        Descarga datos para múltiples símbolos
        """
        all_data = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if data is not None:
                all_data[symbol] = data
        return all_data

    def get_company_info(self, symbol):
        """
        Obtiene información general de la compañía
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extraer información relevante
            company_info = {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'country': info.get('country', 'N/A'),
                'website': info.get('website', 'N/A')
            }

            return company_info
        except Exception as e:
            print(f"Error obteniendo info de {symbol}: {str(e)}")
            return None
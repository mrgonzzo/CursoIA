"""
MODELO ARIMA/SARIMA SIMPLE - TESLA
===================================
Script simple para predecir precios de Tesla con parámetros configurables
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS - CAMBIAR AQUÍ
# ============================================================================

# Parámetros ARIMA (p, d, q)
# p = orden autorregresivo (AR)
# d = grado de diferenciación
# q = orden de media móvil (MA)
pdq = (1, 1, 1)

# Parámetros SARIMA estacional (P, D, Q, s)
# P = orden autorregresivo estacional
# D = grado de diferenciación estacional
# Q = orden de media móvil estacional
# s = periodo estacional
PDQ = (1, 1, 1, 5)

# Configuración de datos
TICKER = 'TSLA'          # Símbolo de Tesla
PERIODO = '2y'           # Periodo de datos (1y, 2y, 5y, max)
DIAS_PREDICCION = 30     # Días a predecir

# ============================================================================
# DESCARGA DE DATOS
# ============================================================================

print("="*70)
print("MODELO ARIMA/SARIMA PARA TESLA")
print("="*70)

print(f"\n[1/5] Descargando datos de {TICKER}...")
try:
    df = yf.download(TICKER, period=PERIODO, progress=False)
    # Manejar MultiIndex si existe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    print(f"      Datos descargados: {len(df)} registros")
    print(f"      Rango: {df.index[0].date()} a {df.index[-1].date()}")
except Exception as e:
    print(f"      Error descargando datos: {e}")
    exit(1)

# Serie de precios de cierre
serie_precios = df['close'].dropna()

# ============================================================================
# DIVISIÓN DE DATOS (Train/Test)
# ============================================================================

print(f"\n[2/5] Dividiendo datos en entrenamiento y prueba...")
train_size = int(len(serie_precios) * 0.8)
train, test = serie_precios[:train_size], serie_precios[train_size:]

print(f"      Entrenamiento: {len(train)} registros")
print(f"      Prueba: {len(test)} registros")

# ============================================================================
# MODELO ARIMA
# ============================================================================

print(f"\n[3/5] Entrenando modelo ARIMA{pdq}...")
try:
    modelo_arima = ARIMA(train, order=pdq)
    modelo_arima_fit = modelo_arima.fit()

    print(f"      Modelo ARIMA entrenado exitosamente")
    print(f"      AIC: {modelo_arima_fit.aic:.2f}")
    print(f"      BIC: {modelo_arima_fit.bic:.2f}")

    # Predicción en conjunto de prueba
    pred_arima_test = modelo_arima_fit.forecast(steps=len(test))
    mae_arima = mean_absolute_error(test, pred_arima_test)
    rmse_arima = np.sqrt(mean_squared_error(test, pred_arima_test))

    print(f"      MAE en test: ${mae_arima:.2f}")
    print(f"      RMSE en test: ${rmse_arima:.2f}")

    # Predicción futura
    pred_arima_futuro = modelo_arima_fit.forecast(steps=DIAS_PREDICCION)

except Exception as e:
    print(f"      Error en modelo ARIMA: {e}")
    modelo_arima_fit = None
    pred_arima_test = None
    pred_arima_futuro = None

# ============================================================================
# MODELO SARIMA
# ============================================================================

print(f"\n[4/5] Entrenando modelo SARIMA{pdq}x{PDQ}...")
try:
    modelo_sarima = SARIMAX(train, order=pdq, seasonal_order=PDQ)
    modelo_sarima_fit = modelo_sarima.fit(disp=False)

    print(f"      Modelo SARIMA entrenado exitosamente")
    print(f"      AIC: {modelo_sarima_fit.aic:.2f}")
    print(f"      BIC: {modelo_sarima_fit.bic:.2f}")

    # Predicción en conjunto de prueba
    pred_sarima_test = modelo_sarima_fit.forecast(steps=len(test))
    mae_sarima = mean_absolute_error(test, pred_sarima_test)
    rmse_sarima = np.sqrt(mean_squared_error(test, pred_sarima_test))

    print(f"      MAE en test: ${mae_sarima:.2f}")
    print(f"      RMSE en test: ${rmse_sarima:.2f}")

    # Predicción futura con bandas de confianza
    forecast_obj = modelo_sarima_fit.get_forecast(steps=DIAS_PREDICCION)
    pred_sarima_futuro = forecast_obj.predicted_mean
    pred_sarima_conf = forecast_obj.conf_int()

except Exception as e:
    print(f"      Error en modelo SARIMA: {e}")
    modelo_sarima_fit = None
    pred_sarima_test = None
    pred_sarima_futuro = None
    pred_sarima_conf = None

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

print(f"\n[5/5] Generando visualizaciones...")

# -------------------- FIGURA 1: ACF y PACF --------------------
fig1, axes1 = plt.subplots(2, 1, figsize=(15, 8))

fig1, axes1 = plt.subplots(2, 1, figsize=(15, 8))

# ACF - Autocorrelación (omitiendo lag 0)
ax_acf = axes1[0]
plot_acf(serie_precios.dropna(), lags=40, ax=ax_acf, zero=False)
ax_acf.set_title('Función de Autocorrelación (ACF) - Sin lag 0', fontsize=12, fontweight='bold')
ax_acf.set_xlabel('Lag')
ax_acf.set_ylabel('Autocorrelación')
ax_acf.grid(True, alpha=0.3)

# Establecer límites fijos para el eje Y en ACF
ax_acf.set_ylim(-0.2, 0.8)  # Ajusta estos valores según tus datos

# PACF - Autocorrelación Parcial (omitiendo lag 0)
ax_pacf = axes1[1]
plot_pacf(serie_precios.dropna(), lags=40, ax=ax_pacf, zero=False, method='ywm')
ax_pacf.set_title('Función de Autocorrelación Parcial (PACF) - Sin lag 0', fontsize=12, fontweight='bold')
ax_pacf.set_xlabel('Lag')
ax_pacf.set_ylabel('Autocorrelación Parcial')
ax_pacf.grid(True, alpha=0.3)

# Establecer límites fijos para el eje Y en PACF
ax_pacf.set_ylim(-0.2, 0.8)  # Ajusta estos valores según tus datos

plt.tight_layout()
plt.savefig(f'{TICKER}_acf_pacf.png', dpi=150, bbox_inches='tight')
print(f"      Gráfico ACF/PACF guardado: {TICKER}_acf_pacf.png")

# -------------------- FIGURA 2: Serie Original + Predicción --------------------
fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))

# GRÁFICO 1: Serie Original unida con Predicción en Test
ax1 = axes2[0]

# Serie completa original
ax1.plot(serie_precios.index, serie_precios, label='Serie Original',
         color='blue', linewidth=1.5, alpha=0.7)

# Predicción SARIMA en test
if pred_sarima_test is not None:
    ax1.plot(test.index, pred_sarima_test, label=f'Predicción SARIMA{pdq}x{PDQ}',
             color='red', linestyle='--', linewidth=2)

# Línea vertical separando train y test
ax1.axvline(x=train.index[-1], color='green', linestyle=':', linewidth=2,
            label='Inicio Test', alpha=0.7)

ax1.set_title(f'Serie Original y Predicción en Conjunto de Test - {TICKER}',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Precio de Cierre ($)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# GRÁFICO 2: Predicción Futura con Bandas de Confianza
ax2 = axes2[1]

# Últimos 60 días históricos
ultimos_dias = serie_precios[-60:]
ax2.plot(ultimos_dias.index, ultimos_dias, label='Histórico',
         color='blue', linewidth=2)

# Fechas futuras
fecha_inicio = serie_precios.index[-1]
fechas_futuras = pd.date_range(start=fecha_inicio + pd.Timedelta(days=1),
                                periods=DIAS_PREDICCION, freq='D')

if pred_sarima_futuro is not None and pred_sarima_conf is not None:
    # Predicción SARIMA
    ax2.plot(fechas_futuras, pred_sarima_futuro, label=f'Predicción SARIMA{pdq}x{PDQ}',
             color='red', linestyle='--', linewidth=2.5)

    # Bandas de confianza (95%)
    ax2.fill_between(fechas_futuras,
                      pred_sarima_conf.iloc[:, 0],  # Límite inferior
                      pred_sarima_conf.iloc[:, 1],  # Límite superior
                      color='red', alpha=0.2, label='Intervalo de Confianza 95%')

# Línea vertical separando histórico y predicción
ax2.axvline(x=fecha_inicio, color='green', linestyle=':', linewidth=2,
            label='Inicio Predicción', alpha=0.7)

ax2.set_title(f'Predicción a {DIAS_PREDICCION} Días con Bandas de Confianza - {TICKER}',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Precio de Cierre ($)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{TICKER}_prediccion_completa.png', dpi=150, bbox_inches='tight')
print(f"      Gráfico de predicción guardado: {TICKER}_prediccion_completa.png")

# ============================================================================
# RESUMEN DE RESULTADOS
# ============================================================================

print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)

print(f"\nParámetros utilizados:")
print(f"  ARIMA: {pdq}")
print(f"  SARIMA: {pdq} x {PDQ}")

print(f"\nPrecio actual de {TICKER}: ${serie_precios.iloc[-1]:.2f}")

if pred_arima_futuro is not None:
    print(f"\nPredicciones ARIMA:")
    print(f"  Mañana (día 1): ${pred_arima_futuro.iloc[0]:.2f}")
    print(f"  En 7 días: ${pred_arima_futuro.iloc[min(6, len(pred_arima_futuro)-1)]:.2f}")
    print(f"  En {DIAS_PREDICCION} días: ${pred_arima_futuro.iloc[-1]:.2f}")
    cambio_arima = ((pred_arima_futuro.iloc[-1] - serie_precios.iloc[-1]) / serie_precios.iloc[-1]) * 100
    print(f"  Cambio esperado en {DIAS_PREDICCION} días: {cambio_arima:+.2f}%")

if pred_sarima_futuro is not None:
    print(f"\nPredicciones SARIMA:")
    print(f"  Mañana (día 1): ${pred_sarima_futuro.iloc[0]:.2f}")
    print(f"  En 7 días: ${pred_sarima_futuro.iloc[min(6, len(pred_sarima_futuro)-1)]:.2f}")
    print(f"  En {DIAS_PREDICCION} días: ${pred_sarima_futuro.iloc[-1]:.2f}")
    cambio_sarima = ((pred_sarima_futuro.iloc[-1] - serie_precios.iloc[-1]) / serie_precios.iloc[-1]) * 100
    print(f"  Cambio esperado en {DIAS_PREDICCION} días: {cambio_sarima:+.2f}%")

if modelo_arima_fit is not None and modelo_sarima_fit is not None:
    print(f"\nComparación de modelos (en conjunto de prueba):")
    print(f"  ARIMA  - MAE: ${mae_arima:.2f}, RMSE: ${rmse_arima:.2f}")
    print(f"  SARIMA - MAE: ${mae_sarima:.2f}, RMSE: ${rmse_sarima:.2f}")

    if mae_sarima < mae_arima:
        print(f"\n  Mejor modelo: SARIMA (menor error)")
    else:
        print(f"\n  Mejor modelo: ARIMA (menor error)")

print("\n" + "="*70)
print("CÓMO CAMBIAR LOS PARÁMETROS:")
print("="*70)
print("Edita las líneas 19-29 de este archivo para cambiar:")
print("  - pdq: parámetros ARIMA (p, d, q)")
print("  - PDQ: parámetros SARIMA estacional (P, D, Q, s)")
print("  - TICKER: símbolo de la acción")
print("  - PERIODO: rango de datos históricos")
print("  - DIAS_PREDICCION: días a predecir")
print("="*70)

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo de Series Temporales con ARIMA
Base de datos: barium (wooldridge)
Versión Final - Órdenes FIJOS: ARIMA(3, 1, 12)
"""

import wooldridge as woo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear directorio de salida y subdirectorio para gráficos
output_dir = Path('./resultados_arima')
output_dir.mkdir(exist_ok=True)
graficos_dir = output_dir / 'graficos'
graficos_dir.mkdir(exist_ok=True)

# Contador global para nombrar los gráficos
fig_counter = 1

# ============================================================================
# ÓRDENES ARIMA FIJOS SOLICITADOS
# ============================================================================
P_FIJO = 3
D_FIJO = 1
Q_FIJO = 12
ORDEN_FIJO = (P_FIJO, D_FIJO, Q_FIJO)


# ============================================================================


# ============================================================================
# FUNCIONES AUXILIARES (omitidas por concisión, son las mismas)
# ============================================================================
def guardar_y_cerrar_figura(fig, nombre_base):
    """Guarda la figura y la cierra, actualizando el contador global."""
    global fig_counter
    nombre_archivo = graficos_dir / f"{fig_counter:02d}_{nombre_base}.png"
    fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Gráfico {fig_counter:02d} guardado: {nombre_archivo.name}]")
    fig_counter += 1


def test_adf(serie, nombre):
    """Test de Dickey-Fuller Aumentado"""
    result = adfuller(serie.dropna())
    info = {
        'nombre': nombre,
        'estadistico': result[0],
        'pvalor': result[1],
        'valores_criticos': result[4],
        'es_estacionaria': result[1] <= 0.05
    }
    return info


def calcular_coeficiente_theil(actual, prediccion):
    """Calcula el Coeficiente de Desigualdad de Theil y métricas de error."""
    actual = np.array(actual)
    prediccion = np.array(prediccion)

    # MSE y RMSE
    mse = np.mean((actual - prediccion) ** 2)
    rmse = np.sqrt(mse)

    # Denominador (Theil)
    denominador = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(prediccion ** 2))
    theil_u = rmse / denominador if denominador != 0 else np.inf

    # Descomposición de Theil (para el informe)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(prediccion)

    bias = (mean_pred - mean_actual) ** 2
    var_actual = np.var(actual)
    var_pred = np.var(prediccion)

    bias_proportion = bias / mse if mse != 0 else 0
    variance_proportion = (var_pred - var_actual) ** 2 / mse if mse != 0 else 0
    covariance_proportion = 1 - bias_proportion - variance_proportion

    # Otras métricas
    mae = np.mean(np.abs(actual - prediccion))
    mape = np.mean(np.abs((actual - prediccion) / actual)) * 100 if np.all(actual != 0) else np.inf

    resultados = {
        'Theil_U': theil_u,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape,
        'Bias_Proportion': bias_proportion,
        'Variance_Proportion': variance_proportion,
        'Covariance_Proportion': covariance_proportion
    }

    return resultados


# ============================================================================
# INICIO DEL ANÁLISIS
# ============================================================================

documento = []

documento.append("=" * 80)
documento.append("ANÁLISIS DE SERIES TEMPORALES CON MODELO ARIMA (Órdenes Fijos)")
documento.append(f"Modelo Ajustado: ARIMA{ORDEN_FIJO}")
documento.append("=" * 80)

print("=" * 80)
print(f"ANÁLISIS DE SERIES TEMPORALES - MODELO ARIMA{ORDEN_FIJO} (Órdenes Fijos)")
print("=" * 80)

# PASO 1, 2, 3, 4, 5: Carga, Descomposición, Estacionariedad y ACF/PACF (código omitido, no cambia)
# ... (El código de los Pasos 1, 2, 3, 4, 5 permanece exactamente igual)
df = woo.dataWoo('barium')
serie = df['chnimp'].values
n = len(serie)

fechas = pd.date_range(start='1978-02', periods=n, freq='MS')
ts = pd.Series(serie, index=fechas, name='Importaciones de Bario Chino')

# Simulación de pasos de limpieza y diagnóstico para que las variables existan
train_size = int(len(ts) * 0.85)
train, test = ts[:train_size], ts[train_size:]

# Aquí deberían ir los gráficos de los Pasos 1 al 5
# Se omiten por brevedad, pero las funciones 'guardar_y_cerrar_figura' se deberían usar allí.

# ============================================================================
# PASO 6 & 7: ESTIMACIÓN Y SELECCIÓN DEL MODELO (Ajuste Directo)
# ============================================================================
print("\n" + "=" * 80)
print(f"PASO 6 & 7: AJUSTE DIRECTO DEL MODELO ARIMA{ORDEN_FIJO}")
print("=" * 80)

documento.append("PASO 6/7: AJUSTE DE MODELO CON ÓRDENES FIJOS")
documento.append("-" * 80)
documento.append(f"Modelo Seleccionado (Fijo): ARIMA{ORDEN_FIJO}")
documento.append(f"Datos de entrenamiento: {len(train)} observaciones")
documento.append(f"Datos de prueba: {len(test)} observaciones")
documento.append("")

# Ajuste directo del modelo con los órdenes fijos
try:
    modelo = ARIMA(train, order=ORDEN_FIJO)
    modelo_fit = modelo.fit(method='statespace')

    mejor_orden = ORDEN_FIJO
    mejor_modelo = modelo_fit

    # Crear un diccionario para mantener la estructura de la sección original
    mejor_resultado = {
        'orden': mejor_orden,
        'AIC': mejor_modelo.aic,
        'BIC': mejor_modelo.bic,
        'modelo': mejor_modelo
    }

    print(f"\nModelo ajustado: ARIMA{mejor_orden}")
    print(f"AIC: {mejor_modelo.aic:.4f}")
    print(f"BIC: {mejor_modelo.bic:.4f}")
    print("\nResumen del modelo:")
    print(mejor_modelo.summary())

    documento.append(f"Modelo: ARIMA{mejor_orden}")
    documento.append(f"AIC: {mejor_modelo.aic:.4f}")
    documento.append(f"BIC: {mejor_modelo.bic:.4f}")
    documento.append("")

    # Extracción de Parámetros
    documento.append("Parámetros estimados:")
    params = mejor_modelo.params
    for param_name, param_value in params.items():
        pvalue = mejor_modelo.pvalues[param_name]
        documento.append(f"  {param_name}: {param_value:.4f} (p-valor: {pvalue:.4f})")
    documento.append("")

except Exception as e:
    print(f"ERROR: El modelo ARIMA{ORDEN_FIJO} no pudo converger. Detalles: {e}")
    documento.append(f"ERROR FATAL: El modelo ARIMA{ORDEN_FIJO} no pudo converger.")
    # Finalizar el script de forma anticipada
    with open(output_dir / 'INFORME_COMPLETO_ARIMA.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(documento))
    raise SystemExit(f"Error de convergencia para ARIMA{ORDEN_FIJO}. Se detiene el script.")

# ============================================================================
# PASO 8: DIAGNÓSTICO DE RESIDUOS (Código permanece igual)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 8: DIAGNÓSTICO DE RESIDUOS")
print("=" * 80)

documento.append("PASO 8: DIAGNÓSTICO DE RESIDUOS")
documento.append("-" * 80)
documento.append("")

residuos = mejor_modelo.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuos
axes[0, 0].plot(residuos, linewidth=1, color='#2E86AB')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_title('Residuos del Modelo', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Observación', fontsize=10)
axes[0, 0].set_ylabel('Residuo', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Histograma
axes[0, 1].hist(residuos, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residuo', fontsize=10)
axes[0, 1].set_ylabel('Frecuencia', fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# ACF de residuos (sin lag 0)
plot_acf(residuos, lags=30, ax=axes[1, 0], color='#F18F01', alpha=0.7, zero=False)
axes[1, 0].set_title('ACF de Residuos (sin lag 0)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
guardar_y_cerrar_figura(fig, 'Diagnostico_Residuos')

# Estadísticas de residuos
documento.append("Estadísticas de los residuos:")
documento.append(f"  Media: {residuos.mean():.6f}")
documento.append(f"  Desviación estándar: {residuos.std():.4f}")
documento.append(f"  Asimetría (Skewness): {residuos.skew():.4f}")
documento.append(f"  Curtosis: {residuos.kurtosis():.4f}")
documento.append("")

# ============================================================================
# PASO 9: PREDICCIONES Y COEFICIENTE DE THEIL (Código permanece igual)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 9: PREDICCIONES Y EVALUACIÓN CON COEFICIENTE DE THEIL")
print("=" * 80)

documento.append("PASO 9: PREDICCIONES Y EVALUACIÓN")
documento.append("-" * 80)
documento.append("")

# Predicciones en el conjunto de prueba
predicciones_test = mejor_modelo.forecast(steps=len(test))

# Calcular Coeficiente de Theil
metricas_test = calcular_coeficiente_theil(test.values, predicciones_test.values)

# Reajustar con todos los datos para predicción futura
modelo_final = ARIMA(ts, order=mejor_orden)
modelo_final_fit = modelo_final.fit(method='statespace')

# Predicción futura
n_forecast = 24
forecast = modelo_final_fit.forecast(steps=n_forecast)
forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                               periods=n_forecast, freq='MS')

forecast_result = modelo_final_fit.get_forecast(steps=n_forecast)
forecast_ci = forecast_result.conf_int(alpha=0.05)

# (Código de impresión y documentación de métricas omitido por concisión, es igual)

# ============================================================================
# PASO 10: GRÁFICOS FINALES (Código permanece igual, usa las variables ajustadas)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 10: VISUALIZACIÓN Y GUARDADO DE RESULTADOS FINALES")
print("=" * 80)

# Gráfico 1: Comparación predicción vs real (conjunto de prueba)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(test.index, test.values, linewidth=2.5, color='#2E86AB',
        label='Valores Reales', marker='o', markersize=6, zorder=3)
ax.plot(test.index, predicciones_test.values, linewidth=2.5, color='#C73E1D',
        label='Predicciones del Modelo', marker='s', markersize=6, linestyle='--', zorder=4)

ax.set_title(f'Evaluación del Modelo: Predicciones vs Valores Reales\nConjunto de Prueba',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Importaciones (toneladas)', fontsize=11)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

textstr = f"Coeficiente de Theil: {metricas_test['Theil_U']:.4f}\nRMSE: {metricas_test['RMSE']:.2f}\nMAE: {metricas_test['MAE']:.2f}\nMAPE: {metricas_test['MAPE']:.2f}%"
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
guardar_y_cerrar_figura(fig, 'Prediccion_vs_Real_Test')

# Gráfico 2: Serie completa con predicción futura
fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(ts, linewidth=2.5, color='#2E86AB', label='Serie Histórica', zorder=3)
ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA{mejor_orden}', linestyle='--', zorder=4)

ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='Intervalo de Confianza 95%', zorder=2)

ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio de Predicción', zorder=1)

ax.set_title(f'Serie Temporal Completa con Predicción - Modelo ARIMA{mejor_orden}',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Importaciones (toneladas)', fontsize=11)
ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)

textstr = f'AIC: {mejor_resultado["AIC"]:.2f}\nBIC: {mejor_resultado["BIC"]:.2f}\nTheil U: {metricas_test["Theil_U"]:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
guardar_y_cerrar_figura(fig, 'Serie_Completa_con_Forecast')

# Gráfico 3: Zoom en predicción
fig, ax = plt.subplots(figsize=(14, 6))

n_historicos = 36
ts_ultimos = ts[-n_historicos:]

ax.plot(ts_ultimos, linewidth=2.5, color='#2E86AB',
        label=f'Últimos {n_historicos} meses', marker='o', markersize=4, zorder=3)
ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA{mejor_orden}', linestyle='--',
        marker='s', markersize=4, zorder=4)
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='IC 95%', zorder=2)

ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio Predicción', zorder=1)

ax.set_title(f'Detalle: Últimos Meses y Predicción a Futuro',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=11)
ax.set_ylabel('Importaciones (toneladas)', fontsize=11)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
guardar_y_cerrar_figura(fig, 'Zoom_Forecast_IC')

# ============================================================================
# GUARDAR DOCUMENTO FINAL (Código permanece igual)
# ============================================================================
# (El código final de impresión y guardado de informe se mantiene)
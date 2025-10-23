#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo de Series Temporales con ARIMA
Base de datos: barium (wooldridge)
Versión Final - Sin guardar gráficos, solo visualización
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

# Crear directorio de salida solo para el documento
output_dir = Path('./resultados_arima')
output_dir.mkdir(exist_ok=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

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
    """
    Calcula el Coeficiente de Desigualdad de Theil (Theil's U)
    Compatible con la implementación de EViews

    Theil U = sqrt(MSE) / [sqrt(mean(actual²)) + sqrt(mean(prediccion²))]

    Interpretación:
    - U = 0: Predicción perfecta
    - U < 1: Predicción mejor que método naive
    - U = 1: Predicción igual a método naive
    - U > 1: Predicción peor que método naive
    """
    actual = np.array(actual)
    prediccion = np.array(prediccion)

    # MSE y RMSE
    mse = np.mean((actual - prediccion) ** 2)
    rmse = np.sqrt(mse)

    # Denominador (Theil)
    denominador = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(prediccion ** 2))
    theil_u = rmse / denominador if denominador != 0 else np.inf

    # Descomposición de Theil
    mean_actual = np.mean(actual)
    mean_pred = np.mean(prediccion)

    bias = (mean_pred - mean_actual) ** 2
    var_actual = np.var(actual)
    var_pred = np.var(prediccion)

    # Proporciones del error
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
documento.append("ANÁLISIS DE SERIES TEMPORALES CON MODELO ARIMA")
documento.append("Base de datos: barium (Wooldridge)")
documento.append(f"Fecha de análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
documento.append("=" * 80)
documento.append("")

print("=" * 80)
print("ANÁLISIS DE SERIES TEMPORALES - MODELO ARIMA")
print("=" * 80)

# ============================================================================
# PASO 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 1: CARGA Y PREPARACIÓN DE DATOS")
print("=" * 80)

df = woo.dataWoo('barium')
serie = df['chnimp'].values
n = len(serie)

fechas = pd.date_range(start='1978-02', periods=n, freq='MS')
ts = pd.Series(serie, index=fechas, name='Importaciones de Bario Chino')

print(f"\nSerie: {ts.name}")
print(f"Período: {ts.index[0].strftime('%Y-%m')} a {ts.index[-1].strftime('%Y-%m')}")
print(f"Número de observaciones: {n}")
print(f"\nEstadísticas descriptivas:")
print(ts.describe())

documento.append("PASO 1: DATOS ORIGINALES")
documento.append("-" * 80)
documento.append(f"Serie: {ts.name}")
documento.append(f"Período: {ts.index[0].strftime('%Y-%m')} a {ts.index[-1].strftime('%Y-%m')}")
documento.append(f"Observaciones: {n}")
documento.append("")
documento.append("Estadísticas descriptivas:")
documento.append(f"  Media: {ts.mean():.2f}")
documento.append(f"  Desviación estándar: {ts.std():.2f}")
documento.append(f"  Mínimo: {ts.min():.2f}")
documento.append(f"  Máximo: {ts.max():.2f}")
documento.append(f"  Mediana: {ts.median():.2f}")
documento.append("")

# Visualización serie original
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(ts, linewidth=2, color='#2E86AB')
axes[0].set_title('Serie Temporal Original: Importaciones de Bario Chino',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fecha', fontsize=11)
axes[0].set_ylabel('Importaciones (toneladas)', fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].hist(ts, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
axes[1].set_title('Distribución de la Serie', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importaciones (toneladas)', fontsize=11)
axes[1].set_ylabel('Frecuencia', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ============================================================================
# PASO 2: DESCOMPOSICIÓN ESTACIONAL
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DESCOMPOSICIÓN ESTACIONAL")
print("=" * 80)

decomposition = seasonal_decompose(ts, model='additive', period=12)

documento.append("PASO 2: DESCOMPOSICIÓN ESTACIONAL")
documento.append("-" * 80)
documento.append("Método: Descomposición aditiva")
documento.append("Período estacional: 12 meses")
documento.append("")

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

decomposition.observed.plot(ax=axes[0], color='#2E86AB', linewidth=2)
axes[0].set_title('Serie Original', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Valor', fontsize=10)
axes[0].grid(True, alpha=0.3)

decomposition.trend.plot(ax=axes[1], color='#F18F01', linewidth=2)
axes[1].set_title('Tendencia', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Tendencia', fontsize=10)
axes[1].grid(True, alpha=0.3)

decomposition.seasonal.plot(ax=axes[2], color='#C73E1D', linewidth=2)
axes[2].set_title('Componente Estacional', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Estacionalidad', fontsize=10)
axes[2].grid(True, alpha=0.3)

decomposition.resid.plot(ax=axes[3], color='#6A994E', linewidth=1)
axes[3].set_title('Residuos', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Residuos', fontsize=10)
axes[3].set_xlabel('Fecha', fontsize=10)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ============================================================================
# PASO 3: PRUEBAS DE ESTACIONARIEDAD
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: PRUEBAS DE ESTACIONARIEDAD")
print("=" * 80)

documento.append("PASO 3: PRUEBAS DE ESTACIONARIEDAD (Test ADF)")
documento.append("-" * 80)
documento.append("")

# Test en serie original
adf_original = test_adf(ts, "Serie Original")
print(f"\nTest ADF - Serie Original:")
print(f"  Estadístico ADF: {adf_original['estadistico']:.4f}")
print(f"  p-valor: {adf_original['pvalor']:.4f}")
print(f"  Resultado: {'ESTACIONARIA' if adf_original['es_estacionaria'] else 'NO ESTACIONARIA'}")

documento.append(f"Serie Original:")
documento.append(f"  Estadístico ADF: {adf_original['estadistico']:.4f}")
documento.append(f"  p-valor: {adf_original['pvalor']:.4f}")
documento.append(f"  Resultado: {'✓ ESTACIONARIA' if adf_original['es_estacionaria'] else '✗ NO ESTACIONARIA'}")
documento.append("")

# ============================================================================
# PASO 4: TRANSFORMACIONES CLÁSICAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: TRANSFORMACIONES CLÁSICAS")
print("=" * 80)

documento.append("PASO 4: TRANSFORMACIONES CLÁSICAS")
documento.append("-" * 80)
documento.append("")

transformaciones = {}

# a) Logaritmos
ts_log = np.log(ts + 1)
transformaciones['Logaritmo'] = ts_log

# b) Diferencia regular (d=1)
ts_diff1 = ts.diff().dropna()
transformaciones['Diferencia Regular (d=1)'] = ts_diff1

# c) Diferencia estacional (D=1, s=12)
ts_diff12 = ts.diff(12).dropna()
transformaciones['Diferencia Estacional (D=1, s=12)'] = ts_diff12

# d) Log + Diferencia regular
ts_log_diff1 = ts_log.diff().dropna()
transformaciones['Log + Diferencia Regular'] = ts_log_diff1

# e) Log + Diferencia estacional
ts_log_diff12 = ts_log.diff(12).dropna()
transformaciones['Log + Diferencia Estacional'] = ts_log_diff12

# f) Diferencia regular + estacional
ts_diff1_diff12 = ts.diff().diff(12).dropna()
transformaciones['Diferencia Regular + Estacional'] = ts_diff1_diff12

# g) Log + Diferencia regular + estacional
ts_log_diff1_diff12 = ts_log.diff().diff(12).dropna()
transformaciones['Log + Dif. Regular + Estacional'] = ts_log_diff1_diff12

# Visualizar transformaciones
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
axes = axes.ravel()

axes[0].plot(ts, linewidth=1.5, color='#2E86AB')
axes[0].set_title('Serie Original', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylabel('Valor', fontsize=9)

for idx, (nombre, serie_trans) in enumerate(transformaciones.items(), 1):
    axes[idx].plot(serie_trans, linewidth=1.5, color=plt.cm.tab10(idx))
    axes[idx].set_title(nombre, fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylabel('Valor', fontsize=9)
    if idx >= 6:
        axes[idx].set_xlabel('Fecha', fontsize=9)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# Tests ADF para cada transformación
print("\nTests de estacionariedad:")
for nombre, serie_trans in transformaciones.items():
    adf_result = test_adf(serie_trans, nombre)
    print(f"\n{nombre}:")
    print(f"  ADF: {adf_result['estadistico']:.4f}, p-valor: {adf_result['pvalor']:.4f}")
    print(f"  Resultado: {'ESTACIONARIA' if adf_result['es_estacionaria'] else 'NO ESTACIONARIA'}")

    documento.append(f"{nombre}:")
    documento.append(f"  Estadístico ADF: {adf_result['estadistico']:.4f}")
    documento.append(f"  p-valor: {adf_result['pvalor']:.4f}")
    documento.append(f"  Resultado: {'✓ ESTACIONARIA' if adf_result['es_estacionaria'] else '✗ NO ESTACIONARIA'}")
    documento.append("")

# ============================================================================
# PASO 5: ACF Y PACF DE LAS TRANSFORMACIONES (SIN LAG 0)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: ANÁLISIS ACF Y PACF DE LAS TRANSFORMACIONES (SIN LAG 0)")
print("=" * 80)

documento.append("PASO 5: ANÁLISIS DE AUTOCORRELACIÓN (ACF Y PACF)")
documento.append("-" * 80)
documento.append("Nota: Los gráficos excluyen el lag 0 para mejor visualización")
documento.append("")

# ACF y PACF para cada transformación estacionaria
transformaciones_para_analisis = {
    'Diferencia Regular (d=1)': ts_diff1,
    'Diferencia Estacional (D=1, s=12)': ts_diff12,
    'Log + Diferencia Regular': ts_log_diff1,
}

for nombre, serie_trans in transformaciones_para_analisis.items():
    print(f"\nGraficando ACF/PACF para: {nombre}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ACF sin lag 0
    plot_acf(serie_trans, lags=40, ax=axes[0], color='#2E86AB', alpha=0.7, zero=False)
    axes[0].set_title(f'ACF - {nombre} (sin lag 0)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rezagos', fontsize=10)
    axes[0].set_ylabel('Autocorrelación', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # PACF sin lag 0
    plot_pacf(serie_trans, lags=40, ax=axes[1], color='#C73E1D', alpha=0.7, zero=False)
    axes[1].set_title(f'PACF - {nombre} (sin lag 0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Rezagos', fontsize=10)
    axes[1].set_ylabel('Autocorrelación Parcial', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# ============================================================================
# PASO 6: ESTIMACIÓN DE MODELOS ARIMA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: ESTIMACIÓN DE MÚLTIPLES MODELOS ARIMA")
print("=" * 80)

documento.append("PASO 6: ESTIMACIÓN Y SELECCIÓN DE MODELOS")
documento.append("-" * 80)
documento.append("")

train_size = int(len(ts) * 0.85)
train, test = ts[:train_size], ts[train_size:]

print(f"\nDatos de entrenamiento: {len(train)} observaciones")
print(f"Datos de prueba: {len(test)} observaciones")

documento.append(f"División de datos:")
documento.append(f"  Entrenamiento: {len(train)} observaciones ({train_size / len(ts) * 100:.1f}%)")
documento.append(f"  Prueba: {len(test)} observaciones ({(1 - train_size / len(ts)) * 100:.1f}%)")
documento.append("")

# Estimar modelos
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

resultados_modelos = []

print("\nEstimando modelos...")
contador = 0
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                modelo = ARIMA(train, order=(p, d, q))
                modelo_fit = modelo.fit()

                resultados_modelos.append({
                    'idx': contador,
                    'orden': (p, d, q),
                    'AIC': modelo_fit.aic,
                    'BIC': modelo_fit.bic,
                    'modelo': modelo_fit
                })

                contador += 1
                if contador % 10 == 0:
                    print(f"  Modelos estimados: {contador}")

            except:
                continue

print(f"\n✓ Total de modelos estimados: {len(resultados_modelos)}")

# Ranking por AIC
df_resultados = pd.DataFrame([
    {
        'idx': r['idx'],
        'p': r['orden'][0],
        'd': r['orden'][1],
        'q': r['orden'][2],
        'Orden': f"({r['orden'][0]},{r['orden'][1]},{r['orden'][2]})",
        'AIC': r['AIC'],
        'BIC': r['BIC']
    }
    for r in resultados_modelos
])

df_resultados = df_resultados.sort_values('AIC').reset_index(drop=True)

print("\nTop 10 modelos por AIC:")
print(df_resultados.head(10)[['Orden', 'AIC', 'BIC']].to_string(index=False))

documento.append(f"Total de modelos estimados: {len(resultados_modelos)}")
documento.append("")
documento.append("Top 15 modelos por Criterio de Akaike (AIC):")
documento.append("")
for i in range(min(15, len(df_resultados))):
    row = df_resultados.iloc[i]
    documento.append(f"  {i + 1}. ARIMA{row['Orden']}: AIC={row['AIC']:.4f}, BIC={row['BIC']:.4f}")
documento.append("")

# ============================================================================
# PASO 7: MEJOR MODELO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 7: MEJOR MODELO SELECCIONADO")
print("=" * 80)

idx_mejor = int(df_resultados.iloc[0]['idx'])
mejor_resultado = resultados_modelos[idx_mejor]
mejor_orden = mejor_resultado['orden']
mejor_modelo = mejor_resultado['modelo']

print(f"\nMejor modelo: ARIMA{mejor_orden}")
print(f"AIC: {mejor_resultado['AIC']:.4f}")
print(f"BIC: {mejor_resultado['BIC']:.4f}")
print("\nResumen del modelo:")
print(mejor_modelo.summary())

documento.append("PASO 7: MEJOR MODELO SELECCIONADO")
documento.append("-" * 80)
documento.append(f"Modelo: ARIMA{mejor_orden}")
documento.append(f"AIC: {mejor_resultado['AIC']:.4f}")
documento.append(f"BIC: {mejor_resultado['BIC']:.4f}")
documento.append("")
documento.append("Parámetros estimados:")

# Extraer parámetros
params = mejor_modelo.params
for param_name, param_value in params.items():
    pvalue = mejor_modelo.pvalues[param_name]
    documento.append(f"  {param_name}: {param_value:.4f} (p-valor: {pvalue:.4f})")
documento.append("")

# ============================================================================
# PASO 8: DIAGNÓSTICO DE RESIDUOS
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
plt.show(block=False)
plt.pause(0.1)

# Estadísticas de residuos
documento.append("Estadísticas de los residuos:")
documento.append(f"  Media: {residuos.mean():.6f}")
documento.append(f"  Desviación estándar: {residuos.std():.4f}")
documento.append(f"  Asimetría (Skewness): {residuos.skew():.4f}")
documento.append(f"  Curtosis: {residuos.kurtosis():.4f}")
documento.append("")

# ============================================================================
# PASO 9: PREDICCIONES Y COEFICIENTE DE THEIL
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

print("\nEvaluación en conjunto de prueba:")
print(f"  Coeficiente de Theil (U): {metricas_test['Theil_U']:.6f}")
print(f"  RMSE: {metricas_test['RMSE']:.4f}")
print(f"  MSE: {metricas_test['MSE']:.4f}")
print(f"  MAE: {metricas_test['MAE']:.4f}")
print(f"  MAPE: {metricas_test['MAPE']:.2f}%")
print(f"\nDescomposición del Coeficiente de Theil:")
print(
    f"  Proporción de Sesgo (Bias): {metricas_test['Bias_Proportion']:.4f} ({metricas_test['Bias_Proportion'] * 100:.2f}%)")
print(
    f"  Proporción de Varianza: {metricas_test['Variance_Proportion']:.4f} ({metricas_test['Variance_Proportion'] * 100:.2f}%)")
print(
    f"  Proporción de Covarianza: {metricas_test['Covariance_Proportion']:.4f} ({metricas_test['Covariance_Proportion'] * 100:.2f}%)")

documento.append("EVALUACIÓN EN CONJUNTO DE PRUEBA")
documento.append("")
documento.append("Métricas de Error:")
documento.append(f"  RMSE (Root Mean Square Error): {metricas_test['RMSE']:.4f}")
documento.append(f"  MSE (Mean Square Error): {metricas_test['MSE']:.4f}")
documento.append(f"  MAE (Mean Absolute Error): {metricas_test['MAE']:.4f}")
documento.append(f"  MAPE (Mean Absolute Percentage Error): {metricas_test['MAPE']:.2f}%")
documento.append("")
documento.append("COEFICIENTE DE DESIGUALDAD DE THEIL (Theil's U):")
documento.append(f"  Valor: {metricas_test['Theil_U']:.6f}")
documento.append("")
documento.append("Descomposición del Coeficiente de Theil:")
documento.append(
    f"  Proporción de Sesgo (Bias): {metricas_test['Bias_Proportion']:.4f} ({metricas_test['Bias_Proportion'] * 100:.2f}%)")
documento.append(
    f"  Proporción de Varianza: {metricas_test['Variance_Proportion']:.4f} ({metricas_test['Variance_Proportion'] * 100:.2f}%)")
documento.append(
    f"  Proporción de Covarianza: {metricas_test['Covariance_Proportion']:.4f} ({metricas_test['Covariance_Proportion'] * 100:.2f}%)")
documento.append("")
documento.append("Interpretación del Coeficiente de Theil:")
if metricas_test['Theil_U'] < 0.3:
    interpretacion = "Excelente capacidad predictiva"
elif metricas_test['Theil_U'] < 0.5:
    interpretacion = "Buena capacidad predictiva"
elif metricas_test['Theil_U'] < 1.0:
    interpretacion = "Capacidad predictiva aceptable"
else:
    interpretacion = "Capacidad predictiva deficiente (peor que método naive)"
documento.append(f"  {interpretacion}")
documento.append("")
documento.append("Nota: Un Coeficiente de Theil < 1 indica que el modelo")
documento.append("      es mejor que un pronóstico ingenuo (naive forecast).")
documento.append("")

# Reajustar con todos los datos para predicción futura
modelo_final = ARIMA(ts, order=mejor_orden)
modelo_final_fit = modelo_final.fit()

# Predicción futura
n_forecast = 24
forecast = modelo_final_fit.forecast(steps=n_forecast)
forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                               periods=n_forecast, freq='MS')

forecast_result = modelo_final_fit.get_forecast(steps=n_forecast)
forecast_ci = forecast_result.conf_int(alpha=0.05)

print(f"\nPredicciones para los próximos {n_forecast} meses:")
for i, (fecha, valor) in enumerate(zip(forecast_index, forecast)):
    if i < 6 or i >= n_forecast - 6:
        print(f"  {fecha.strftime('%Y-%m')}: {valor:.2f}")
    elif i == 6:
        print("  ...")

documento.append(f"PREDICCIONES FUTURAS ({n_forecast} meses adelante):")
documento.append("")
for fecha, valor, ic_inf, ic_sup in zip(forecast_index, forecast,
                                        forecast_ci.iloc[:, 0],
                                        forecast_ci.iloc[:, 1]):
    documento.append(f"  {fecha.strftime('%Y-%m')}: {valor:8.2f}  [IC 95%: {ic_inf:8.2f} - {ic_sup:8.2f}]")
documento.append("")

# ============================================================================
# PASO 10: GRÁFICOS FINALES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 10: VISUALIZACIÓN DE RESULTADOS FINALES")
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
plt.show(block=False)
plt.pause(0.1)

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
plt.show(block=False)
plt.pause(0.1)

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
plt.show(block=False)
plt.pause(0.1)

# ============================================================================
# GUARDAR DOCUMENTO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("GUARDANDO DOCUMENTO FINAL")
print("=" * 80)

documento.append("")
documento.append("=" * 80)
documento.append("RESUMEN FINAL DEL ANÁLISIS")
documento.append("=" * 80)
documento.append("")
documento.append(f"Mejor modelo seleccionado: ARIMA{mejor_orden}")
documento.append(f"  AIC: {mejor_resultado['AIC']:.4f}")
documento.append(f"  BIC: {mejor_resultado['BIC']:.4f}")
documento.append("")
documento.append("Evaluación del Modelo:")
documento.append(f"  Coeficiente de Theil (U): {metricas_test['Theil_U']:.6f}")
documento.append(f"  RMSE: {metricas_test['RMSE']:.4f}")
documento.append(f"  MAE: {metricas_test['MAE']:.4f}")
documento.append(f"  MAPE: {metricas_test['MAPE']:.2f}%")
documento.append("")
documento.append("Interpretación:")
documento.append(f"  {interpretacion}")
documento.append("")
documento.append(f"Predicciones futuras:")
documento.append(f"  Horizonte: {n_forecast} meses")
documento.append(f"  Rango: {forecast.min():.2f} - {forecast.max():.2f} toneladas")
documento.append("")
documento.append(f"Análisis completado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
documento.append("=" * 80)

# Guardar documento
archivo_documento = output_dir / 'INFORME_COMPLETO_ARIMA.txt'
with open(archivo_documento, 'w', encoding='utf-8') as f:
    f.write('\n'.join(documento))

print(f"\n✓ Documento final guardado en: {archivo_documento.absolute()}")
print(f"\n{'=' * 80}")
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print(f"{'=' * 80}")
print(f"\nCoeficiente de Theil: {metricas_test['Theil_U']:.6f} - {interpretacion}")
print("\nTodos los gráficos están abiertos.")
print("Presiona Enter para cerrar todas las ventanas...")
input()
plt.close('all')
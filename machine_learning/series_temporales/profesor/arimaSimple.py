#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Simple para Ajustar un Modelo ARIMA Específico
Cambia los parámetros (p, d, q) según necesites
"""

import wooldridge as woo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN DEL MODELO - CAMBIA AQUÍ LOS PARÁMETROS
# ============================================================================

p = 1  # Orden AR (autorregresivo)
d = 1  # Orden de diferenciación
q = 1  # Orden MA (media móvil)

print("="*80)
print(f"AJUSTE DE MODELO ARIMA({p}, {d}, {q})")
print("="*80)

# ============================================================================
# CARGAR DATOS
# ============================================================================
print("\n1. Cargando datos...")

df = woo.dataWoo('barium')
serie = df['chnimp'].values
n = len(serie)

fechas = pd.date_range(start='1978-02', periods=n, freq='MS')
ts = pd.Series(serie, index=fechas, name='Importaciones de Bario Chino')

print(f"   Serie: {ts.name}")
print(f"   Observaciones: {n}")
print(f"   Media: {ts.mean():.2f}")
print(f"   Desv. Std: {ts.std():.2f}")

# ============================================================================
# AJUSTAR EL MODELO
# ============================================================================
print(f"\n2. Ajustando modelo ARIMA({p}, {d}, {q})...")

modelo = ARIMA(ts, order=(p, d, q))
modelo_ajustado = modelo.fit()

print(f"   ✓ Modelo ajustado exitosamente")
print(f"\n   AIC: {modelo_ajustado.aic:.4f}")
print(f"   BIC: {modelo_ajustado.bic:.4f}")
print(f"   Log-Likelihood: {modelo_ajustado.llf:.4f}")

# ============================================================================
# RESUMEN DEL MODELO
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DEL MODELO")
print("="*80)
print(modelo_ajustado.summary())

# ============================================================================
# PARÁMETROS ESTIMADOS
# ============================================================================
print("\n" + "="*80)
print("PARÁMETROS ESTIMADOS")
print("="*80)

for param_name, param_value in modelo_ajustado.params.items():
    pvalue = modelo_ajustado.pvalues[param_name]
    significativo = "***" if pvalue < 0.01 else "**" if pvalue < 0.05 else "*" if pvalue < 0.1 else ""
    print(f"  {param_name:15s}: {param_value:10.4f}  (p-valor: {pvalue:.4f}) {significativo}")

print("\n  Significancia: *** p<0.01, ** p<0.05, * p<0.1")

# ============================================================================
# RESIDUOS
# ============================================================================
print("\n3. Analizando residuos...")

residuos = modelo_ajustado.resid

print(f"\n   Media de residuos: {residuos.mean():.6f}")
print(f"   Desv. Std residuos: {residuos.std():.4f}")
print(f"   Asimetría: {residuos.skew():.4f}")
print(f"   Curtosis: {residuos.kurtosis():.4f}")

# ============================================================================
# ACF Y PACF DE RESIDUOS (SIN LAG 0)
# ============================================================================
print("\n4. Graficando ACF y PACF de residuos (sin lag 0)...")

# Calcular límites apropiados basados en los datos
acf_values = acf(residuos, nlags=30, fft=False)[1:]  # Sin lag 0
pacf_values = pacf(residuos, nlags=30)[1:]  # Sin lag 0

# Calcular límites de confianza (aproximadamente ±1.96/sqrt(n))
n = len(residuos)
conf_limit = 1.96 / np.sqrt(n)

# Determinar escala del eje Y
max_acf = max(abs(acf_values.max()), abs(acf_values.min()))
max_pacf = max(abs(pacf_values.max()), abs(pacf_values.min()))
y_limit = max(0.3, min(1.0, max(max_acf, max_pacf) * 1.3))  # Al menos 0.3, máximo 1.0

print(f"   Límite de confianza 95%: ±{conf_limit:.3f}")
print(f"   Escala del eje Y: ±{y_limit:.2f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ACF de residuos
plot_acf(residuos, lags=30, ax=axes[0], color='#2E86AB', alpha=0.7, zero=False)
axes[0].set_title(f'ACF de Residuos - ARIMA({p},{d},{q}) (sin lag 0)',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Rezagos', fontsize=11)
axes[0].set_ylabel('Autocorrelación', fontsize=11)
axes[0].set_ylim(-y_limit, y_limit)  # Escala dinámica
axes[0].axhline(y=conf_limit, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0].axhline(y=-conf_limit, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0].grid(True, alpha=0.3)

# PACF de residuos
plot_pacf(residuos, lags=30, ax=axes[1], color='#C73E1D', alpha=0.7, zero=False)
axes[1].set_title(f'PACF de Residuos - ARIMA({p},{d},{q}) (sin lag 0)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Rezagos', fontsize=11)
axes[1].set_ylabel('Autocorrelación Parcial', fontsize=11)
axes[1].set_ylim(-y_limit, y_limit)  # Escala dinámica
axes[1].axhline(y=conf_limit, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].axhline(y=-conf_limit, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].grid(True, alpha=0.3)

# Añadir información del modelo
textstr = f'AIC: {modelo_ajustado.aic:.2f}\nBIC: {modelo_ajustado.bic:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# DIAGNÓSTICO DE RESIDUOS
# ============================================================================
print("\n5. Diagnóstico completo de residuos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Gráfico de residuos
axes[0, 0].plot(residuos, linewidth=1, color='#2E86AB')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_title('Serie de Residuos', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Observación', fontsize=10)
axes[0, 0].set_ylabel('Residuo', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Histograma
axes[0, 1].hist(residuos, bins=20, color='#A23B72', alpha=0.7, edgecolor='black', density=True)
# Agregar curva normal
mu, sigma = residuos.mean(), residuos.std()
x = np.linspace(residuos.min(), residuos.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal teórica')
axes[0, 1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residuo', fontsize=10)
axes[0, 1].set_ylabel('Densidad', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# ACF de residuos
plot_acf(residuos, lags=20, ax=axes[1, 0], color='#F18F01', alpha=0.7, zero=False)
axes[1, 0].set_title('ACF de Residuos (sin lag 0)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(-0.3, 0.3)  # Limitar escala
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Diagnóstico de Residuos - ARIMA({p},{d},{q})',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show(block=False)

# ============================================================================
# VALORES AJUSTADOS (FITTED VALUES - ŷ)
# ============================================================================
print("\n6. Obteniendo valores ajustados (fitted values)...")

# Valores ajustados dentro de la muestra
fitted_values = modelo_ajustado.fittedvalues

print(f"   ✓ Valores ajustados obtenidos: {len(fitted_values)} observaciones")

# ============================================================================
# PREDICCIONES
# ============================================================================
print("\n7. Generando predicciones...")

n_forecast = 24  # Meses a predecir

# Predicción
forecast = modelo_ajustado.forecast(steps=n_forecast)
forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                               periods=n_forecast, freq='MS')

# Intervalos de confianza
forecast_result = modelo_ajustado.get_forecast(steps=n_forecast)
forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95%

print(f"\n   Predicciones para los próximos {n_forecast} meses:")
print(f"   {'Fecha':<12} {'Predicción':>12} {'IC Inf 95%':>12} {'IC Sup 95%':>12}")
print("   " + "-"*50)
for fecha, pred, ic_inf, ic_sup in zip(forecast_index, forecast,
                                        forecast_ci.iloc[:, 0],
                                        forecast_ci.iloc[:, 1]):
    print(f"   {fecha.strftime('%Y-%m'):<12} {pred:>12.2f} {ic_inf:>12.2f} {ic_sup:>12.2f}")

# ============================================================================
# TABLA COMPARATIVA: SERIE ORIGINAL VS AJUSTADOS VS PREDICCIONES
# ============================================================================
print("\n8. Generando tabla comparativa completa...")

# Crear tabla combinada
print(f"\n{'='*80}")
print("TABLA COMPARATIVA: VALORES ORIGINALES, AJUSTADOS Y PREDICCIONES")
print(f"{'='*80}\n")

print(f"{'Fecha':<12} {'Original':>12} {'Ajustado (ŷ)':>15} {'Residuo':>12}")
print("-"*55)

# Mostrar primeros 10 registros
for i in range(min(10, len(ts))):
    fecha = ts.index[i]
    original = ts.iloc[i]
    if i < len(fitted_values):
        ajustado = fitted_values.iloc[i]
        residuo = original - ajustado
        print(f"{fecha.strftime('%Y-%m'):<12} {original:>12.2f} {ajustado:>15.2f} {residuo:>12.2f}")
    else:
        print(f"{fecha.strftime('%Y-%m'):<12} {original:>12.2f} {'':>15} {'':>12}")

print("   ...")

# Mostrar últimos 10 registros históricos
for i in range(max(0, len(ts)-10), len(ts)):
    fecha = ts.index[i]
    original = ts.iloc[i]
    if i < len(fitted_values):
        ajustado = fitted_values.iloc[i]
        residuo = original - ajustado
        print(f"{fecha.strftime('%Y-%m'):<12} {original:>12.2f} {ajustado:>15.2f} {residuo:>12.2f}")
    else:
        print(f"{fecha.strftime('%Y-%m'):<12} {original:>12.2f} {'':>15} {'':>12}")

print("-"*55)
print(f"{'Fecha':<12} {'Original':>12} {'Predicción':>15} {'IC Inf 95%':>12} {'IC Sup 95%':>12}")
print("-"*68)

# Mostrar predicciones futuras
for fecha, pred, ic_inf, ic_sup in zip(forecast_index, forecast,
                                        forecast_ci.iloc[:, 0],
                                        forecast_ci.iloc[:, 1]):
    print(f"{fecha.strftime('%Y-%m'):<12} {'---':>12} {pred:>15.2f} {ic_inf:>12.2f} {ic_sup:>12.2f}")

print("="*80)

# ============================================================================
# GRÁFICO 1: SERIE ORIGINAL + VALORES AJUSTADOS + PREDICCIÓN
# ============================================================================
print("\n9. Graficando serie completa con valores ajustados y predicción...")

fig, ax = plt.subplots(figsize=(16, 7))

# Serie original
ax.plot(ts, linewidth=2, color='#2E86AB', label='Serie Original',
        marker='o', markersize=3, alpha=0.7, zorder=2)

# Valores ajustados (fitted values - ŷ)
ax.plot(fitted_values, linewidth=2.5, color='#F18F01',
        label='Valores Ajustados (ŷ)', linestyle='-', alpha=0.8, zorder=3)

# Predicción futura
ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA({p},{d},{q})', linestyle='--', zorder=4)

# Intervalo de confianza
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='IC 95%', zorder=1)

# Línea vertical
ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio de Predicción', zorder=1)

ax.set_title(f'Serie Original, Valores Ajustados y Predicción - ARIMA({p},{d},{q})',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Importaciones (toneladas)', fontsize=12)
ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)

# Información del modelo
textstr = f'ARIMA({p},{d},{q})\n\nAIC: {modelo_ajustado.aic:.2f}\nBIC: {modelo_ajustado.bic:.2f}\n\nAzul: Observado\nNaranja: Ajustado (ŷ)\nRojo: Predicción'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# GRÁFICO 2: PREDICCIÓN ESTÁNDAR (sin valores ajustados)
# ============================================================================
print("\n10. Graficando predicción estándar...")

fig, ax = plt.subplots(figsize=(16, 7))

# Serie histórica
ax.plot(ts, linewidth=2.5, color='#2E86AB', label='Serie Histórica', zorder=3)

# Predicción
ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA({p},{d},{q})', linestyle='--', zorder=4)

# Intervalo de confianza
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='Intervalo de Confianza 95%', zorder=2)

# Línea vertical
ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio de Predicción', zorder=1)

ax.set_title(f'Serie Temporal y Predicción - Modelo ARIMA({p},{d},{q})',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Importaciones (toneladas)', fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)

# Información del modelo
textstr = f'ARIMA({p},{d},{q})\n\nAIC: {modelo_ajustado.aic:.2f}\nBIC: {modelo_ajustado.bic:.2f}\n\nPredicciones: {n_forecast} meses'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# GRÁFICO 3: ZOOM (Últimos meses + predicción)
# ============================================================================
print("\n11. Graficando zoom en predicción...")

fig, ax = plt.subplots(figsize=(14, 6))

n_historicos = 36
ts_ultimos = ts[-n_historicos:]

ax.plot(ts_ultimos, linewidth=2.5, color='#2E86AB',
        label=f'Últimos {n_historicos} meses', marker='o', markersize=5, zorder=3)
ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA({p},{d},{q})', linestyle='--',
        marker='s', markersize=5, zorder=4)
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='IC 95%', zorder=2)

ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio Predicción', zorder=1)

ax.set_title(f'Detalle: Últimos Meses y Predicción - ARIMA({p},{d},{q})',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Importaciones (toneladas)', fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# GRÁFICO 4: ZOOM CON VALORES AJUSTADOS
# ============================================================================
print("\n12. Graficando zoom con valores ajustados...")

fig, ax = plt.subplots(figsize=(14, 6))

n_historicos = 36
ts_ultimos = ts[-n_historicos:]
fitted_ultimos = fitted_values[-n_historicos:]

ax.plot(ts_ultimos, linewidth=2, color='#2E86AB',
        label=f'Serie Original (últimos {n_historicos} meses)',
        marker='o', markersize=5, alpha=0.7, zorder=2)

ax.plot(fitted_ultimos, linewidth=2.5, color='#F18F01',
        label='Valores Ajustados (ŷ)', linestyle='-',
        marker='s', markersize=4, alpha=0.8, zorder=3)

ax.plot(forecast_index, forecast, linewidth=2.5, color='#C73E1D',
        label=f'Predicción ARIMA({p},{d},{q})', linestyle='--',
        marker='^', markersize=5, zorder=4)

ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='#C73E1D', alpha=0.2,
                label='IC 95%', zorder=1)

ax.axvline(x=ts.index[-1], color='gray', linestyle=':', linewidth=2,
           alpha=0.7, label='Inicio Predicción', zorder=1)

ax.set_title(f'Detalle: Original, Ajustado (ŷ) y Predicción - ARIMA({p},{d},{q})',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Importaciones (toneladas)', fontsize=12)
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"\nModelo: ARIMA({p}, {d}, {q})")
print(f"AIC: {modelo_ajustado.aic:.4f}")
print(f"BIC: {modelo_ajustado.bic:.4f}")
print(f"Log-Likelihood: {modelo_ajustado.llf:.4f}")
print(f"\nObservaciones usadas: {len(ts)}")
print(f"Predicciones generadas: {n_forecast} meses")
print(f"Rango de predicción: {forecast.min():.2f} - {forecast.max():.2f}")
print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print("\nPara probar otro modelo, cambia los valores de p, d, q al inicio del script")
print("Presiona Enter para cerrar los gráficos...")
input()
plt.close('all')
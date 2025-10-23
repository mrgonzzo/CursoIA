import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wooldridge as wd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings

# Configuración
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Carga de Datos y Preprocesamiento
# ====================================

# Cargar el dataset 'barium'
data = wd.data('barium')

# Usaremos la variable chnexp: exportaciones chinas de óxido de bario.
ts = data['chnexp'].copy()
# Asignar índice de tiempo anual
# Asumimos que la serie es anual (1978 a 1988), la librería wooldridge ya tiene el orden correcto
ts.index = pd.to_datetime(data.index + 1977, format='%Y')

# Dividir la serie en entrenamiento y prueba (ej. 80% para entrenamiento)
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

print(f"Serie de tiempo cargada: {ts.name} (Exportaciones de Bario)")
print(f"Período total: {ts.index.min().year} a {ts.index.max().year}")
print(f"Período de entrenamiento: {len(train)} observaciones")
print(f"Período de prueba (predicción): {len(test)} observaciones")

# 2. Análisis de Estacionariedad y Determinación de 'd'
# ======================================================

# Usar auto_arima para encontrar el valor óptimo de 'd' (diferenciación)
# m=1 porque son datos anuales (sin estacionalidad).
stepwise_fit = auto_arima(train, seasonal=False, m=1,
                          start_p=0, max_p=3, start_q=0, max_q=3,
                          d=None, max_d=2, trace=False, stepwise=True,
                          suppress_warnings=True, error_action='ignore')

d_optimo = stepwise_fit.order[1]
print(f"\nOrden de diferenciación (d) determinado por auto_arima: {d_optimo}")

# 3. Determinación de 'p' y 'q' (Análisis ACF/PACF)
# =================================================

# Aplicar la diferenciación óptima
if d_optimo > 0:
    ts_diff = train.diff(d_optimo).dropna()
else:
    ts_diff = train

# Gráficos ACF y PACF para determinar p y q
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_diff, lags=len(ts_diff) - 1, ax=ax[0], title=f"ACF (d={d_optimo})")
plot_pacf(ts_diff, lags=len(ts_diff) - 1, ax=ax[1], title=f"PACF (d={d_optimo})")
plt.suptitle('Diagnóstico de Correlación de Residuos Diferenciados')
plt.show()

# 4. Ranking de Modelos (Grid Search y AIC)
# =========================================

# Rango de parámetros a probar
p_values = range(3)  # p = [0, 1, 2]
q_values = range(3)  # q = [0, 1, 2]
d = d_optimo  # Usar el 'd' óptimo

model_rank = []

# Búsqueda de modelos
for p in p_values:
    for q in q_values:
        if p == 0 and q == 0:
            continue
        try:
            # Ajustar el modelo ARIMA
            model = ARIMA(train, order=(p, d, q), freq='A')
            model_fit = model.fit()

            # Guardar el resultado
            model_rank.append({
                'Modelo': f'ARIMA({p}, {d}, {q})',
                'AIC': model_fit.aic
            })
        except Exception as e:
            # print(f"Error ajustando ARIMA({p}, {d}, {q}): {e}")
            continue

# Crear un DataFrame con el ranking
df_ranking = pd.DataFrame(model_rank).sort_values(by='AIC', ascending=True)
df_ranking.reset_index(drop=True, inplace=True)

# Seleccionar el mejor modelo basado en el AIC
if df_ranking.empty:
    raise ValueError("No se pudo ajustar ningún modelo ARIMA válido.")

best_model_name = df_ranking.iloc[0]['Modelo']
best_p, best_d, best_q = map(int, best_model_name[6:-1].split(', '))

print("\n--- Ranking de Modelos por AIC ---")
print(df_ranking.head(5))
print(f"\nEl mejor modelo (AIC más bajo) es: {best_model_name}")

# 5. Ajuste del Mejor Modelo y Predicción
# =======================================

# Ajustar el mejor modelo a la serie completa (para predicción)
# Nota: statsmodels usa todo el 'ts' para predecir si no se especifica 'start/end'
best_model = ARIMA(ts, order=(best_p, best_d, best_q), freq='A')
best_model_fit = best_model.fit()

# Definir el horizonte de predicción (el período de prueba)
start_pred = test.index[0]
end_pred = test.index[-1]

# Realizar la predicción y obtener los intervalos de confianza
forecast = best_model_fit.get_prediction(start=start_pred, end=end_pred)
forecast_df = forecast.predicted_mean.rename('Predicción')
conf_int = forecast.conf_int(alpha=0.05)
conf_int.columns = ['Límite Inferior (95%)', 'Límite Superior (95%)']

# 6. Gráfico de Serie Original, Predicción y Bandas de Confianza
# ==============================================================

plt.figure(figsize=(14, 6))

# Serie histórica
plt.plot(ts, label='Serie Original (chnexp)', color='blue', marker='o')

# Separador de entrenamiento/prueba
plt.axvline(train.index[-1], color='red', linestyle='--', label='Fin de Entrenamiento (Inicio Predicción)')

# Predicción
plt.plot(forecast_df, label=f'Predicción {best_model_name}', color='green', marker='x')

# Bandas de confianza
plt.fill_between(conf_int.index, conf_int['Límite Inferior (95%)'], conf_int['Límite Superior (95%)'],
                 color='green', alpha=0.1, label='Banda de Confianza 95%')

# Puntos de la serie real para el periodo de prueba
plt.scatter(test.index, test, marker='s', color='orange', s=50, label='Valores Reales (Test)')

plt.title(f'Pronóstico de Exportaciones de Bario con el Modelo {best_model_name}')
plt.xlabel('Año')
plt.ylabel('Exportaciones Chinas de Óxido de Bario (chnexp)')
plt.legend()
plt.show()
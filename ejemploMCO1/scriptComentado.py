# === IMPORTACIÓN DE LIBRERÍAS ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# === CONFIGURACIÓN DE GRÁFICOS Y SEMILLA ===
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
np.random.seed(42)

print("=== ANÁLISIS DE REGRESIÓN LINEAL MÚLTIPLE ===\n")

# === 1. GENERACIÓN DE DATOS SIMULADOS ===
print("1. GENERANDO DATOS SIMULADOS...")
n = 200
ingresos = np.random.normal(50000, 15000, n)
experiencia = np.random.normal(10, 4, n)
educacion = np.random.normal(15, 3, n)

# Fórmula poblacional con ruido
precio_real = (20000 +
              300 * ingresos/1000 +
              1500 * experiencia +
              800 * educacion +
              np.random.normal(0, 5000, n))

# Crear DataFrame
df = pd.DataFrame({
    'precio_casa': precio_real,
    'ingresos': ingresos,
    'experiencia': experiencia,
    'educacion': educacion
})

print(f"Dataset creado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(df.head().round(2))

# === 2. ANÁLISIS EXPLORATORIO ===
print("\n2. ANÁLISIS EXPLORATORIO...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histograma del precio
axes[0,0].hist(df['precio_casa'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribución del Precio de Casas')
axes[0,0].set_xlabel('Precio ($)')
axes[0,0].set_ylabel('Frecuencia')

# Dispersión entre variables independientes y precio
variables = ['ingresos', 'experiencia', 'educacion']
colors = ['red', 'blue', 'green']
for i, var in enumerate(variables):
    axes[0,1].scatter(df[var], df['precio_casa'], alpha=0.6, color=colors[i], label=var)
axes[0,1].set_title('Relación entre Variables Independientes y Precio')
axes[0,1].set_xlabel('Variables Independientes')
axes[0,1].set_ylabel('Precio Casa ($)')
axes[0,1].legend()

# Matriz de correlación
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title('Matriz de Correlación')

# Boxplot de variables independientes
df_boxplot = df.drop('precio_casa', axis=1)
sns.boxplot(data=df_boxplot, ax=axes[1,1])
axes[1,1].set_title('Distribución de Variables Independientes')

plt.tight_layout()
plt.show()

# === 3. MODELO DE REGRESIÓN LINEAL MÚLTIPLE ===
print("\n3. ESTIMACIÓN DEL MODELO MCO MÚLTIPLE...")

X = df[['ingresos', 'experiencia', 'educacion']]
X = sm.add_constant(X)  # Añadir intercepto
y = df['precio_casa']

modelo_sm = sm.OLS(y, X).fit()

print("=== RESUMEN DEL MODELO ===")
print(modelo_sm.summary())

# === 4. MÉTRICAS DE BONDAD DE AJUSTE ===
print("\n4. MÉTRICAS DE BONDAD DE AJUSTE")

y_pred = modelo_sm.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
r2 = modelo_sm.rsquared
r2_ajustado = modelo_sm.rsquared_adj

print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_ajustado:.4f}")
print(f"RECM (RMSE): {rmse:.2f} $")
print(f"EMA (MAE): {mae:.2f} $")
print(f"PME (MAPE): {mape:.2f} %")

# === 5. ANÁLISIS DE RESIDUOS ===
print("\n5. ANÁLISIS DE RESIDUOS...")

residuos = y - y_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0,0].scatter(y_pred, residuos, alpha=0.6, color='purple')
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_xlabel('Valores Predichos')
axes[0,0].set_ylabel('Residuos')
axes[0,0].set_title('Residuos vs Valores Predichos')

sm.qqplot(residuos, line='45', ax=axes[0,1])
axes[0,1].set_title('Q-Q Plot de Residuos')

axes[1,0].hist(residuos, bins=20, alpha=0.7, color='orange', edgecolor='black')
axes[1,0].set_xlabel('Residuos')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Distribución de Residuos')

axes[1,1].plot(range(len(residuos)), residuos, 'o-', alpha=0.6, color='green')
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_xlabel('Orden de Observación')
axes[1,1].set_ylabel('Residuos')
axes[1,1].set_title('Residuos vs Orden de Observación')

plt.tight_layout()
plt.show()

# === 6. COMPARACIÓN VISUAL: REALES vs PREDICHOS ===
print("\n6. COMPARACIÓN: VALORES REALES vs PREDICHOS")

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.6, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales ($)')
plt.ylabel('Valores Predichos ($)')
plt.title('Valores Reales vs Predichos - MCO Múltiple')
plt.grid(True, alpha=0.3)

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea Perfecta')

plt.legend()
plt.show()

# === 7. INTERPRETACIÓN ECONÓMICA ===
print("\n7. INTERPRETACIÓN ECONÓMICA DEL MODELO")
print("="*50)

coeficientes = modelo_sm.params
std_errors = modelo_sm.bse

print("Coeficientes estimados:")
for i, var in enumerate(['Intercepto', 'Ingresos', 'Experiencia', 'Educación']):
    print(f"{var:12}: {coeficientes[i]:>8.2f} ± {std_errors[i]:.2f}")

print("\nInterpretación:")
print(f"- Por cada $1000 adicional en ingresos, el precio aumenta ${coeficientes[1]*1000:.2f}")
print(f"- Por cada año adicional de experiencia, el precio aumenta ${coeficientes[2]:.2f}")
print(f"- Por cada año adicional de educación, el precio aumenta ${coeficientes[3]:.2f}")

# === 8. COMPARACIÓN DE MÉTRICAS CON OUTLIERS ===
print("\n8. COMPARACIÓN DE MÉTRICAS DE ERROR")
print("="*40)

y_outlier = y.copy()
y_outlier[0] = y_outlier[0] * 3  # Introducir outlier
y_pred_outlier = y_pred.copy()

rmse_outlier = np.sqrt(mean_squared_error(y_outlier, y_pred_outlier))
mae_outlier = mean_absolute_error(y_outlier, y_pred_outlier)

print("Efecto de outliers en las métricas:")
print(f"RECM sin outlier: {rmse:.2f}")
print(f"RECM con outlier: {rmse_outlier:.2f}")
print(f"Incremento RECM: {((rmse_outlier-rmse)/rmse)*100:.1f}%")
print(f"EMA sin outlier: {mae:.2f}")
print(f"EMA con outlier: {mae_outlier:.2f}")
print(f"Incremento EMA: {((mae_outlier-mae)/mae)*100:.1f}%")

print("\n" + "="*60
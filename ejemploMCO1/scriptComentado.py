# === IMPORTACIÓN DE LIBRERÍAS ===
# numpy: Operaciones matemáticas y generación de números aleatorios
import numpy as np
# pandas: Manipulación y análisis de datos estructurados
import pandas as pd
# matplotlib: Visualización y creación de gráficos
import matplotlib.pyplot as plt
# seaborn: Visualización estadística basada en matplotlib
import seaborn as sns
# sklearn: Machine learning, métricas y preprocesamiento
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
# statsmodels: Modelos estadísticos y pruebas econométricas
import statsmodels.api as sm

# === CONFIGURACIÓN DE GRÁFICOS Y SEMILLA ===
# Configurar tamaño por defecto de las figuras (ancho, alto)
plt.rcParams['figure.figsize'] = (12, 8)
# Establecer estilo de seaborn con grid blanco
sns.set_style("whitegrid")
# Fijar semilla para reproducibilidad de resultados aleatorios
np.random.seed(42)

# Mensaje inicial del programa
print("=== ANÁLISIS DE REGRESIÓN LINEAL MÚLTIPLE ===\n")

# === 1. GENERACIÓN DE DATOS SIMULADOS ===
print("1. GENERANDO DATOS SIMULADOS...")
# Número de observaciones en el dataset
n = 200
# Generar ingresos con distribución normal (media=50000, desv=15000)
ingresos = np.random.normal(50000, 15000, n)
# Generar experiencia laboral con distribución normal (media=10, desv=4)
experiencia = np.random.normal(10, 4, n)
# Generar años de educación con distribución normal (media=15, desv=3)
educacion = np.random.normal(15, 3, n)

# Fórmula poblacional con ruido (relación lineal verdadera)
# Precio base + efecto ingresos + efecto experiencia + efecto educación + ruido aleatorio
precio_real = (20000 +
              300 * ingresos/1000 +  # Coeficiente de 300 por cada $1000 de ingresos
              1500 * experiencia +    # Coeficiente de 1500 por año de experiencia
              800 * educacion +       # Coeficiente de 800 por año de educación
              np.random.normal(0, 5000, n))  # Término de error aleatorio

# Crear DataFrame con las variables generadas
df = pd.DataFrame({
    'precio_casa': precio_real,  # Variable dependiente
    'ingresos': ingresos,        # Variable independiente 1
    'experiencia': experiencia,  # Variable independiente 2
    'educacion': educacion       # Variable independiente 3
})

# Mostrar información del dataset y primeras filas
print(f"Dataset creado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(df.head().round(2))

# === 2. ANÁLISIS EXPLORATORIO ===
print("\n2. ANÁLISIS EXPLORATORIO...")

# Crear figura con 2x2 subgráficos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- Subgráfico 1: Histograma del precio de casas ---
axes[0,0].hist(df['precio_casa'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribución del Precio de Casas')
axes[0,0].set_xlabel('Precio ($)')
axes[0,0].set_ylabel('Frecuencia')

# --- Subgráfico 2: Dispersión entre variables independientes y precio ---
variables = ['ingresos', 'experiencia', 'educacion']  # Lista de variables predictoras
colors = ['red', 'blue', 'green']  # Colores para cada variable
# Graficar scatter plots para cada variable independiente
for i, var in enumerate(variables):
    axes[0,1].scatter(df[var], df['precio_casa'], alpha=0.6, color=colors[i], label=var)
axes[0,1].set_title('Relación entre Variables Independientes y Precio')
axes[0,1].set_xlabel('Variables Independientes')
axes[0,1].set_ylabel('Precio Casa ($)')
axes[0,1].legend()  # Mostrar leyenda

# --- Subgráfico 3: Matriz de correlación ---
# Calcular matriz de correlación entre todas las variables
corr_matrix = df.corr()
# Crear heatmap de correlaciones con anotaciones
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title('Matriz de Correlación')

# --- Subgráfico 4: Boxplot de variables independientes ---
# Excluir variable dependiente para el boxplot
df_boxplot = df.drop('precio_casa', axis=1)
# Crear boxplot para visualizar distribución y outliers
sns.boxplot(data=df_boxplot, ax=axes[1,1])
axes[1,1].set_title('Distribución de Variables Independientes')

# Ajustar espaciado entre subgráficos
plt.tight_layout()
# Mostrar todos los gráficos
plt.show()

# === 3. MODELO DE REGRESIÓN LINEAL MÚLTIPLE ===
print("\n3. ESTIMACIÓN DEL MODELO MCO MÚLTIPLE...")

# Preparar matriz de características (variables independientes)
X = df[['ingresos', 'experiencia', 'educacion']]
# Añadir columna de unos para el término constante (intercepto)
X = sm.add_constant(X)
# Variable dependiente
y = df['precio_casa']

# Crear y ajustar modelo de Mínimos Cuadrados Ordinarios
modelo_sm = sm.OLS(y, X).fit()

# Mostrar resumen completo del modelo
print("=== RESUMEN DEL MODELO ===")
print(modelo_sm.summary())

# === 4. MÉTRICAS DE BONDAD DE AJUSTE ===
print("\n4. MÉTRICAS DE BONDAD DE AJUSTE")

# Predecir valores usando el modelo ajustado
y_pred = modelo_sm.predict(X)

# Calcular métricas de error
mse = mean_squared_error(y, y_pred)  # Error Cuadrático Medio
rmse = np.sqrt(mse)                  # Raíz del Error Cuadrático Medio
mae = mean_absolute_error(y, y_pred) # Error Absoluto Medio
mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Error Porcentual Absoluto Medio
r2 = modelo_sm.rsquared              # Coeficiente de determinación
r2_ajustado = modelo_sm.rsquared_adj # R² ajustado por número de predictores

# Mostrar métricas formateadas
print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_ajustado:.4f}")
print(f"RECM (RMSE): {rmse:.2f} $")
print(f"EMA (MAE): {mae:.2f} $")
print(f"PME (MAPE): {mape:.2f} %")

# === 5. ANÁLISIS DE RESIDUOS ===
print("\n5. ANÁLISIS DE RESIDUOS...")

# Calcular residuos (diferencia entre valores reales y predichos)
residuos = y - y_pred

# Crear figura para análisis de residuos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- Subgráfico 1: Residuos vs Valores Predichos ---
axes[0,0].scatter(y_pred, residuos, alpha=0.6, color='purple')
axes[0,0].axhline(y=0, color='red', linestyle='--')  # Línea en y=0
axes[0,0].set_xlabel('Valores Predichos')
axes[0,0].set_ylabel('Residuos')
axes[0,0].set_title('Residuos vs Valores Predichos')

# --- Subgráfico 2: Q-Q Plot de Residuos ---
# Gráfico de cuantiles para verificar normalidad de residuos
sm.qqplot(residuos, line='45', ax=axes[0,1])
axes[0,1].set_title('Q-Q Plot de Residuos')

# --- Subgráfico 3: Histograma de Residuos ---
axes[1,0].hist(residuos, bins=20, alpha=0.7, color='orange', edgecolor='black')
axes[1,0].set_xlabel('Residuos')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].set_title('Distribución de Residuos')

# --- Subgráfico 4: Residuos vs Orden de Observación ---
axes[1,1].plot(range(len(residuos)), residuos, 'o-', alpha=0.6, color='green')
axes[1,1].axhline(y=0, color='red', linestyle='--')  # Línea en y=0
axes[1,1].set_xlabel('Orden de Observación')
axes[1,1].set_ylabel('Residuos')
axes[1,1].set_title('Residuos vs Orden de Observación')

# Ajustar y mostrar gráficos de residuos
plt.tight_layout()
plt.show()

# === 6. COMPARACIÓN VISUAL: REALES vs PREDICHOS ===
print("\n6. COMPARACIÓN: VALORES REALES vs PREDICHOS")

# Crear figura para comparación
plt.figure(figsize=(10, 6))
# Graficar valores reales vs predichos
plt.scatter(y, y_pred, alpha=0.6, color='teal')
# Línea de perfecta predicción (y = x)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales ($)')
plt.ylabel('Valores Predichos ($)')
plt.title('Valores Reales vs Predichos - MCO Múltiple')
plt.grid(True, alpha=0.3)

# Calcular límites para la línea de perfecta predicción
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea Perfecta')

plt.legend()
plt.show()

# === 7. INTERPRETACIÓN ECONÓMICA ===
print("\n7. INTERPRETACIÓN ECONÓMICA DEL MODELO")
print("="*50)

# Extraer coeficientes y errores estándar del modelo
coeficientes = modelo_sm.params
std_errors = modelo_sm.bse

# Mostrar coeficientes con formato
print("Coeficientes estimados:")
for i, var in enumerate(['Intercepto', 'Ingresos', 'Experiencia', 'Educación']):
    print(f"{var:12}: {coeficientes[i]:>8.2f} ± {std_errors[i]:.2f}")

# Interpretación económica de los coeficientes
print("\nInterpretación:")
print(f"- Por cada $1000 adicional en ingresos, el precio aumenta ${coeficientes[1]*1000:.2f}")
print(f"- Por cada año adicional de experiencia, el precio aumenta ${coeficientes[2]:.2f}")
print(f"- Por cada año adicional de educación, el precio aumenta ${coeficientes[3]:.2f}")

# === 8. COMPARACIÓN DE MÉTRICAS CON OUTLIERS ===
print("\n8. COMPARACIÓN DE MÉTRICAS DE ERROR")
print("="*40)

# Crear copia de los datos reales e introducir un outlier artificial
y_outlier = y.copy()
y_outlier[0] = y_outlier[0] * 3  # Multiplicar primera observación por 3
y_pred_outlier = y_pred.copy()   # Mantener mismas predicciones

# Calcular métricas con outlier
rmse_outlier = np.sqrt(mean_squared_error(y_outlier, y_pred_outlier))
mae_outlier = mean_absolute_error(y_outlier, y_pred_outlier)

# Mostrar comparación de métricas con y sin outlier
print("Efecto de outliers en las métricas:")
print(f"RECM sin outlier: {rmse:.2f}")
print(f"RECM con outlier: {rmse_outlier:.2f}")
print(f"Incremento RECM: {((rmse_outlier-rmse)/rmse)*100:.1f}%")
print(f"EMA sin outlier: {mae:.2f}")
print(f"EMA con outlier: {mae_outlier:.2f}")
print(f"Incremento EMA: {((mae_outlier-mae)/mae)*100:.1f}%")

# Línea final del programa
print("\n" + "="*60)

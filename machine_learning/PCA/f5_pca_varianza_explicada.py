import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# =============================
# 1. Cargar datos
# =============================
df = pd.read_csv('USArrests.csv')  # Cambia por tu CSV real

# =============================
# 2. Separar variables predictoras
# =============================
X = df.drop('USArrests', axis=1)
y = df['USArrests']

# =============================
# 3. Crear y entrenar pipeline PCA
# =============================
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=min(X.shape[0], X.shape[1]))
)

X_pca = pca_pipeline.fit_transform(X)
pca_model = pca_pipeline.named_steps['pca']

# =============================
# 4. Varianza explicada individual
# =============================
varianza_individual = pca_model.explained_variance_ratio_
print("=== VARIANZA EXPLICADA INDIVIDUAL POR COMPONENTE ===")
for i, var in enumerate(varianza_individual):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print("\nInterpretación:")
print("Cada PC captura un porcentaje de la varianza total de los datos. "
      "Valores mayores indican que ese componente resume más información de los datos originales.\n")

# =============================
# 5. Varianza acumulada
# =============================
varianza_acumulada = np.cumsum(varianza_individual)
print("=== VARIANZA ACUMULADA ===")
for i, var in enumerate(varianza_acumulada):
    print(f"PC1 a PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print("\nInterpretación:")
print("La varianza acumulada indica cuánta información total se conserva usando los primeros k componentes. "
      "Por ejemplo, si alcanzamos 80% de varianza con 2 componentes, estos 2 son suficientes para reducir dimensionalidad sin perder información significativa.\n")

# =============================
# 6. Gráfico combinado: varianza individual y acumulada
# =============================
plt.figure(figsize=(8,5))
plt.bar(range(1, len(varianza_individual)+1), varianza_individual, alpha=0.6, label='Varianza individual')
plt.plot(range(1, len(varianza_individual)+1), varianza_acumulada, marker='o', color='red', label='Varianza acumulada')
plt.axhline(y=0.8, color='green', linestyle='--', label='80% umbral')
plt.xlabel('Componentes principales')
plt.ylabel('Varianza explicada')
plt.title('Varianza individual y acumulada')
plt.xticks(range(1, len(varianza_individual)+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# =============================
# 7. Scree plot (Regla del codo)
# =============================
plt.figure(figsize=(7,4))
plt.plot(range(1, len(varianza_individual)+1), varianza_individual, marker='o', linestyle='-')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada')
plt.title('Scree Plot (Regla del codo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(range(1, len(varianza_individual)+1))
plt.show()

print("Interpretación de la regla del codo:")
print("El codo se produce en el punto donde la pendiente del Scree Plot se suaviza notablemente. "
      "Ese número de componentes es suficiente, ya que agregar más aporta poca varianza adicional.")

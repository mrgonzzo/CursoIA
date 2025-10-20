from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# EJERCICIOS CON DATASETS REALES INTEGRADOS
# ============================================================================

# 1. INSTALAR (una sola vez en terminal):
# pip install scikit-learn seaborn pandas matplotlib numpy

# ============================================================================
# EJERCICIO 1: PCA - Dataset Iris (Clasificación de flores)
# ============================================================================

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Cargar dataset Iris (150 flores, 4 características: largo/ancho sépalo y pétalo)
iris = load_iris()
X = iris.data
y = iris.target
nombres_especies = iris.target_names

print("=" * 60)
print("EJERCICIO 1: PCA - Iris Dataset")
print("=" * 60)
print(f"Datos: {X.shape[0]} flores, {X.shape[1]} características")
print(f"Especies: {', '.join(nombres_especies)}\n")

# Estandarizar
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# PCA: reducir de 4 a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_escalado)

print(f"Varianza explicada: {pca.explained_variance_ratio_}")
print(f"Varianza total: {sum(pca.explained_variance_ratio_):.1%}\n")

# Visualizar
plt.figure(figsize=(8, 6))
colores = ['red', 'green', 'blue']
for i, (especie, color) in enumerate(zip(nombres_especies, colores)):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=especie, c=color, s=100, alpha=0.6)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: Iris - 4D → 2D')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



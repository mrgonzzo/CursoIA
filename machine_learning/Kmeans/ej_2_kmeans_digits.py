from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# EJERCICIO 2: K-Means - Digits Dataset (Reconocimiento de dígitos)
# ============================================================================

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

print("\n" + "=" * 60)
print("EJERCICIO 2: K-Means - Handwritten Digits")
print("=" * 60)

# Cargar dataset: 1797 imágenes de dígitos (0-9), 8x8 píxeles
digits = load_digits()
X = digits.data
y = digits.target

print(f"Datos: {X.shape[0]} imágenes, {X.shape[1]} píxeles cada una")
print(f"Dígitos: {sorted(set(y))}\n")

# Reducir dimensionalidad primero (64 → 10 para visualizar mejor)
pca_reduccion = PCA(n_components=10)
X_reducido = pca_reduccion.fit_transform(X)

# K-Means: buscar 10 clusters (uno por dígito)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_reducido)

print(f"Inercia (suma distancias intra-cluster): {kmeans.inertia_:.2f}")
print(f"Tamaño clusters: {pd.Series(labels).value_counts().sort_index().to_dict()}\n")

# Visualizar algunas imágenes
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    imagen = X[i].reshape(8, 8)
    ax.imshow(imagen, cmap='gray')
    ax.set_title(f'Dígito: {y[i]}, Cluster: {labels[i]}')
    ax.axis('off')

plt.suptitle('Dataset Digits: Primeras 10 imágenes')
plt.tight_layout()
plt.show()
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# K-MEANS - Ejemplo 3: wine dataset
# ============================================================================
print("=" * 70)
print("K-MEANS - EJEMPLO 3: wine dataset")
print("Agrupamiento de vinos con validación de clusters")
print("=" * 70)

# Cargar dataset: 150 flores de 3 especies con 4 características
wine = load_wine()
X = wine.data
y_true = wine.target  # Etiquetas reales (solo para comparación)
feature_names = wine.feature_names
target_names = wine.target_names

print(f"\nDatos: {X.shape[0]} vinos")
print(f"Características: {feature_names}")
print(f"Especies reales: {', '.join(target_names)}")
print(f"Distribución real: {pd.Series(y_true).value_counts().sort_index().to_dict()}\n")

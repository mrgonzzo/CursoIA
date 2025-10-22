from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------------------------
# 1. CARGAR Y ESCALAR DATOS
# ----------------------------------------------------------------------------
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Usamos solo 30 muestras para que el dendrograma sea legible
X_muestra = X_scaled[:30]

# Lista de métodos de linkage a comparar
metodos = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

# Directorio para guardar imágenes
if not os.path.exists('dendrogramas'):
    os.makedirs('dendrogramas')

# Corte de distancia para colorear clusters
corte_dist = 5.0  # puedes ajustar según el dataset

# Diccionario con explicación breve de cada método
explicaciones = {
    'single': "Distancia mínima entre clusters; genera clusters alargados y sensibles a outliers.",
    'complete': "Distancia máxima entre clusters; produce clusters compactos y uniformes.",
    'average': "Distancia promedio entre todos los pares de puntos; intermedio entre single y complete.",
    'weighted': "Promedia distancias ponderando clusters por tamaño; similar a average.",
    'centroid': "Distancia entre centroides de clusters; puede dar fusiones inesperadas.",
    'median': "Similar a centroid pero usando mediana; se comporta intermedia entre centroid y ward.",
    'ward': "Minimiza la varianza dentro del cluster; clusters muy compactos y esféricos."
}

# ----------------------------------------------------------------------------
# 2. GENERAR DENDROGRAMAS COLOREADOS
# ----------------------------------------------------------------------------
for method in metodos:
    print(f"\nMétodo: {method.upper()}")
    print(f"Descripción: {explicaciones[method]}")

    # Calcular linkage
    Z = linkage(X_muestra, method=method)

    # Obtener clusters según corte de distancia
    clusters = fcluster(Z, t=corte_dist, criterion='distance')
    n_clusters = len(np.unique(clusters))
    print(f"Número de clusters estimados con corte {corte_dist}: {n_clusters}")

    # Crear figura
    plt.figure(figsize=(12, 6))

    dendrogram(
        Z,
        labels=[f'V{i}' for i in range(30)],
        color_threshold=corte_dist,
        above_threshold_color='gray'
    )
    plt.title(f'Dendrograma Wine - Método: {method}')
    plt.xlabel('Vino')
    plt.ylabel('Distancia')
    plt.axhline(y=corte_dist, color='red', linestyle='--', alpha=0.7, label='Corte de distancia')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Guardar PNG
    filename = f'dendrogramas/dendrograma_{method}.png'
    plt.savefig(filename)
    plt.close()

    print(f"Dendrograma guardado como: {filename}")

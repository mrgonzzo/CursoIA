from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# EJERCICIO 8: K-Means - Olivetti Faces Dataset (Reconocimiento de rostros)
# ============================================================================

print("\n" + "=" * 60)
print("EJERCICIO 8: K-Means - Olivetti Faces Dataset")
print("=" * 60)

# Cargar dataset: 400 imágenes de rostros de 40 personas diferentes
# Cada imagen es de 64x64 píxeles
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data
y = faces.target

print(f"Datos: {X.shape[0]} imágenes de rostros")
print(f"Dimensión de cada imagen: {X.shape[1]} píxeles (64x64)")
print(f"Número de personas diferentes: {len(np.unique(y))}\n")

# Reducir dimensionalidad primero (4096 → 50 dimensiones)
print("Aplicando PCA para reducir dimensionalidad...")
pca = PCA(n_components=50, whiten=True, random_state=42)
X_pca = pca.fit_transform(X)
print(f"Varianza explicada con 50 componentes: {sum(pca.explained_variance_ratio_):.2%}\n")

# Aplicar K-Means para agrupar rostros
# Usamos 10 clusters para agrupar las 40 personas en 10 grupos
n_clusters = 10
print(f"Aplicando K-Means con {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_pca)

print(f"Inercia (calidad del clustering): {kmeans.inertia_:.2f}")
print(f"\nDistribución de imágenes por cluster:")
distribucion = pd.Series(cluster_labels).value_counts().sort_index()
for cluster, count in distribucion.items():
    print(f"  Cluster {cluster}: {count} rostros")

# Visualizar resultados
fig = plt.figure(figsize=(16, 8))

# Plot 1: Mostrar rostros representativos de cada cluster (centroides)
print("\nVisualizando centroides de cada cluster...")
for i in range(n_clusters):
    # Encontrar el rostro más cercano al centroide del cluster
    mask = cluster_labels == i
    cluster_faces = X_pca[mask]

    if len(cluster_faces) > 0:
        # Calcular distancias al centroide
        distances = np.linalg.norm(cluster_faces - kmeans.cluster_centers_[i], axis=1)
        idx_closest = np.where(mask)[0][np.argmin(distances)]

        # Mostrar la imagen
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(X[idx_closest].reshape(64, 64), cmap='gray')
        ax.set_title(f'Cluster {i}\n({distribucion[i]} rostros)')
        ax.axis('off')

plt.suptitle('K-Means: Rostros representativos de cada cluster', fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# Plot 2: Visualizar distribución en 2D (primeros 2 componentes de PCA)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot con clusters
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                      cmap='tab10', s=30, alpha=0.6)
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2,
           label='Centroides')
ax1.set_xlabel('Primera Componente Principal')
ax1.set_ylabel('Segunda Componente Principal')
ax1.set_title('Distribución de Rostros en 2D (coloreado por cluster)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot con personas reales (para comparar)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                       cmap='tab20', s=30, alpha=0.6)
ax2.set_xlabel('Primera Componente Principal')
ax2.set_ylabel('Segunda Componente Principal')
ax2.set_title('Distribución de Rostros en 2D (coloreado por persona real)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nNota: Compara ambos gráficos para ver cómo K-Means agrupa los rostros")
print("vs. las etiquetas reales de las personas.")
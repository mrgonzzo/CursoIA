from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# EJERCICIO: K-MEANS + RADAR CHART - Wine Quality Dataset
# ============================================================================

print("=" * 60)
print("EJERCICIO: K-MEANS + RADAR CHART - Wine Quality Dataset")
print("=" * 60)

# Cargar dataset Wine Quality (vinos)
wine_data = load_wine()
X = wine_data.data
y_true = wine_data.target
feature_names = wine_data.feature_names

print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características\n")

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Añadir resultados a DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['Cluster'] = y_kmeans

# Mostrar resumen
print("Centroides del modelo K-Means (en espacio estandarizado):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=feature_names).head())
print("\nDistribución de muestras por cluster:")
print(df['Cluster'].value_counts(), "\n")

# ============================================================================
# VISUALIZACIÓN PCA 2D (para referencia)
# ============================================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
colores = ['red', 'green', 'blue']
for i in range(n_clusters):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1],
                c=colores[i], s=60, label=f'Cluster {i}', alpha=0.6)
plt.title("K-Means (Wine Dataset) - PCA 2D Projection")
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# GRÁFICO TIPO TELARAÑA (Radar Chart)
# ============================================================================

print("Generando gráfico tipo telaraña (perfil químico por cluster)...")

# Calcular la media de cada característica por cluster
mean_por_cluster = df.groupby('Cluster').mean()

# Seleccionar algunas variables (para que sea legible)
variables = feature_names[:6]  # puedes ajustar el número
num_vars = len(variables)

# Calcular ángulos del círculo
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el círculo

# Crear figura polar
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, cluster in enumerate(mean_por_cluster.index):
    valores = mean_por_cluster.loc[cluster, variables].values.tolist()
    valores += valores[:1]  # cerrar la figura
    ax.plot(angles, valores, color=colores[i % len(colores)],
            linewidth=2, label=f'Cluster {cluster}')
    ax.fill(angles, valores, color=colores[i % len(colores)], alpha=0.25)

# Configuración estética
ax.set_xticks(angles[:-1])
ax.set_xticklabels(variables, fontsize=10)
ax.set_title("Perfil químico medio por Cluster (K-Means)", fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('KMeans_RadarChart_Wine.png')
plt.close()

print("Gráfico tipo telaraña generado correctamente (KMeans_RadarChart_Wine.png).")

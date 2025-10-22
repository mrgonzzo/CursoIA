# ============================================================================
# COMPARATIVA DE MÉTODOS: PCA, K-Means y Clustering Jerárquico
# Dataset: Wine
# Incluye varianza explicada, métricas de calidad y tiempo de ejecución
# ============================================================================

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# ----------------------------------------------------------------------------
# 1. CARGA Y ESCALADO DE DATOS
# ----------------------------------------------------------------------------
wine = load_wine()
X = wine.data
y_true = wine.target
feature_names = wine.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 2. PCA - Reducción de Dimensionalidad
# ----------------------------------------------------------------------------
print("\nEjecutando PCA...")
start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
time_pca = time.time() - start
var_exp = pca.explained_variance_ratio_.sum()

print(f"PCA completado en {time_pca:.4f} s (varianza explicada: {var_exp:.2%})")

# ----------------------------------------------------------------------------
# 3. K-MEANS CLUSTERING
# ----------------------------------------------------------------------------
print("\nEjecutando K-Means...")
start = time.time()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
time_kmeans = time.time() - start

sil_k = silhouette_score(X_scaled, y_kmeans)
db_k = davies_bouldin_score(X_scaled, y_kmeans)
ch_k = calinski_harabasz_score(X_scaled, y_kmeans)

print(f"K-Means completado en {time_kmeans:.4f} s")

# ----------------------------------------------------------------------------
# 4. CLUSTERING JERÁRQUICO
# ----------------------------------------------------------------------------
print("\nEjecutando Clustering Jerárquico (Ward)...")
start = time.time()
Z = linkage(X_scaled, method='ward')
y_hier = fcluster(Z, t=3, criterion='maxclust')
time_hier = time.time() - start

sil_h = silhouette_score(X_scaled, y_hier)
db_h = davies_bouldin_score(X_scaled, y_hier)
ch_h = calinski_harabasz_score(X_scaled, y_hier)

print(f"Jerárquico completado en {time_hier:.4f} s")

# ----------------------------------------------------------------------------
# 5. TABLA DE RESULTADOS Y RANKING
# ----------------------------------------------------------------------------
resultados = pd.DataFrame({
    'Método': ['PCA (referencia)', 'K-Means', 'Jerárquico'],
    'Silhouette': [np.nan, sil_k, sil_h],
    'Davies-Bouldin (↓)': [np.nan, db_k, db_h],
    'Calinski-Harabasz (↑)': [np.nan, ch_k, ch_h],
    'Varianza Explicada (PCA)': [var_exp, np.nan, np.nan],
    'Tiempo (s)': [time_pca, time_kmeans, time_hier]
})

# Ranking solo entre métodos de clustering
resultados['Ranking'] = (
    resultados['Silhouette'].rank(ascending=False) +
    resultados['Calinski-Harabasz (↑)'].rank(ascending=False) +
    resultados['Davies-Bouldin (↓)'].rank(ascending=True)
)

resultados = resultados.sort_values('Ranking', na_position='first').reset_index(drop=True)

print("\n" + "=" * 80)
print("RESULTADOS COMPARATIVOS")
print("=" * 80)
print(resultados.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

# ----------------------------------------------------------------------------
# 6. VISUALIZACIÓN PCA + CLUSTERS
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
colores = ['red', 'green', 'blue']

# K-Means
axs[0].set_title("K-Means sobre componentes PCA")
for i in range(3):
    mask = y_kmeans == i
    axs[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {i}', s=80, alpha=0.6, edgecolors='black', color=colores[i])
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")

# Jerárquico
axs[1].set_title("Jerárquico (Ward) sobre componentes PCA")
for i in range(3):
    mask = y_hier == i + 1
    axs[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {i+1}', s=80, alpha=0.6, edgecolors='black', color=colores[i])
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 7. INTERPRETACIÓN FINAL
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("INTERPRETACIÓN Y CONCLUSIÓN")
print("=" * 80)
print(f"\n📊 PCA explicó el {var_exp:.2%} de la varianza total con 2 componentes "
      f"en {time_pca:.4f} segundos.")
print("Sirve como referencia visual o paso previo al clustering.")

if resultados.iloc[-1]['Método'] == 'K-Means':
    print("\n🏆 El mejor método de agrupamiento fue **K-Means**: "
          "produce clusters compactos y es el más rápido.")
elif resultados.iloc[-1]['Método'] == 'Jerárquico':
    print("\n🏆 El mejor método de agrupamiento fue **Jerárquico**: "
          "revela una estructura jerárquica más rica aunque tarda más.")
else:
    print("\nℹ️ PCA no se usa para clustering, pero ayuda a visualizar los grupos obtenidos.")

print("\n📌 Recomendaciones:")
print("- Usa **PCA** para visualizar o reducir la dimensionalidad antes de agrupar.")
print("- Usa **K-Means** si sospechas que los grupos son compactos y similares.")
print("- Usa **Jerárquico** si quieres explorar niveles o subgrupos sin fijar K.")
print("- Considera **tiempo de ejecución**: PCA y K-Means son mucho más rápidos que Ward.")

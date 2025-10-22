# ============================================================================
# COMPARATIVA DE MÉTODOS: PCA, K-Means y Clustering Jerárquico
# Dataset: Wine
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

# ----------------------------------------------------------------------------
# 1. CARGAR Y PREPARAR DATOS
# ----------------------------------------------------------------------------
wine = load_wine()
X = wine.data
y_true = wine.target
feature_names = wine.feature_names

X_scaled = StandardScaler().fit_transform(X)

# ----------------------------------------------------------------------------
# 2. PCA (Reducción de dimensionalidad)
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_.sum()

# ----------------------------------------------------------------------------
# 3. K-MEANS
# ----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

sil_k = silhouette_score(X_scaled, y_kmeans)
db_k = davies_bouldin_score(X_scaled, y_kmeans)
ch_k = calinski_harabasz_score(X_scaled, y_kmeans)

# ----------------------------------------------------------------------------
# 4. CLUSTERING JERÁRQUICO
# ----------------------------------------------------------------------------
Z = linkage(X_scaled, method='ward')
y_hier = fcluster(Z, t=3, criterion='maxclust')

sil_h = silhouette_score(X_scaled, y_hier)
db_h = davies_bouldin_score(X_scaled, y_hier)
ch_h = calinski_harabasz_score(X_scaled, y_hier)

# ----------------------------------------------------------------------------
# 5. RESULTADOS Y RANKING
# ----------------------------------------------------------------------------
resultados = pd.DataFrame({
    'Método': ['K-Means', 'Jerárquico'],
    'Silhouette': [sil_k, sil_h],
    'Davies-Bouldin (↓)': [db_k, db_h],
    'Calinski-Harabasz (↑)': [ch_k, ch_h]
})

# Ranking de calidad: más alto Silhouette + Calinski, menor Davies
resultados['Ranking'] = (
    resultados['Silhouette'].rank(ascending=False) +
    resultados['Calinski-Harabasz (↑)'].rank(ascending=False) +
    resultados['Davies-Bouldin (↓)'].rank(ascending=True)
)

resultados = resultados.sort_values('Ranking').reset_index(drop=True)

print("\n" + "=" * 70)
print("RESULTADOS COMPARATIVOS")
print("=" * 70)
print(resultados.to_string(index=False))

# ----------------------------------------------------------------------------
# 6. VISUALIZACIÓN COMPARATIVA
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(resultados['Método'], resultados['Silhouette'], color=['skyblue', 'lightcoral'])
plt.title('Comparativa de Métodos de Clustering (Silhouette Score)')
plt.ylabel('Silhouette Score')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 7. INTERPRETACIÓN Y CONCLUSIÓN
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("INTERPRETACIÓN Y CONCLUSIÓN")
print("=" * 70)

print(f"\n📊 PCA explicó el {var_exp:.2%} de la varianza total con 2 componentes.")
print("Esto permite visualizar los clusters pero no agruparlos por sí solo.")

if resultados.iloc[0]['Método'] == 'K-Means':
    print("\n🏆 El mejor método de agrupamiento fue **K-Means**, "
          "lo que indica que los vinos forman clusters más compactos y bien separados.")
    print("Es ideal cuando se sospecha que los grupos son esféricos y de tamaño similar.")
else:
    print("\n🏆 El mejor método de agrupamiento fue **Jerárquico**, "
          "lo que indica que los vinos tienen una estructura jerárquica más rica.")
    print("Es útil para descubrir subgrupos o relaciones progresivas entre clases.")

print("\n📌 En general:")
print("- Usa **PCA** para explorar o reducir variables antes de agrupar.")
print("- Usa **K-Means** si conoces el número aproximado de clusters y tus datos son bien definidos.")
print("- Usa **Jerárquico** si quieres descubrir niveles de similitud o no sabes cuántos clusters hay.")

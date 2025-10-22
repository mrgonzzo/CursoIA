from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CLUSTERING JERÁRQUICO - Ejemplo 3: Seeds Dataset (Clasificación de semillas)
# ============================================================================

print("=" * 70)
print("CLUSTERING JERÁRQUICO - EJEMPLO 3: Seeds Dataset")
print("Agrupamiento de variedades de trigo mediante clustering jerárquico")
print("=" * 70)

# Cargar dataset de semillas de trigo
# Este dataset contiene mediciones geométricas de granos de trigo
# 3 variedades: Kama, Rosa, Canadian

# Como el dataset de seeds no está en sklearn, crearemos datos simulados basados en sus características
np.random.seed(42)

# Simular 210 semillas (70 de cada variedad)
n_per_variety = 70
varieties = ['Kama', 'Rosa', 'Canadian']

# Características: área, perímetro, compacidad, longitud, ancho, coef_asimetría, longitud_ranura
features = ['area', 'perimeter', 'compactness', 'kernel_length',
           'kernel_width', 'asymmetry_coef', 'groove_length']

# Generar datos con diferentes características para cada variedad
data_kama = np.random.randn(n_per_variety, 7) * np.array([2, 1.5, 0.02, 0.3, 0.2, 0.5, 0.3]) + \
            np.array([15, 14, 0.88, 5.5, 3.3, 2.5, 5.2])

data_rosa = np.random.randn(n_per_variety, 7) * np.array([1.8, 1.3, 0.015, 0.25, 0.18, 0.4, 0.28]) + \
            np.array([18, 16, 0.85, 6.0, 3.7, 4.0, 5.8])

data_canadian = np.random.randn(n_per_variety, 7) * np.array([1.5, 1.2, 0.018, 0.28, 0.2, 0.45, 0.25]) + \
                np.array([12, 13, 0.81, 5.0, 2.9, 5.5, 4.9])

X = np.vstack([data_kama, data_rosa, data_canadian])
y_true = np.array([0]*n_per_variety + [1]*n_per_variety + [2]*n_per_variety)

print(f"\nDatos: {X.shape[0]} semillas de trigo")
print(f"Características: {', '.join(features)}")
print(f"Variedades: {', '.join(varieties)}")
print(f"Distribución real: {n_per_variety} semillas por variedad\n")

# Mostrar estadísticas
df = pd.DataFrame(X, columns=features)
df['variety'] = [varieties[i] for i in y_true]

print("Estadísticas descriptivas:")
print(df.groupby('variety')[features[:4]].mean().round(2))

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular matriz de distancias
print("\nCalculando matriz de distancias...")
distances = pdist(X_scaled, metric='euclidean')

# Probar diferentes métodos de enlace
print("\nComparando métodos de enlace:")
linkage_methods = ['single', 'complete', 'average', 'ward']
linkage_results = {}
cophenetic_corrs = {}

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    linkage_results[method] = Z

    # Calcular correlación cofenética
    c, coph_dists = cophenet(Z, pdist(X_scaled))
    cophenetic_corrs[method] = c

    # Formar clusters (usar siempre 3 clusters para comparación justa)
    clusters = fcluster(Z, t=3, criterion='maxclust')

    # Verificar que se formaron múltiples clusters antes de calcular silhouette
    n_clusters_formed = len(np.unique(clusters))
    if n_clusters_formed > 1:
        silhouette = silhouette_score(X_scaled, clusters)
        print(f"  {method:10s}: Silhouette = {silhouette:.3f}, Cophenetic = {c:.3f}")
    else:
        print(f"  {method:10s}: Solo 1 cluster formado, Cophenetic = {c:.3f}")

# Usar Ward como método principal (generalmente mejor para clustering)
best_method = 'ward'
Z = linkage_results[best_method]

# Determinar número de clusters cortando el dendrograma
print(f"\nUsando método '{best_method}' para análisis detallado...")

# Probar diferentes números de clusters
print("\nEvaluando diferentes números de clusters:")
for n_clusters in [2, 3, 4, 5]:
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    silhouette = silhouette_score(X_scaled, clusters)
    print(f"  {n_clusters} clusters: Silhouette = {silhouette:.3f}")

# Formar 3 clusters (sabemos que hay 3 variedades)
best_n_clusters = 3
clusters = fcluster(Z, t=best_n_clusters, criterion='maxclust')

print(f"\nFormando {best_n_clusters} clusters...")
print(f"Distribución de clusters:")
for cluster_id in range(1, best_n_clusters + 1):
    count = np.sum(clusters == cluster_id)
    print(f"  Cluster {cluster_id}: {count} semillas")

# PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualización
fig = plt.figure(figsize=(18, 14))

# Plot 1: Dendrograma completo con todas las semillas
ax1 = plt.subplot(3, 3, (1, 3))
dendro = dendrogram(Z, no_labels=True, color_threshold=10)
ax1.set_title(f'Dendrograma Completo - Método: {best_method.upper()}', fontsize=12)
ax1.set_xlabel('Índice de Semilla')
ax1.set_ylabel('Distancia')
ax1.axhline(y=10, c='red', linestyle='--', linewidth=2,
           label=f'{best_n_clusters} clusters')
ax1.legend()

# Plot 2: Dendrograma de una muestra (primeras 50 semillas)
ax2 = plt.subplot(3, 3, 4)
Z_sample = linkage(X_scaled[:50], method=best_method)
dendro_sample = dendrogram(Z_sample, labels=[f'S{i}' for i in range(50)],
                          leaf_font_size=6)
ax2.set_title('Dendrograma (Muestra de 50 semillas)')
ax2.set_xlabel('Semilla')
ax2.set_ylabel('Distancia')
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Plot 3: Comparación de métodos de enlace
ax3 = plt.subplot(3, 3, 5)
methods_colors = {'single': 'blue', 'complete': 'green', 'average': 'orange', 'ward': 'red'}
silhouettes = []
for method in linkage_methods:
    Z_temp = linkage_results[method]
    clusters_temp = fcluster(Z_temp, t=3, criterion='maxclust')
    sil = silhouette_score(X_scaled, clusters_temp)
    silhouettes.append(sil)

ax3.bar(linkage_methods, silhouettes, color=[methods_colors[m] for m in linkage_methods],
       alpha=0.7, edgecolor='black')
ax3.set_ylabel('Silhouette Score')
ax3.set_title('Comparación de Métodos de Enlace')
ax3.set_ylim(0, max(silhouettes) * 1.2)
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(silhouettes):
    ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 4: Clusters en 2D (PCA)
ax4 = plt.subplot(3, 3, 6)
colores_clusters = ['red', 'green', 'blue']
for i in range(1, best_n_clusters + 1):
    mask = clusters == i
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=f'Cluster {i}', c=colores_clusters[i-1],
               s=80, alpha=0.6, edgecolors='black')

ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax4.set_title('Clusters en 2D (PCA)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Variedades reales en 2D (para comparación)
ax5 = plt.subplot(3, 3, 7)
colores_variedades = ['purple', 'orange', 'cyan']
for i, variedad in enumerate(varieties):
    mask = y_true == i
    ax5.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=variedad, c=colores_variedades[i],
               s=80, alpha=0.6, edgecolors='black')

ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax5.set_title('Variedades Reales (comparación)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Matriz de confusión
ax6 = plt.subplot(3, 3, 8)
confusion_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(1, 4):
        confusion_matrix[i, j-1] = np.sum((y_true == i) & (clusters == j))

im = ax6.imshow(confusion_matrix, cmap='Greens', aspect='auto')
ax6.set_xticks(range(3))
ax6.set_yticks(range(3))
ax6.set_xticklabels([f'Cluster {i}' for i in range(1, 4)])
ax6.set_yticklabels(varieties)
ax6.set_xlabel('Cluster Jerárquico')
ax6.set_ylabel('Variedad Real')
ax6.set_title('Matriz de Confusión')

for i in range(3):
    for j in range(3):
        text = ax6.text(j, i, int(confusion_matrix[i, j]),
                       ha="center", va="center",
                       color="white" if confusion_matrix[i, j] > 35 else "black",
                       fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax6, label='Número de semillas')

# Plot 7: Perfil de características por cluster
ax7 = plt.subplot(3, 3, 9)
cluster_profiles = []
for i in range(1, best_n_clusters + 1):
    mask = clusters == i
    profile = X_scaled[mask].mean(axis=0)
    cluster_profiles.append(profile)

cluster_profiles = np.array(cluster_profiles)
x_pos = np.arange(len(features))
width = 0.25

for i in range(best_n_clusters):
    ax7.bar(x_pos + i * width, cluster_profiles[i],
           width, label=f'Cluster {i+1}', alpha=0.7,
           color=colores_clusters[i], edgecolor='black')

ax7.set_xticks(x_pos + width)
ax7.set_xticklabels(['Area', 'Perim', 'Comp', 'KLen', 'KWid', 'Asym', 'Groove'],
                    rotation=45, ha='right')
ax7.set_ylabel('Valor Estandarizado')
ax7.set_title('Perfil de Características por Cluster')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')
ax7.axhline(y=0, color='black', linewidth=0.5)

plt.suptitle('Clustering Jerárquico: Análisis de Semillas de Trigo',
            fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

# Análisis de pureza
print("\n" + "=" * 70)
print("ANÁLISIS DE CONCORDANCIA CON VARIEDADES REALES")
print("=" * 70)

for cluster_id in range(1, best_n_clusters + 1):
    print(f"\nCluster {cluster_id}:")
    mask = clusters == cluster_id
    variedades_en_cluster = y_true[mask]

    for var_idx, var_nombre in enumerate(varieties):
        count = np.sum(variedades_en_cluster == var_idx)
        porcentaje = (count / len(variedades_en_cluster)) * 100 if len(variedades_en_cluster) > 0 else 0
        if count > 0:
            print(f"  {var_nombre:10s}: {count:3d} semillas ({porcentaje:5.1f}%)")

# Interpretación
print("\n" + "=" * 70)
print("INTERPRETACIÓN")
print("=" * 70)
print(f"""
El clustering jerarquico con metodo '{best_method}' ha identificado {best_n_clusters} grupos naturales
en las semillas de trigo. El dendrograma muestra la jerarquia completa de agrupamiento,
permitiendo explorar diferentes niveles de granularidad.

Ventajas del clustering jerarquico observadas:
> No requiere especificar K de antemano (aunque lo validamos con 3)
> El dendrograma proporciona informacion visual rica sobre relaciones
> Resultados deterministicos (mismo resultado en cada ejecucion)
> Captura la estructura jerarquica natural de las variedades de trigo
""")

# ============================================================================
# VISUALIZACIONES AVANZADAS Y COMPLEJAS
# ============================================================================

print("\n" + "=" * 70)
print("GENERANDO VISUALIZACIONES AVANZADAS...")
print("=" * 70)

# Calcular métricas adicionales para diferentes números de clusters
n_clusters_range = range(2, 11)
metrics_by_n = {
    'silhouette': [],
    'calinski': [],
    'davies_bouldin': []
}

for n in n_clusters_range:
    clusters_temp = fcluster(Z, t=n, criterion='maxclust')
    metrics_by_n['silhouette'].append(silhouette_score(X_scaled, clusters_temp))
    metrics_by_n['calinski'].append(calinski_harabasz_score(X_scaled, clusters_temp))
    metrics_by_n['davies_bouldin'].append(davies_bouldin_score(X_scaled, clusters_temp))

# Figura 2: Visualizaciones avanzadas
fig2 = plt.figure(figsize=(20, 16))

# Plot 1: Método del codo con múltiples métricas
ax1 = plt.subplot(3, 4, 1)
ax1_twin = ax1.twinx()
ax1.plot(n_clusters_range, metrics_by_n['silhouette'], 'o-', color='blue',
         linewidth=2, markersize=8, label='Silhouette')
ax1_twin.plot(n_clusters_range, metrics_by_n['davies_bouldin'], 's-', color='red',
              linewidth=2, markersize=8, label='Davies-Bouldin')
ax1.axvline(x=best_n_clusters, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('Silhouette Score', color='blue')
ax1_twin.set_ylabel('Davies-Bouldin Score', color='red')
ax1.set_title('Método del Codo - Múltiples Métricas')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: Calinski-Harabasz Score
ax2 = plt.subplot(3, 4, 2)
ax2.plot(n_clusters_range, metrics_by_n['calinski'], 'o-', color='purple',
         linewidth=2, markersize=8)
ax2.axvline(x=best_n_clusters, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label=f'K óptimo = {best_n_clusters}')
ax2.set_xlabel('Número de Clusters')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.set_title('Calinski-Harabasz Score')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Gráfico de Silueta detallado
ax3 = plt.subplot(3, 4, 3)
silhouette_vals = silhouette_samples(X_scaled, clusters)
y_lower = 10
colors_sil = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i in range(1, best_n_clusters + 1):
    cluster_silhouette_vals = silhouette_vals[clusters == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=colors_sil[i-1], edgecolor=colors_sil[i-1], alpha=0.7)
    ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

silhouette_avg = np.mean(silhouette_vals)
ax3.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
           label=f'Promedio: {silhouette_avg:.3f}')
ax3.set_xlabel('Coeficiente de Silueta')
ax3.set_ylabel('Cluster')
ax3.set_title('Análisis de Silueta por Cluster')
ax3.legend()
ax3.set_yticks([])

# Plot 4: Heatmap de matriz de distancias (muestra)
ax4 = plt.subplot(3, 4, 4)
# Tomar una muestra para que sea visible
sample_idx = np.random.choice(len(X_scaled), 50, replace=False)
sample_idx = np.sort(sample_idx)
dist_matrix_sample = squareform(pdist(X_scaled[sample_idx]))
im = ax4.imshow(dist_matrix_sample, cmap='YlOrRd', aspect='auto')
ax4.set_title('Matriz de Distancias (muestra de 50)')
ax4.set_xlabel('Muestra')
ax4.set_ylabel('Muestra')
plt.colorbar(im, ax=ax4, label='Distancia Euclidiana')

# Plot 5: Dendrograma con colores por cluster
ax5 = plt.subplot(3, 4, (5, 6))
# Calcular el threshold para colorear
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(['#FF6B6B', '#4ECDC4', '#45B7D1'])
threshold = (Z[-best_n_clusters+1, 2] + Z[-best_n_clusters, 2]) / 2
dendro_colored = dendrogram(Z, no_labels=True, color_threshold=threshold,
                            above_threshold_color='gray')
ax5.set_title(f'Dendrograma Coloreado ({best_n_clusters} Clusters)', fontsize=12)
ax5.set_xlabel('Índice de Semilla')
ax5.set_ylabel('Distancia Ward')
ax5.axhline(y=threshold, c='red', linestyle='--', linewidth=2,
           label=f'Threshold = {threshold:.2f}')
ax5.legend()

# Plot 6: Comparación de correlaciones cofenéticas
ax6 = plt.subplot(3, 4, 7)
methods_list = list(cophenetic_corrs.keys())
corr_values = list(cophenetic_corrs.values())
bars = ax6.bar(methods_list, corr_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
              alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Correlación Cofenética')
ax6.set_title('Correlación Cofenética por Método')
ax6.set_ylim(0, 1)
ax6.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Umbral bueno (0.75)')
ax6.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    ax6.text(i, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
ax6.legend()

# Plot 7: PCA 3D - Clusters
ax7 = fig2.add_subplot(3, 4, 8, projection='3d')
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
colors_3d = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i in range(1, best_n_clusters + 1):
    mask = clusters == i
    ax7.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
               label=f'Cluster {i}', c=colors_3d[i-1], s=50, alpha=0.6,
               edgecolors='black', linewidth=0.5)
ax7.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
ax7.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
ax7.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
ax7.set_title('Visualización 3D de Clusters (PCA)')
ax7.legend()

# Plot 8: Heatmap de características por cluster
ax8 = plt.subplot(3, 4, 9)
cluster_means = np.array([X[clusters == i].mean(axis=0) for i in range(1, best_n_clusters + 1)])
im = ax8.imshow(cluster_means.T, cmap='RdYlGn', aspect='auto')
ax8.set_xticks(range(best_n_clusters))
ax8.set_yticks(range(len(features)))
ax8.set_xticklabels([f'C{i}' for i in range(1, best_n_clusters + 1)])
ax8.set_yticklabels(['Area', 'Perímetro', 'Compacidad', 'Long. Kernel',
                     'Ancho Kernel', 'Asimetría', 'Long. Ranura'], fontsize=9)
ax8.set_xlabel('Cluster')
ax8.set_title('Perfil de Características (valores originales)')
plt.colorbar(im, ax=ax8)

# Plot 9: Distribución de tamaños de clusters
ax9 = plt.subplot(3, 4, 10)
cluster_sizes = [np.sum(clusters == i) for i in range(1, best_n_clusters + 1)]
explode = [0.05] * best_n_clusters
ax9.pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(1, best_n_clusters + 1)],
       autopct='%1.1f%%', startangle=90, colors=colors_3d, explode=explode,
       shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax9.set_title('Distribución de Tamaños de Clusters')

# Plot 10: Boxplot de características discriminantes
ax10 = plt.subplot(3, 4, 11)
df_analysis = pd.DataFrame(X_scaled, columns=features)
df_analysis['cluster'] = clusters
# Seleccionar las 3 características más discriminantes
feature_importance = []
for feat in features:
    variance_between = df_analysis.groupby('cluster')[feat].var().mean()
    feature_importance.append((feat, variance_between))
feature_importance.sort(key=lambda x: x[1], reverse=True)
top_features = [f[0] for f in feature_importance[:3]]

df_melted = df_analysis[top_features + ['cluster']].melt(id_vars='cluster',
                                                          var_name='Feature',
                                                          value_name='Value')
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='cluster', ax=ax10,
           palette=colors_3d)
ax10.set_title('Top 3 Características Discriminantes')
ax10.set_xlabel('Característica')
ax10.set_ylabel('Valor Estandarizado')
ax10.legend(title='Cluster', loc='upper right')
ax10.grid(True, alpha=0.3, axis='y')

# Plot 11: Análisis de estabilidad - Dendrogramas de submuestras
ax11 = plt.subplot(3, 4, 12)
np.random.seed(123)
for i in range(5):
    subsample_idx = np.random.choice(len(X_scaled), 150, replace=False)
    Z_sub = linkage(X_scaled[subsample_idx], method=best_method)
    dendro_sub = dendrogram(Z_sub, no_labels=True, ax=ax11,
                           color_threshold=0, above_threshold_color=f'C{i}',
                           no_plot=False if i == 0 else True)
ax11.set_title('Estabilidad del Clustering\n(5 submuestras aleatorias)')
ax11.set_xlabel('Muestras')
ax11.set_ylabel('Distancia')

plt.suptitle('Análisis Avanzado de Clustering Jerárquico - Seeds Dataset',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ============================================================================
# ANÁLISIS ADICIONAL: COMPARACIÓN DE DISTANCIAS
# ============================================================================

print("\n" + "=" * 70)
print("ANÁLISIS DE MÉTRICAS DE DISTANCIA")
print("=" * 70)

fig3 = plt.figure(figsize=(18, 10))

distance_metrics = ['euclidean', 'cityblock', 'cosine', 'correlation']
silhouettes_dist = []

for idx, metric in enumerate(distance_metrics, 1):
    ax = plt.subplot(2, 4, idx)

    # Calcular linkage con diferente métrica de distancia
    if metric in ['euclidean', 'manhattan', 'cosine', 'correlation']:
        # Ward solo funciona con euclidean, usar average para otras métricas
        if metric == 'euclidean':
            Z_metric = linkage(X_scaled, method='ward')
        else:
            Z_metric = linkage(X_scaled, method='average', metric=metric)
        clusters_metric = fcluster(Z_metric, t=best_n_clusters, criterion='maxclust')

        # Visualizar en 2D
        for i in range(1, best_n_clusters + 1):
            mask = clusters_metric == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=f'Cluster {i}', c=colors_3d[i-1],
                      s=60, alpha=0.6, edgecolors='black', linewidth=0.5)

        sil_score = silhouette_score(X_scaled, clusters_metric)
        silhouettes_dist.append(sil_score)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        metric_name = 'Manhattan' if metric == 'cityblock' else metric.capitalize()
        ax.set_title(f'{metric_name}\nSilhouette: {sil_score:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Dendrograma correspondiente
        ax_dendro = plt.subplot(2, 4, idx + 4)
        dendro_metric = dendrogram(Z_metric, no_labels=True, ax=ax_dendro)
        ax_dendro.set_title(f'Dendrograma - {metric_name}')
        ax_dendro.set_xlabel('Muestras')
        ax_dendro.set_ylabel('Distancia')

plt.suptitle('Comparación de Métricas de Distancia en Clustering Jerárquico',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ============================================================================
# RESUMEN FINAL DE MÉTRICAS
# ============================================================================

print("\n" + "=" * 70)
print("RESUMEN DE MÉTRICAS DE EVALUACIÓN")
print("=" * 70)

print(f"\nMétodo de enlace: {best_method.upper()}")
print(f"Número de clusters: {best_n_clusters}")
print(f"\nMétricas de calidad:")
print(f"  Silhouette Score:        {silhouette_score(X_scaled, clusters):.4f}")
print(f"  Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, clusters):.4f}")
print(f"  Davies-Bouldin Score:    {davies_bouldin_score(X_scaled, clusters):.4f}")
print(f"  Correlación Cofenética:  {cophenetic_corrs[best_method]:.4f}")

print("\nInterpretación de métricas:")
print("  • Silhouette ([-1, 1]): Más cercano a 1 es mejor")
print("  • Calinski-Harabasz: Mayor es mejor")
print("  • Davies-Bouldin: Menor es mejor")
print("  • Cofenética ([0, 1]): Más cercano a 1 indica mejor preservación de distancias")

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)
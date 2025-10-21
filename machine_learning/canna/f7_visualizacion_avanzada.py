import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

print("--- f7_visualizacion_avanzada.py: Visualización PCA y K-Means ---")

# 1. Cargar resultados
df_final = pd.read_csv('resumen_pca_kmeans_final.csv')
pca_df = pd.read_csv('pca_scores.csv', index_col='Strain')

# --- 7a. Método del Codo (Para justificar K=3) ---
sse = []
K_limit = min(6, len(pca_df.index))
K_range = range(1, K_limit)

if K_range:
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pca_df)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, sse, marker='o', linestyle='--')
    plt.title('Método del Codo para K-Means')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (SSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kmeans_elbow_plot.png')
    plt.close()
    print("✅ Gráfico del Método del Codo guardado como 'kmeans_elbow_plot.png'.")

# --- 7b. Visualización PCA y K-Means (El mapa de la variación) ---

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Cluster',
    style='Type',
    data=df_final,
    palette='viridis',
    s=150,
    alpha=0.8
)

# Añadir etiquetas de las cepas
for index, row in df_final.iterrows():
    plt.text(row['PC1'] * 1.02, row['PC2'] * 1.02, row['Strain'],
             horizontalalignment='left', size='small', color='black', weight='semibold')

plt.title('Variación y Agrupamiento de Cepas de Cannabis (PCA + K-Means)')
plt.xlabel('Componente Principal 1 (Eje de Máxima Variación Químico/Efectos)')
plt.ylabel('Componente Principal 2')
plt.legend(title='Grupo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_kmeans_clustering_plot.png')
plt.close()
print("✅ Gráfico de dispersión PCA/K-Means guardado como 'pca_kmeans_clustering_plot.png'.")
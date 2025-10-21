import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- 1. Dataset: Datos Simplificados de Cepas de Cannabis ---
csv_data = """Strain,Type,THC_Level,CBD_Level,Myrcene,Limonene,Caryophyllene,Relaxed,Happy,Euphoric,Uplifted,Sleepy,Anxiety
Blue Dream,Hybrid,17.0,0.1,0.5,0.3,0.1,65,75,80,60,30,10
Amnesia Haze,Sativa,20.0,0.0,0.1,0.6,0.2,30,85,90,95,10,5
OG Kush,Hybrid,22.0,0.1,0.8,0.2,0.6,90,70,60,40,65,15
Granddaddy Purple,Indica,23.0,0.2,0.7,0.1,0.4,95,60,50,20,90,5
Trainwreck,Sativa,18.0,0.0,0.2,0.4,0.3,40,78,85,88,15,8
White Widow,Hybrid,19.0,0.0,0.4,0.5,0.3,55,80,75,70,25,12
Northern Lights,Indica,21.0,0.1,0.6,0.1,0.5,92,65,55,35,80,7
Sour Diesel,Sativa,24.0,0.0,0.3,0.7,0.2,35,90,92,98,5,2
Gorilla Glue #4,Hybrid,28.0,0.0,0.9,0.2,0.7,85,72,68,50,60,18
Girl Scout Cookies,Hybrid,25.0,0.1,0.7,0.3,0.5,75,82,70,65,45,14
"""

data = pd.read_csv(StringIO(csv_data))

# --- 2. Preprocesamiento de Datos ---
features_cols = ['THC_Level', 'CBD_Level', 'Myrcene', 'Limonene', 'Caryophyllene',
                 'Relaxed', 'Happy', 'Euphoric', 'Uplifted', 'Sleepy', 'Anxiety']
X = data[features_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Análisis de Componentes Principales (PCA) ---
# Reducción a 2 componentes para la visualización.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['CP1', 'CP2'])
data_pca = pd.concat([data.reset_index(drop=True), pca_df], axis=1)

# --- 4. Método del Codo y K-Means ---
sse = []
K_range = range(1, 6)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_df)
    sse.append(kmeans.inertia_)

# Elegimos K=3 (Indica, Sativa, Hybrid)
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data_pca['Cluster'] = kmeans_model.fit_predict(pca_df)

# --- 5. Visualizaciones ---

# GRÁFICO 1: Método del Codo para K-Means
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o')
plt.title('Gráfico 1: Método del Codo - Encontrando K óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('SSE (Inercia)')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'K={optimal_k} elegido')
plt.legend()
plt.grid(True)
plt.show()  #

# GRÁFICO 2: Clusters K-Means en el Espacio PCA
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='CP1',
    y='CP2',
    hue='Cluster',
    style='Type',
    data=data_pca,
    palette='Set1',
    s=150,
    alpha=0.8
)
for line in range(0, data_pca.shape[0]):
    plt.text(data_pca.CP1[line] * 1.02, data_pca.CP2[line] * 1.02, data_pca.Strain[line],
             horizontalalignment='left', size='small', color='black', weight='semibold')

plt.title('Gráfico 2: Variación entre Cepas en el Espacio de PCA')
plt.xlabel(f'Componente Principal 1 (CP1 - {pca.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'Componente Principal 2 (CP2 - {pca.explained_variance_ratio_[1] * 100:.1f}%)')
plt.legend(title='Grupo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()  #
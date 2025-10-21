import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# URLs del dataset NSL-KDD (usaremos kddcup.data_10_percent.gz)
url_train = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack'
]

# --- 1. Carga y Preprocesamiento de Datos ---
print("Cargando y preprocesando datos...")
try:
    # Carga del 10% del dataset KDD CUP 99 (base del NSL-KDD)
    data = pd.read_csv(url_train, header=None, names=col_names, compression='gzip')
except Exception as e:
    print(f"Error al cargar el dataset: {e}. Asegúrate de tener conexión o descarga el archivo localmente.")
    exit()

# 1.1. Conversión de Variables Categóricas (One-Hot Encoding)
# Convierte 'protocol_type', 'service' y 'flag' a columnas binarias.
data_encoded = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'], drop_first=True)

# 1.2. Separación de Características
# Excluimos 'attack' (etiqueta de validación) y 'num_outbound_cmds' (siempre 0 en este set)
X = data_encoded.drop(columns=['attack', 'num_outbound_cmds'])

# 1.3. Estandarización
# PCA y K-Means se basan en la distancia, por lo que la escala es crucial.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Dimensiones después de One-Hot Encoding: {X.shape}")

# --- 2. Análisis de Componentes Principales (PCA) ---
print("\nAplicando PCA para reducción de dimensionalidad...")
# Seleccionamos componentes para retener el 95% de la varianza
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Componentes Principales (PCA) generadas: {X_pca.shape[1]}")
print(f"Varianza Total Explicada: {sum(pca.explained_variance_ratio_):.4f}")

pca_df = pd.DataFrame(X_pca)

# --- 3. Determinación del Número Óptimo de Clusters (Método del Codo) ---
print("\nCalculando SSE para el Método del Codo...")
sse = []
K_range = range(1, 15)

for k in K_range:
    # n_init=10 asegura que se elige el mejor resultado entre 10 corridas
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_df)
    sse.append(kmeans.inertia_)

# --- 4. Aplicación de K-Means y Análisis ---
# Asumiendo K=4 (un punto de inflexión común para KDD/NSL-KDD)
optimal_k = 4
print(f"\nAplicando K-Means con K = {optimal_k}...")

kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(pca_df)

data_encoded['Cluster'] = cluster_labels
data_encoded['Attack_Type'] = data['attack'].apply(lambda x: 'Normal' if x == 'normal.' else 'Attack')

# --- 5. Resultados del Clustering ---
print("\n--- Conteo de Tráfico Normal/Anómalo por Cluster (Validación) ---")
cluster_summary = data_encoded.groupby('Cluster')['Attack_Type'].value_counts().unstack(fill_value=0)
print(cluster_summary)

# Visualización de las tasas de error promedio por cluster
print("\nTasas de Error Promedio por Cluster:")
error_cols = ['serror_rate', 'rerror_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate']
print(data_encoded.groupby('Cluster')[error_cols].mean())

# --- 6. Visualizaciones ---

# Gráfico 1: Método del Codo
plt.figure(figsize=(9, 5))
plt.plot(K_range, sse, marker='o')
plt.title('Método del Codo para K-Means (NSL-KDD)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'K óptimo elegido ({optimal_k})')
plt.legend()
plt.show() #

# Gráfico 2: Clusters en el Espacio PCA
if X_pca.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=pca_df.iloc[:, 0],
        y=pca_df.iloc[:, 1],
        hue=data_encoded['Cluster'],
        palette='viridis',
        legend='full',
        alpha=0.6,
        s=10 # Tamaño de los puntos
    )
    plt.title('Clusters K-Means en las 2 Primeras Componentes de PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show() #
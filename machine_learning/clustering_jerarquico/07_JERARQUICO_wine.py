from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# EJERCICIO 3: Clustering Jerárquico - Wine Dataset (Clasificación de vinos)
# ============================================================================

print("\n" + "=" * 60)
print("EJERCICIO 3: Clustering Jerárquico - Wine Dataset")
print("=" * 60)

# Cargar dataset: 178 vinos, 13 características químicas
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

print(f"Datos: {X.shape[0]} vinos, {X.shape[1]} características químicas")
print(f"Características: {', '.join(feature_names[:5])}...\n")

# Estandarizar los datos
X_escalado = StandardScaler().fit_transform(X)

# ============================================================================
# CLUSTERING JERÁRQUICO Y DENDROGRAMA
# ============================================================================

# Usamos solo 30 muestras para visualizar mejor el dendrograma
X_muestra = X_escalado[:30]
linkage_matrix = linkage(X_muestra, method='ward')

# Dendrograma
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=[f'V{i}' for i in range(30)])
plt.title('Dendrograma: Agrupamiento Jerárquico de Vinos')
plt.xlabel('Vino')
plt.ylabel('Distancia')
plt.axhline(y=50, c='red', linestyle='--', alpha=0.5, label='Corte propuesto')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('Dendrograma_Agrupamiento_Vinos.png')
plt.close()

# ============================================================================
# ASIGNAR CLUSTERS SEGÚN UMBRAL DE CORTE
# ============================================================================

# Generar etiquetas de cluster para todas las muestras
Z = linkage(X_escalado, method='ward')
y_clusters = fcluster(Z, t=50, criterion='distance')

print(f"Número de clusters formados: {len(np.unique(y_clusters))}")
print(pd.Series(y_clusters).value_counts(), "\n")

# ============================================================================
# GRÁFICO TIPO TELARAÑA (Radar Chart)
# ============================================================================

print("Generando gráfico tipo telaraña (perfil químico por cluster)...")

# Crear DataFrame con las etiquetas de cluster
df = pd.DataFrame(X, columns=feature_names)
df['Cluster'] = y_clusters

# Calcular la media por cluster
mean_por_cluster = df.groupby('Cluster').mean()

# Seleccionar algunas variables para que el gráfico sea legible
variables = feature_names[:6]  # puedes cambiar cuántas usar
num_vars = len(variables)

# Calcular ángulos del círculo
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el círculo

# Crear figura polar
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

for i, cluster in enumerate(mean_por_cluster.index):
    valores = mean_por_cluster.loc[cluster, variables].values.tolist()
    valores += valores[:1]
    ax.plot(angles, valores, color=colores[i % len(colores)],
            linewidth=2, label=f'Cluster {cluster}')
    ax.fill(angles, valores, color=colores[i % len(colores)], alpha=0.25)

# Configuración estética
ax.set_xticks(angles[:-1])
ax.set_xticklabels(variables, fontsize=10)
ax.set_title("Perfil químico medio por Cluster (Clustering Jerárquico)", fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('Jerarquico_RadarChart_Wine.png')
plt.close()

print("Gráfico tipo telaraña generado correctamente (Jerarquico_RadarChart_Wine.png).")

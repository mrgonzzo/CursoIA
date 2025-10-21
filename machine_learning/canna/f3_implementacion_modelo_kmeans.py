import pandas as pd
from sklearn.cluster import KMeans
import joblib

print("--- f3_implementacion_modelo_kmean.py: Entrenamiento K-Means ---")

# Cargar las puntuaciones PCA (resultado de f2)
pca_df = pd.read_csv('pca_scores.csv', index_col='Strain')

# 1. Implementación y Entrenamiento del Modelo K-Means
optimal_k = 3 # Asumimos 3 clusters

kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(pca_df)

# 2. Guardar el Modelo K-Means y las Etiquetas
joblib.dump(kmeans_model, 'kmeans_model.pkl')
print(f"✅ Modelo K-Means entrenado con K={optimal_k} y guardado en 'kmeans_model.pkl'.")

labels_df = pd.DataFrame(cluster_labels,
                         columns=['Cluster'],
                         index=pca_df.index)
labels_df.to_csv('kmeans_labels.csv')
print("✅ Etiquetas de clúster guardadas en 'kmeans_labels.csv'.")
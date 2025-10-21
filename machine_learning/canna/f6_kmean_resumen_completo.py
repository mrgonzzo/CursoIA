import pandas as pd

print("--- f6_kmeans_resumen_completo.py: Resumen de Clústeres K-Means ---")

# Cargar los datos agrupados
df_clustered = pd.read_csv('resumen_pca_kmeans_final.csv', index_col='Strain')

# 1. Definir características numéricas para el perfil
feature_cols = df_clustered.select_dtypes(include=['float64', 'int64']).columns.tolist()
features_to_analyze = [col for col in feature_cols if col not in ['PC1', 'PC2', 'Cluster']]

# 2. Calcular el Perfil Promedio de cada Clúster
cluster_profile = df_clustered.groupby('Cluster')[features_to_analyze].mean()

print("\n--- Variación: Perfil Químico y de Efectos Promedio por Clúster K-Means ---")
print(cluster_profile.round(2))

# 3. Validación: Conteo de Tipos Originales por Clúster
cluster_counts = df_clustered.groupby('Cluster')['Type'].value_counts().unstack(fill_value=0)

print("\n--- Conteo de Tipos Biológicos Originales por Clúster (Validación) ---")
print(cluster_counts)

# Guardar el perfil de clústeres
cluster_profile.to_csv('kmeans_cluster_profile.csv')
print("✅ Perfiles guardados en 'kmeans_cluster_profile.csv'.")
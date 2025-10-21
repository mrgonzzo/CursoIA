import pandas as pd

print("--- f4_pca_resumen_completo.py: Resumen Consolidado ---")

# 1. Cargar todos los resultados
df_original = pd.read_csv('cannabis_data_original.csv', index_col='Strain')
pca_scores = pd.read_csv('pca_scores.csv', index_col='Strain')
kmeans_labels = pd.read_csv('kmeans_labels.csv', index_col='Strain')

# 2. Combinar DataFrames en orden: Originales -> PCA Scores -> K-Means Labels
df_resumen = df_original.join(pca_scores)
df_final = df_resumen.join(kmeans_labels)

# 3. Guardar el resumen final
df_final.to_csv('resumen_pca_kmeans_final.csv')
print("✅ Resumen final consolidado guardado en 'resumen_pca_kmeans_final.csv'.")
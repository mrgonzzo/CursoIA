import pandas as pd
from sklearn.decomposition import PCA
import joblib

print("--- f2_implementacion_modelo_pca.py: Entrenamiento PCA ---")

# Cargar los datos escalados (resultado de f1)
df_scaled = pd.read_csv('cannabis_data_scaled.csv', index_col='Strain')

# 1. Implementación y Entrenamiento del Modelo PCA
n_components = 2
pca_model = PCA(n_components=n_components)
pca_scores_array = pca_model.fit_transform(df_scaled)

# 2. Guardar el Modelo PCA y los Scores
joblib.dump(pca_model, 'pca_model.pkl')
print("✅ Modelo PCA entrenado y guardado en 'pca_model.pkl'.")

# Crear DataFrame de Scores
column_names = [f'PC{i+1}' for i in range(n_components)]
pca_scores_df = pd.DataFrame(pca_scores_array,
                             columns=column_names,
                             index=df_scaled.index)

pca_scores_df.to_csv('pca_scores.csv')
print(f"✅ Scores PCA ({n_components} componentes) guardados en 'pca_scores.csv'.")
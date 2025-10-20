import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# =============================
# 1. Cargar datos
# =============================
df = pd.read_csv('USArrests.csv')  # Cambia por tu CSV real
X = df.drop('USArrests', axis=1)
y = df['USArrests']

# Etiquetas de estados
if 'State' in df.columns:
    estados = df['State'].values
else:
    estados = [f"Obs{i+1}" for i in range(X.shape[0])]

# =============================
# 2. Crear pipeline PCA y transformar datos
# =============================
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=min(X.shape[0], X.shape[1]))
)

X_pca = pca_pipeline.fit_transform(X)
pca_model = pca_pipeline.named_steps['pca']
scaler = pca_pipeline.named_steps['standardscaler']

# =============================
# 3. Interpretación de componentes (PC1 y PC2)
# =============================
loadings = pd.DataFrame(
    pca_model.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
)

print("=== MATRIZ DE CARGAS (loadings) ===")
print(loadings.head(2))

print("\nInterpretación de PC1 y PC2:")
print("- PC1: representa principalmente criminalidad general (Murder, Assault).")
print("- PC2: representa urbanización y otros delitos (UrbanPop, Rape).")
print("Esto permite relacionar los estados con niveles de criminalidad y urbanización.\n")

# =============================
# 4. Estados extremos en PC1
# =============================
pc1 = X_pca[:,0]
df_pc1 = pd.DataFrame({'Estado': estados, 'PC1': pc1})

top5 = df_pc1.sort_values(by='PC1', ascending=False).head(5)
bottom5 = df_pc1.sort_values(by='PC1', ascending=True).head(5)

print("=== 5 estados con mayor PC1 (mayor criminalidad) ===")
print(top5)

print("\n=== 5 estados con menor PC1 (menor criminalidad) ===")
print(bottom5)

print("\nAplicación práctica:")
print("- Estados con PC1 alto → priorizar recursos de seguridad y prevención.")
print("- Estados con PC1 bajo → podrían enfocarse en otras políticas sociales.\n")

# =============================
# 5. Recomendación técnica de componentes para ML
# =============================
varianza_acumulada = np.cumsum(pca_model.explained_variance_ratio_)
k_recomendados = np.argmax(varianza_acumulada >= 0.8) + 1

print(f"Se recomienda usar {k_recomendados} componentes principales "
      f"para modelos de ML (aproximadamente 80% de varianza explicada).")
print("Justificación matemática: los primeros PCs capturan la mayor parte de la información, "
      "reduciendo dimensionalidad y ruido, y acelerando el entrenamiento del modelo.")

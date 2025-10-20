import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# ---------------------------
# 1. Cargar datos
# ---------------------------
df = pd.read_csv('USArrests.csv')  # Cambia por tu CSV real

# ---------------------------
# 2. Separar variables predictoras
# ---------------------------
X = df.drop('USArrests', axis=1)
y = df['USArrests']

# ---------------------------
# 3. Crear y entrenar pipeline PCA
# ---------------------------
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=min(X.shape[0], X.shape[1]))  # Máximo componentes
)
X_pca = pca_pipeline.fit_transform(X)

# ---------------------------
# 4. Extraer modelo PCA y matriz de cargas
# ---------------------------
pca_model = pca_pipeline.named_steps['pca']
loadings = pd.DataFrame(
    pca_model.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])]
)

print("=== MATRIZ DE CARGAS (loadings) ===")
print(loadings)

print("\n=== VARIANZA EXPLICADA ===")
for i, var in enumerate(pca_model.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.3f}")
print("Varianza total explicada:", pca_model.explained_variance_ratio_.sum())

# ---------------------------
# 5. Resumen matemático de cada componente
# ---------------------------
print("\n=== RESUMEN MATEMÁTICO DE COMPONENTES ===\n")
for pc in loadings.index:
    formula = " + ".join(
        [f"{loadings.loc[pc, col]:.3f}*{col}" if loadings.loc[pc, col]>=0
         else f"({loadings.loc[pc, col]:.3f})*{col}"
         for col in loadings.columns]
    )
    top_vars = loadings.loc[pc].abs().sort_values(ascending=False)
    print(f"{pc} = {formula}")
    print("Variables que más contribuyen:", list(top_vars.index[:3]))
    print()

# ---------------------------
# 6. Heatmap de cargas
# ---------------------------
plt.figure(figsize=(10,6))
plt.imshow(loadings, cmap='viridis', aspect='auto')
plt.xticks(ticks=np.arange(len(loadings.columns)), labels=loadings.columns, rotation=45)
plt.yticks(ticks=np.arange(len(loadings.index)), labels=loadings.index)
plt.colorbar(label='Carga (loading)')
plt.title('Heatmap de cargas (loadings) PCA', fontsize=14)
plt.xlabel('Variables originales')
plt.ylabel('Componentes principales')
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Guardar pipeline entrenado
# ---------------------------
joblib.dump(pca_pipeline, 'pca_pipeline_entrenado.pkl')
print("\nPipeline PCA guardado como 'pca_pipeline_entrenado.pkl'.")

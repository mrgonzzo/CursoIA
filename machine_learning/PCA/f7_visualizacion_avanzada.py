import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Si tienes columna 'State' para etiquetas
if 'State' in df.columns:
    estados = df['State'].values
else:
    estados = [f"Obs{i+1}" for i in range(X.shape[0])]

# =============================
# 2. Crear pipeline PCA y transformar datos
# =============================
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

X_pca = pca_pipeline.fit_transform(X)
pca_model = pca_pipeline.named_steps['pca']
scaler = pca_pipeline.named_steps['standardscaler']

# =============================
# 3. Biplot
# =============================
pc1 = X_pca[:,0]
pc2 = X_pca[:,1]

plt.figure(figsize=(12,8))
plt.scatter(pc1, pc2, color='skyblue', s=100, alpha=0.7)

# Añadir etiquetas de los estados
for i, estado in enumerate(estados):
    plt.text(pc1[i]+0.02, pc2[i]+0.02, estado, fontsize=8)

# Líneas de referencia en (0,0)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot: Estados proyectados en PC1-PC2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# =============================
# 4. Análisis visual de clusters
# =============================
plt.figure(figsize=(12,8))
plt.scatter(pc1, pc2, c='lightgreen', s=100, alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

# Añadir etiquetas de los estados (opcional)
for i, estado in enumerate(estados):
    plt.text(pc1[i]+0.02, pc2[i]+0.02, estado, fontsize=8)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters visuales en espacio PC1-PC2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# =============================
# 5. Interpretación
# =============================
print("Interpretación:")
print("- Observa los puntos alejados del origen (0,0): posibles outliers.")
print("- Puntos cercanos entre sí forman clusters naturales: estados con características similares en las variables originales.")
print("- El biplot resume la información más importante capturada por las dos primeras componentes principales.")

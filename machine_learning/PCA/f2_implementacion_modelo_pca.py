import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
df = pd.read_csv('USArrests.csv')
# --- Seleccionar solo las variables predictoras ---
X = df.drop('USArrests', axis=1)

# --- Crear el pipeline ---
pca_pipeline = make_pipeline(
    StandardScaler(),  # Paso 1: estandarizar datos
    PCA()              # Paso 2: aplicar PCA
)

# --- Ajustar (fit) el pipeline a los datos ---
pca_pipeline.fit(X)

print("Pipeline creado y ajustado correctamente.")

# Extraer el objeto PCA del pipeline
pca_model = pca_pipeline.named_steps['pca']

# Mostrar proporción de varianza explicada por cada componente
print("Varianza explicada por cada componente:")
print(pca_model.explained_variance_ratio_)

# Mostrar la suma total de varianza explicada
print("\nVarianza total explicada:", pca_model.explained_variance_ratio_.sum())

'''Visualización de los Componentes Principales'''
# --- Variables predictoras y objetivo ---
X = df.drop('USArrests', axis=1)
y = df['USArrests']

# --- Crear y ajustar el pipeline ---
pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
X_pca = pca_pipeline.fit_transform(X)

# --- Convertir a DataFrame para graficar ---
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['USArrests'] = y.values

# --- Visualizar ---
plt.figure(figsize=(8,6))
plt.scatter(
    pca_df['PC1'], pca_df['PC2'],
    c=pca_df['USArrests'], cmap='coolwarm', edgecolor='k', s=80
)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Proyección PCA (2 Componentes Principales)')
plt.colorbar(label='USArrests (0 = bajo, 1 = alto)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
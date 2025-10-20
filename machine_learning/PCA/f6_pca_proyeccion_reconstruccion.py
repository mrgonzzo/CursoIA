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

# =============================
# 2. Crear y entrenar pipeline PCA
# =============================
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=min(X.shape[0], X.shape[1]))
)

X_pca = pca_pipeline.fit_transform(X)
pca_model = pca_pipeline.named_steps['pca']
scaler = pca_pipeline.named_steps['standardscaler']

# =============================
# 3. Proyección de observaciones (automática)
# =============================
print("=== Proyección automática de las primeras 5 observaciones ===")
print(X_pca[:5])

# =============================
# 4. Verificación manual
# =============================
componentes = pca_model.components_  # shape = (n_componentes, n_variables)
X_escalado = scaler.transform(X)
X_pca_manual = np.dot(X_escalado, componentes.T)

# Comparar resultados automáticos vs manual
diferencia_max = np.max(np.abs(X_pca - X_pca_manual))
print(f"\nDiferencia máxima entre proyección automática y manual: {diferencia_max:.10f}")

# =============================
# 5. Reconstrucción de datos
# =============================
# Reconstrucción en escala estandarizada
X_reconstruido_escalado = np.dot(X_pca, componentes)

# Reconstrucción en escala original
X_reconstruido = scaler.inverse_transform(X_reconstruido_escalado)

# DataFrame para comparar
recon_df = pd.DataFrame(X_reconstruido, columns=X.columns)
print("\n=== Primeras filas de los datos reconstruidos ===")
print(recon_df.head())

# Diferencia promedio entre datos originales y reconstruidos
diferencia_promedio = np.mean(np.abs(X - recon_df))
print(f"\nDiferencia promedio entre datos originales y reconstruidos: {diferencia_promedio:.6f}")

# =============================
# 6. Guardar pipeline entrenado
# =============================
joblib.dump(pca_pipeline, 'pca_pipeline_entrenado.pkl')
print("\nPipeline PCA guardado como 'pca_pipeline_entrenado.pkl'.")

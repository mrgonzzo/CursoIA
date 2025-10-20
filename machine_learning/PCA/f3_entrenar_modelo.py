# ===========================
# guardar_pca.py
# ===========================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

from f2_implementacion_modelo_pca import X

# ---------------------------
# Asume que X ya está definido
# X = df.drop('USArrests', axis=1)
# ---------------------------

# Crear pipeline con estandarización y PCA
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)  # Ajusta el número de componentes según necesidad
)

# Entrenar el pipeline
X_pca = pca_pipeline.fit_transform(X)
print("Pipeline PCA entrenado correctamente.")

# Extraer modelo PCA si se quiere inspeccionar
pca_model = pca_pipeline.named_steps['pca']
print("Varianza explicada por componente:", pca_model.explained_variance_ratio_)

# Guardar pipeline entrenado para reutilización
joblib.dump(pca_pipeline, 'pca_pipeline_entrenado.pkl')
print("Pipeline PCA guardado como 'pca_pipeline_entrenado.pkl'.")

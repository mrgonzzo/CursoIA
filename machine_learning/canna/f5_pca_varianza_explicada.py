import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

print("--- f5_pca_varianza_explicada.py: Análisis de Varianza PCA ---")

# 1. Cargar modelos y datos
pca_model = joblib.load('pca_model.pkl')
df_scaled = pd.read_csv('cannabis_data_scaled.csv', index_col='Strain')
features = df_scaled.columns

# 2. Varianza Explicada por Componente
explained_variance_ratio = pca_model.explained_variance_ratio_

# 3. Análisis de las Cargas (Loadings)
pca_loadings = pd.DataFrame(pca_model.components_.T,
                            columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
                            index=features)
print("\n--- Cargas de Componentes Principales (Loadings) ---")
print(pca_loadings.round(3))

# --- Guardar Gráfico de Varianza Explicada ---
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Varianza Individual')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Varianza Acumulada')
plt.ylabel('Proporción de Varianza Explicada')
plt.xlabel('Componente Principal')
plt.title('Varianza Explicada por Componentes Principales')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_variance_plot.png')
plt.close()
print("✅ Gráfico de varianza explicada guardado como 'pca_variance_plot.png'.")
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# EJERCICIO 7: PCA - Wine Quality Dataset (Calidad de vinos)
# ============================================================================

print("=" * 60)
print("EJERCICIO 7: PCA - Wine Quality Dataset")
print("=" * 60)

# Cargar dataset Wine Quality (vinos blancos y tintos)
# Nota: Este dataset tiene características químicas como acidez, azúcar, pH, etc.
try:
    # Intentar cargar desde sklearn
    from sklearn.datasets import load_wine
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    feature_names = wine_data.feature_names
    print("Dataset cargado: Wine (178 muestras, 13 características)")
except:
    # Alternativa: crear datos sintéticos basados en características de vino
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 13)
    y = np.random.randint(0, 3, n_samples)
    feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity', 'magnesium',
                     'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                     'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline']

print(f"Datos: {X.shape[0]} muestras de vino, {X.shape[1]} características químicas")
print(f"Características: {', '.join(feature_names[:5])}...\n")

# Estandarizar los datos
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Aplicar PCA para reducir de 13 a 3 dimensiones
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_escalado)

# Mostrar varianza explicada
print("Varianza explicada por cada componente:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print(f"Varianza total acumulada: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)\n")

# Mostrar las características más importantes en PC1
componentes_pc1 = pd.DataFrame({
    'Característica': feature_names,
    'Peso PC1': pca.components_[0]
}).sort_values('Peso PC1', ascending=False)
print("Top 5 características que más influyen en PC1:")
print(componentes_pc1.head())

# Visualización en 3D
fig = plt.figure(figsize=(14, 6))

# Plot 1: 2D (PC1 vs PC2)
ax1 = fig.add_subplot(121)
colores = ['red', 'green', 'blue']
nombres_tipos = ['Tipo A', 'Tipo B', 'Tipo C']
for i in range(3):
    if i in y:
        mask = y == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=nombres_tipos[i], c=colores[i], s=80, alpha=0.6)

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('PCA: Wine Quality - 13D → 2D')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scree Plot (Varianza explicada)
ax2 = fig.add_subplot(122)
pca_full = PCA()
pca_full.fit(X_escalado)
varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

ax2.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada,
         marker='o', linestyle='--', color='blue', linewidth=2)
ax2.axhline(y=0.95, color='red', linestyle='--', label='95% varianza')
ax2.set_xlabel('Número de Componentes')
ax2.set_ylabel('Varianza Acumulada')
ax2.set_title('Scree Plot: Varianza Explicada Acumulada')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

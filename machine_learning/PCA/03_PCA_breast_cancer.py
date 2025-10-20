from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================================
# PCA - Ejemplo 3: Breast Cancer Dataset (Diagnóstico médico)
# ============================================================================

print("=" * 70)
print("PCA - EJEMPLO 3: Breast Cancer Dataset")
print("Reducción de 30 características médicas a 2D para visualización")
print("=" * 70)

# Cargar dataset: 569 tumores con 30 características cada uno
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

print(f"\nDatos: {X.shape[0]} casos de tumores")
print(f"Características: {X.shape[1]} mediciones médicas")
print(f"Clases: {target_names[0]} (0) y {target_names[1]} (1)")
print(f"Distribución: {(y==0).sum()} malignos, {(y==1).sum()} benignos\n")

print("Primeras 10 características:")
for i, name in enumerate(feature_names[:10]):
    print(f"  {i+1}. {name}")
print("  ...")

# Estandarizar (crucial para PCA)
print("\nEstandarizando características...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Análisis de varianza con diferentes números de componentes
print("\nAnalizando varianza explicada por número de componentes:")
pca_full = PCA()
pca_full.fit(X_scaled)

varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
for n_comp in [2, 5, 10, 15, 20]:
    var_exp = varianza_acumulada[n_comp-1]
    print(f"  {n_comp} componentes: {var_exp:.2%} de varianza explicada")

# Aplicar PCA con 2 componentes para visualización
print("\nAplicando PCA: 30D → 2D")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nVarianza explicada por cada componente:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.2%}")

# Analizar qué características contribuyen más a cada PC
print("\nCaracterísticas con mayor peso en PC1:")
pc1_weights = pd.DataFrame({
    'Característica': feature_names,
    'Peso': pca.components_[0]
}).sort_values('Peso', key=abs, ascending=False)
print(pc1_weights.head(5).to_string(index=False))

print("\nCaracterísticas con mayor peso en PC2:")
pc2_weights = pd.DataFrame({
    'Característica': feature_names,
    'Peso': pca.components_[1]
}).sort_values('Peso', key=abs, ascending=False)
print(pc2_weights.head(5).to_string(index=False))

# Visualización
fig = plt.figure(figsize=(18, 12))

# Plot 1: Scree plot - Varianza explicada
ax1 = plt.subplot(2, 3, 1)
n_components = 15
ax1.bar(range(1, n_components+1),
        pca_full.explained_variance_ratio_[:n_components],
        alpha=0.7, color='blue', edgecolor='black')
ax1.plot(range(1, n_components+1),
         pca_full.explained_variance_ratio_[:n_components],
         'ro-', linewidth=2, markersize=8)
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Varianza Explicada')
ax1.set_title('Scree Plot: Varianza por Componente')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Varianza acumulada
ax2 = plt.subplot(2, 3, 2)
ax2.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada,
        'b-', linewidth=2, marker='o', markersize=4)
ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2,
           label='95% varianza')
ax2.axhline(y=0.99, color='orange', linestyle='--', linewidth=2,
           label='99% varianza')
ax2.set_xlabel('Número de Componentes')
ax2.set_ylabel('Varianza Acumulada')
ax2.set_title('Varianza Acumulada vs Componentes')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Datos en 2D (PCA)
ax3 = plt.subplot(2, 3, 3)
colores = ['red', 'green']
for i, (nombre, color) in enumerate(zip(target_names, colores)):
    mask = y == i
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=nombre, c=color, s=60, alpha=0.6, edgecolors='black')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax3.set_title('PCA: Tumores en 2D (30D → 2D)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Heatmap de componentes principales (top 10 características)
ax4 = plt.subplot(2, 3, 4)
top_features = 10
components_df = pd.DataFrame(
    pca.components_[:2, :top_features],
    columns=feature_names[:top_features],
    index=['PC1', 'PC2']
)
im = ax4.imshow(components_df, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
ax4.set_xticks(range(top_features))
ax4.set_xticklabels(feature_names[:top_features], rotation=45, ha='right')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['PC1', 'PC2'])
ax4.set_title('Pesos de las Características en PCs (Top 10)')
plt.colorbar(im, ax=ax4, label='Peso')

# Plot 5: Biplot (PCA + vectores de características)
ax5 = plt.subplot(2, 3, 5)
# Puntos
for i, (nombre, color) in enumerate(zip(target_names, colores)):
    mask = y == i
    ax5.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=nombre, c=color, s=30, alpha=0.3)

# Vectores de las características más importantes (top 5)
scale = 3
top_5_indices = pc1_weights.head(5).index
for i in top_5_indices:
    feature_idx = list(feature_names).index(feature_names[i])
    ax5.arrow(0, 0,
             pca.components_[0, feature_idx] * scale,
             pca.components_[1, feature_idx] * scale,
             head_width=0.3, head_length=0.3, fc='black', ec='black', alpha=0.7)
    ax5.text(pca.components_[0, feature_idx] * scale * 1.15,
            pca.components_[1, feature_idx] * scale * 1.15,
            feature_names[feature_idx], fontsize=8, ha='center')

ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax5.set_title('Biplot: Datos + Vectores de Características')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Distribución de PC1 por clase
ax6 = plt.subplot(2, 3, 6)
for i, (nombre, color) in enumerate(zip(target_names, colores)):
    mask = y == i
    ax6.hist(X_pca[mask, 0], bins=30, alpha=0.6, label=nombre,
            color=color, edgecolor='black')
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax6.set_ylabel('Frecuencia')
ax6.set_title('Distribución de PC1 por Clase')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('PCA: Análisis de Cáncer de Mama (30 dimensiones → 2 dimensiones)',
            fontsize=14, y=0.995)
plt.tight_layout()
plt.show()

# Análisis de separabilidad
print("\n" + "=" * 70)
print("ANÁLISIS DE SEPARABILIDAD")
print("=" * 70)

# Calcular medias de cada clase en espacio PCA
mean_malignant = X_pca[y == 0].mean(axis=0)
mean_benign = X_pca[y == 1].mean(axis=0)

print(f"\nCentroide de tumores malignos en PCA: ({mean_malignant[0]:.2f}, {mean_malignant[1]:.2f})")
print(f"Centroide de tumores benignos en PCA: ({mean_benign[0]:.2f}, {mean_benign[1]:.2f})")
print(f"Distancia entre centroides: {np.linalg.norm(mean_malignant - mean_benign):.2f}")

print(f"\nConclusión: Las clases están {'bien' if np.linalg.norm(mean_malignant - mean_benign) > 5 else 'moderadamente'} separadas en el espacio PCA")
print(f"Esto indica que PCA preserva la información discriminativa importante.")
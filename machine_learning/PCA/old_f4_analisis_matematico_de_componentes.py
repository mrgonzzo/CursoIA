import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from f2_implementacion_modelo_pca import X
from f3_entrenar_modelo import pca_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# --- Suponiendo que ya tienes el pipeline entrenado ---
pca_model = pca_pipeline.named_steps['pca']

# --- Extraer matriz de componentes ---
componentes = pca_model.components_  # Cada fila = componente principal, cada columna = variable

# --- Crear DataFrame para mayor claridad ---
columnas = X.columns  # nombres de las variables originales
pc_labels = [f'PC{i+1}' for i in range(componentes.shape[0])]

loadings_df = pd.DataFrame(componentes, columns=columnas, index=pc_labels)

print("Matriz de cargas (loadings):")
print(loadings_df)

# --- Suponiendo que ya tienes el DataFrame de loadings ---
# loadings_df: filas = componentes, columnas = variables
#
#plt.figure(figsize=(8,6))
#sns.heatmap(
#    loadings_df,
#    annot=True,          # mostrar los valores
#    cmap='coolwarm',     # colores rojo-azul para positivo/negativo
#    center=0,            # centrado en 0
#    linewidths=0.5,
#    cbar_kws={'label': 'Carga (loading)'}
#)
#
#plt.title('Heatmap de cargas (loadings) de PCA', fontsize=14)
#plt.xlabel('Variables originales')
#plt.ylabel('Componentes principales')
#plt.show()


# --- Suponiendo que ya tienes el DataFrame de loadings ---
# loadings_df: filas = componentes, columnas = variables

plt.figure(figsize=(8,6))

# Mostrar matriz de cargas como imagen
plt.imshow(loadings_df, cmap='viridis', aspect='auto')

# Títulos de ejes
plt.xticks(ticks=np.arange(len(loadings_df.columns)), labels=loadings_df.columns, rotation=45)
plt.yticks(ticks=np.arange(len(loadings_df.index)), labels=loadings_df.index)

# Colorbar para interpretar los valores
plt.colorbar(label='Carga (loading)')

plt.title('Heatmap de cargas (loadings) usando imshow', fontsize=14)
plt.xlabel('Variables originales')
plt.ylabel('Componentes principales')

plt.show()

#pca_resumen_componentes
print("="*20)
print("RESUMEN COMPONENTES PCA")
print("="*20)

# ---------------------------
# 1. Cargar datos
# ---------------------------
df = pd.read_csv('USArrests.csv')

# ---------------------------
# 2. Variables predictoras
# ---------------------------
X = df.drop('USArrests', axis=1)
y = df['USArrests']

# ---------------------------
# 3. Crear y entrenar pipeline PCA
# ---------------------------
pca_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)  # Ajusta el número de componentes si quieres
)

X_pca = pca_pipeline.fit_transform(X)

# ---------------------------
# 4. Extraer modelo PCA y matriz de cargas
# ---------------------------
pca_model = pca_pipeline.named_steps['pca']
loadings = pd.DataFrame(
    pca_model.components_,
    columns=X.columns,
    index=[f'PC{i + 1}' for i in range(pca_model.components_.shape[0])]
)

print("=== MATRIZ DE CARGAS (loadings) ===")
print(loadings)
print("\nVarianza explicada por componente:")
print(pca_model.explained_variance_ratio_)
print("Varianza total explicada:", pca_model.explained_variance_ratio_.sum())

# ---------------------------
# 5. Generar resumen matemático
# ---------------------------
print("\n=== RESUMEN MATEMÁTICO DE COMPONENTES ===\n")
for pc in loadings.index:
    # Crear fórmula tipo: PC1 = 0.6*Murder + 0.7*Assault - 0.2*Rape + ...
    formula = " + ".join(
        [f"{loadings.loc[pc, col]:.3f}*{col}" if loadings.loc[pc, col] >= 0
         else f"({loadings.loc[pc, col]:.3f})*{col}"
         for col in loadings.columns]
    )
    print(f"{pc} = {formula}")

    # Mostrar variables que más contribuyen
    top_vars = loadings.loc[pc].abs().sort_values(ascending=False)
    print("Variables que más contribuyen:", list(top_vars.index[:3]))
    print()

    '''¿Qué representa matemáticamente cada componente principal?
    🧠 Concepto clave: Componente principal (PC)

Cada componente principal es una combinación lineal de las variables originales estandarizadas:

𝑃
𝐶
𝑖
=
𝑤
𝑖
1
𝑋
1
+
𝑤
𝑖
2
𝑋
2
+
⋯
+
𝑤
𝑖
𝑝
𝑋
𝑝
PC
i
	​

=w
i1
	​

X
1
	​

+w
i2
	​

X
2
	​

+⋯+w
ip
	​

X
p
	​


Donde:

𝑃
𝐶
𝑖
PC
i
	​

 → i-ésimo componente principal.

𝑋
𝑗
X
j
	​

 → j-ésima variable estandarizada (media 0, desviación 1).

𝑤
𝑖
𝑗
w
ij
	​

 → carga (loading) de la variable 
𝑋
𝑗
X
j
	​

 en el componente 
𝑃
𝐶
𝑖
PC
i
	​

.

🔹 Interpretación matemática

Dirección de máxima varianza

PC1 es el eje que maximiza la varianza de los datos.

PC2 es perpendicular a PC1 y maximiza la varianza residual.

Cada PC siguiente captura la mayor varianza posible sin repetir la información de los anteriores.

Combinación lineal

Cada PC es un nuevo eje en el espacio de variables, formado por pesos (loadings) sobre las variables originales.

Los pesos indican qué variables dominan ese componente.

Carga grande → variable influyente.

Carga pequeña → variable poco relevante.

Signo positivo o negativo → dirección de la relación.

Resumen de información

Cada PC es una nueva variable que resume la información original.

Los primeros PCs contienen la mayor parte de la varianza total, por lo que se usan para reducción de dimensionalidad.

🔹 Ejemplo concreto (simplificado)

Supongamos las cargas para PC1:

Variable	Carga
Murder	0.6
Assault	0.7
UrbanPop	0.1
Rape	-0.2

Entonces matemáticamente:

𝑃
𝐶
1
=
0.6
⋅
𝑀
𝑢
𝑟
𝑑
𝑒
𝑟
+
0.7
⋅
𝐴
𝑠
𝑠
𝑎
𝑢
𝑙
𝑡
+
0.1
⋅
𝑈
𝑟
𝑏
𝑎
𝑛
𝑃
𝑜
𝑝
−
0.2
⋅
𝑅
𝑎
𝑝
𝑒
PC1=0.6⋅Murder+0.7⋅Assault+0.1⋅UrbanPop−0.2⋅Rape

PC1 representa una dirección donde Murder y Assault dominan la variabilidad de los datos.

Es un nuevo eje que combina todas las variables originales en proporciones que maximizan la dispersión de los datos.

🔹 En palabras simples

Cada componente principal es una mezcla lineal de tus variables originales que crea un nuevo eje capturando la mayor variación posible.
Matemáticamente, es un vector en el espacio de características, perpendicular a los PCs anteriores, que resume la estructura de los datos.
    '''
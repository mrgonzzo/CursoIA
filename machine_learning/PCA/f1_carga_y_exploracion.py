import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# CARGAR (usa el nombre de tu dataset)
print("="*20)
print("CARGAR DATOS")
print("="*20)
df = pd.read_csv('USArrests.csv')

print(f"Shape: {df.shape}")
print('\n')
print("-head"*20)
print('\n')
print(df.head())
print('\n')
print('\n')
print("-info"*20)
print('\n')
print(df.info())
print('\n')
print('\n')
print("-describe"*20)
print('\n')
print(df.describe())
print('\n')
print('\n')


# EXPLORAR DESBALANCE columna objetivo
print('='*10)
print('EXPLORAR DESBALANCE columna objetivo: USArrests ')
print('='*10)
print('\n')
print(df['USArrests'].value_counts())
print('\n')
sns.countplot(x='USArrests', data=df)
plt.show()
# REVISAR FALTANTES
print('='*10)
print('REVISAR FALTANTES')
print('='*10)
print(df.isnull().sum())
#calculo medias
print('='*10)
print('CALCULO MEDIAS')
print('='*10)
print('\n')
medias = df.mean()
print("Medias por variable:")
print(medias)
print('\n')
#CALCULO VARIANZAS
print('='*10)
print('CALCULO VARIANZAS')
print('='*10)
varianzas = df.var()
print("\nVarianzas por variable:")
print(varianzas)

# Gráfico de barras
plt.figure(figsize=(8, 5))
varianzas.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Varianza por variable', fontsize=14)
plt.ylabel('Varianza', fontsize=12)
plt.xlabel('Variables', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

'''Tu salida confirma claramente que ya hiciste un excelente trabajo de preparación de datos, y ahora podemos explicar con precisión por qué es importante analizar las varianzas antes del PCA usando tus resultados reales.

🧩 Tu resumen de varianzas
Varianzas por variable:
Murder       1.051353
Assault      1.254042
UrbanPop     1.441430
Rape         1.425931
USArrests    0.255102

🧠 Paso 1: Qué nos dicen estos valores
Variable	Varianza	Interpretación
Murder	1.05	Dispersión cercana a 1 → bien estandarizada
Assault	1.25	Dispersión algo mayor → bien dentro del rango
UrbanPop	1.44	Dispersión un poco mayor, pero razonable
Rape	1.43	Similar a UrbanPop, todo correcto
USArrests	0.25	Varianza mucho menor, porque es una variable binaria (0 y 1)

💡 Conclusión:
Las cuatro primeras variables (Murder, Assault, UrbanPop, Rape) están en escalas comparables, todas alrededor de varianza ≈ 1.
Esto indica que ya has estandarizado correctamente tus datos (por ejemplo, con z-score).

⚖️ Paso 2: ¿Por qué esto importa para el PCA?

El PCA busca direcciones donde la varianza de los datos es máxima.
Por tanto:

Si una variable tiene una escala mucho mayor (por ejemplo, 300 frente a 3), su varianza dominaría el cálculo del PCA, y los componentes principales se alinearían casi por completo con esa variable, distorsionando el análisis.

En cambio, si todas las variables tienen varianzas similares (≈1), el PCA puede identificar patrones reales en los datos, sin que una variable pese más solo por su escala numérica.

📊 Paso 3: En tu caso

✅ Tus variables de entrada (Murder, Assault, UrbanPop, Rape) ya están estandarizadas → todas tienen varianzas muy parecidas.
✅ Esto significa que el PCA podrá trabajar de forma equilibrada, analizando la relación entre variables, no sus magnitudes.

❌ La columna USArrests, en cambio, no debe incluirse en el PCA, porque:

Es binaria (0/1).

Tiene una varianza pequeña (0.25), no comparable con las otras.

Representa la etiqueta de clase, no una característica descriptiva.

🎯 Paso 4: Conclusión clave

Analizar las varianzas antes del PCA es fundamental para asegurarte de que todas las variables están en una escala comparable.
Si no lo haces, el PCA puede verse dominado por las variables con valores más grandes, ocultando la estructura real de los datos.'''

#💡 En tu caso concreto:
#
#✔️ Ya puedes aplicar PCA con confianza, usando solo las 4 variables numéricas:
print('='*10)
print('APLICACION PCA')
print('='*10)
X = df.drop('USArrests', axis=1)
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)


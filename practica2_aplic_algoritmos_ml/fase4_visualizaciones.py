import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from practica2_aplic_algoritmos_ml.fase2_preparacion_de_datos import y_test, X
from practica2_aplic_algoritmos_ml.fase3_entrener_modelo import y_pred, modelo

#====================
#MATRIZ DE CONFUSIÓN
#====================
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title('Matriz de Confusión - Test Set')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()

#========================
#IMPORTANCIA DE VARIABLES
#========================
#print('='*10)
#print('IMPORTANCIA DE VARIABLES')
#print('='*10)
#
## Asegúrate de que X.columns esté disponible desde la Fase 2
#coef_df = pd.DataFrame(modelo.coef_, columns=X.columns, index = modelo.classes_)
#print("\n=== COEFICIENTES DEL MODELO ===")
#print(coef_df)
#
#coef_df.T.plot(kind='barh', figsize=(10, 6))
#plt.title('Importancia de variables por clase')
#plt.xlabel('Coeficiente (peso)')
#plt.tight_layout()
#plt.show()

# ========================
# IMPORTANCIA DE VARIABLES (CORREGIDO)
# ========================

# Extraemos la única fila de coeficientes (índice [0]) para la clasificación binaria
# Esto resuelve el error de dimensionamiento.
coeficientes_binarios = modelo.coef_[0]

# Construimos el DataFrame: Coeficientes como datos, nombres de features como índice.
coef_df = pd.DataFrame(coeficientes_binarios,
                       index=X.columns,  # Usamos las columnas de X como índice (las features)
                       columns=['Coeficiente']) # Damos un nombre descriptivo a la columna

print("\n=== COEFICIENTES DEL MODELO ===")
print(coef_df.sort_values(by='Coeficiente', ascending=False).to_string(header=True))

# Generar la visualización
plt.figure(figsize=(10, 6))
# Trazamos el DataFrame de coeficientes
coef_df['Coeficiente'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Importancia de Variables (Impacto en la Probabilidad de Saturación)')
plt.xlabel('Coeficiente Estandarizado (Impacto en Log-Odds)')
plt.ylabel('Variable de Sistema')
plt.tight_layout()
plt.show()
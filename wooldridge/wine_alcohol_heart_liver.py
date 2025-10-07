# ===============================================
# ANÁLISIS MCO - DATASET WINE (REGRESIÓN MÚLTIPLE)
# ===============================================

# 1. Importar librerías necesarias
# --------------------------------
import pandas as pd
import wooldridge
import statsmodels.formula.api as smf

# 2. Cargar el dataset 'wine'
# ---------------------------
try:
    df_wine = wooldridge.data('wine')
    print("Dataset 'wine' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    exit()

# 3. Definir y Estimación del Modelo MCO
# --------------------------------------
# Objetivo: Estimar el consumo de alcohol (alcohol)
#           usando las muertes cardíacas (heart) y hepáticas (liver).
# Modelo: alcohol = beta_0 + beta_1 * heart + beta_2 * liver + u

formula = 'alcohol ~ heart + liver' # ¡Fórmula de Regresión Múltiple!

# Aplicar MCO (OLS - Ordinary Least Squares)
modelo_mco = smf.ols(formula, data=df_wine)

# Entrenar el modelo
resultados = modelo_mco.fit()

# 4. Mostrar Resultados de la Regresión
# -------------------------------------
print("\n" + "="*70)
print("      RESULTADOS DE LA REGRESIÓN LINEAL MÚLTIPLE (MCO)")
print("    Variable Dependiente: Consumo de Alcohol de Vino (alcohol)")
print("    Variables Independientes: Muertes Cardíacas (heart) y Hepáticas (liver)")
print("="*70 + "\n")

print(resultados.summary())

# 5. Interpretación Clave (Opcional)
# ----------------------------------
coef_heart = resultados.params['heart']
coef_liver = resultados.params['liver']
r_squared = resultados.rsquared

print("\n" + "="*70)
print("INTERPRETACIÓN CLAVE:")
print(f"Coeficiente de 'heart': {coef_heart:.4f}")
print(f"Coeficiente de 'liver': {coef_liver:.4f}")
print(f"R-cuadrado del modelo: {r_squared:.4f}")

print("\nInterpretación:")
print(f"- El coeficiente de 'heart' ({coef_heart:.4f}) indica el cambio en el consumo de alcohol cuando la tasa de muertes cardíacas aumenta en 1, MANTENIENDO CONSTANTE la tasa de muertes hepáticas.")
print(f"- El R-cuadrado ({r_squared:.4f}) sugiere que este porcentaje de la variación total en el consumo de alcohol es explicado conjuntamente por ambas tasas de mortalidad.")
print("="*70)

# ===============================================
# FIN DEL CÓDIGO
# ===============================================
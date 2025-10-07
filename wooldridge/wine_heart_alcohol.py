# ===============================================
# ANALISIS MCO - DATASET WINE (LIBRERÍA WOOLDRIDGE)
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
# Objetivo: Estimar el impacto del consumo de alcohol (alcohol)
#           sobre las muertes por enfermedad cardíaca (heart).
# Modelo: heart = beta_0 + beta_1 * alcohol + u

formula = 'heart ~ alcohol'

# Aplicar MCO (OLS - Ordinary Least Squares)
modelo_mco = smf.ols(formula, data=df_wine)

# Entrenar el modelo
resultados = modelo_mco.fit()

# 4. Mostrar Resultados de la Regresión
# -------------------------------------
print("\n" + "="*50)
print("           RESULTADOS DE LA REGRESIÓN MCO")
print("    Variable Dependiente: Muertes por Enfermedad Cardíaca (heart)")
print("    Variable Independiente: Consumo de Alcohol de Vino (alcohol)")
print("="*50 + "\n")

print(resultados.summary())

# 5. Interpretación Breve del Coeficiente (Opcional)
# --------------------------------------------------
coef_alcohol = resultados.params['alcohol']
p_valor_alcohol = resultados.pvalues['alcohol']

print("\n" + "="*50)
print("INTERPRETACIÓN CLAVE:")
print(f"Coeficiente de 'alcohol': {coef_alcohol:.4f}")
print(f"P-valor asociado: {p_valor_alcohol:.4f}")

if p_valor_alcohol < 0.05:
    significancia = "es estadísticamente significativo al 5%."
else:
    significancia = "NO es estadísticamente significativo al 5%."

print(f"La relación entre el consumo de alcohol de vino y las muertes cardíacas {significancia}")
print("Un coeficiente negativo sugiere que un mayor consumo se asocia con menos muertes cardíacas.")
print("="*50)

# ===============================================
# FIN DEL CÓDIGO
# ===============================================
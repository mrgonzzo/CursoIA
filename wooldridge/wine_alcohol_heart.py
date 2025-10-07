# ===============================================
# ANÁLISIS MCO - DATASET WINE (FÓRMULA INVERSA)
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
# Objetivo: Estimar el impacto de las muertes cardíacas (heart)
#           sobre el consumo de alcohol (alcohol).
# Modelo: alcohol = beta_0 + beta_1 * heart + u

formula = 'alcohol ~ heart' # ¡Aquí está la fórmula invertida!

# Aplicar MCO (OLS - Ordinary Least Squares)
modelo_mco = smf.ols(formula, data=df_wine)

# Entrenar el modelo
resultados = modelo_mco.fit()

# 4. Mostrar Resultados de la Regresión
# -------------------------------------
print("\n" + "="*50)
print("           RESULTADOS DE LA REGRESIÓN MCO")
print("    Variable Dependiente: Consumo de Alcohol de Vino (alcohol)")
print("    Variable Independiente: Muertes por Enfermedad Cardíaca (heart)")
print("="*50 + "\n")

print(resultados.summary())

# 5. Interpretación Breve del Coeficiente (Opcional)
# --------------------------------------------------
coef_heart = resultados.params['heart']
p_valor_heart = resultados.pvalues['heart']

print("\n" + "="*50)
print("INTERPRETACIÓN CLAVE:")
print(f"Coeficiente de 'heart': {coef_heart:.4f}")
print(f"P-valor asociado: {p_valor_heart:.4f}")

# La interpretación ahora es: ¿Cuánto varía el consumo de alcohol (alcohol)
# cuando la tasa de muertes cardíacas (heart) varía en una unidad?
print("Interpretación: Un aumento de 1 en la tasa de muertes cardíacas se asocia con un cambio de")
print(f"{coef_heart:.4f} litros en el consumo per cápita de alcohol de vino.")
print("="*50)

# ===============================================
# FIN DEL CÓDIGO
# ===============================================
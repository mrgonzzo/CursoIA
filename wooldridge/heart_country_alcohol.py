# =========================================================
# ANÁLISIS MCO - DATASET WINE (COUNTRY COMO VARIABLE DUMMY)
# =========================================================

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
#           usando la tasa de muertes cardíacas (heart) y el país (country).
# Advertencia: El modelo está altamente sobreespecificado debido a la inclusión
#              de 20 variables dummy para country con solo 21 observaciones.

formula = 'alcohol ~ heart + country' # ¡'country' se convierte automáticamente en variables dummy!

# Aplicar MCO (OLS - Ordinary Least Squares)
modelo_mco = smf.ols(formula, data=df_wine)

# Entrenar el modelo
resultados = modelo_mco.fit()

# 4. Mostrar Resultados de la Regresión
# -------------------------------------
print("\n" + "="*75)
print("      RESULTADOS DE LA REGRESIÓN LINEAL MÚLTIPLE (MCO)")
print("    Variable Dependiente: Consumo de Alcohol de Vino (alcohol)")
print("    Variables Independientes: Muertes Cardíacas (heart) y País (como Dummies)")
print("="*75 + "\n")

print(resultados.summary())

# 5. Interpretación Clave (Opcional)
# ----------------------------------
r_squared = resultados.rsquared

print("\n" + "="*75)
print("ADVERTENCIA Y CONTEXTO:")
print("Este modelo incluye 20 variables dummy de país. Con solo 21 observaciones,")
print("esto consume casi todos tus grados de libertad. El modelo está sobreajustado.")
print(f"El R-cuadrado ({r_squared:.4f}) será muy alto o 1, pero los resultados")
print("individuales de los coeficientes de país (aparte de la referencia) serán inestables.")
print("="*75)

# ===============================================
# FIN DEL CÓDIGO
# ===============================================
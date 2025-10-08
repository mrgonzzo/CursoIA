import pandas as pd
import numpy as np
import wooldridge as woo
import statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_white

# Cargar datos de Mroz
mroz = woo.dataWoo('mroz')
labels = mroz.head()
print(labels)
print(f"\nDimensiones: {mroz.shape}")

# ============================================
# 1. Modelo MCO simple - Ecuación salarial
# ============================================
print("\n" + "="*50)
print("MODELO 1: Ecuación salarial básica")
print("=!"*50)
mod_mco = ols('lwage ~ educ + exper + hushrs', data=mroz).fit()
print(mod_mco.summary())
mod_mco1 = ols('lwage ~ educ + exper + inlf', data=mroz).fit()
print(mod_mco1.summary())

print("\n" + "!"*50)
print("MODELO 1: Ecuación salarial básica y test het_white")
print("!"*50)
white_test = het_white(mod_mco.resid, exog=mod_mco.model.exog)
print(dict(zip(labels,white_test)))
print("\n" + "!"*50)
# ============================================
# 2. Modelo MCO mejorado
# ============================================
print("\n" + "="*50)
print("MODELO 2: Ecuación salarial mejorada")
print("=/"*50)
mod_mco2 = ols('lwage ~ educ + exper + kidslt6 + kidsge6', data=mroz).fit()
print(mod_mco2.summary())

# ============================================
# 3. EJERCICIO PRÁCTICO - Modelo de Participación Laboral
# ============================================
print("\n" + "=*"*50)
print("EJERCICIO: Participación en mercado laboral")
print("="*50)

# Paso 1: MCO para inlf
print("\n1. Modelo MCO (Modelo de Probabilidad Lineal)")
mod_lpm = ols('inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6', data=mroz).fit()
print(mod_lpm.summary())

# ============================================
# 4. EJERCICIO PRÁCTICO - Modelo completo
# ============================================
print("\n" + "=*"*50)
print("4. EJERCICIO PRÁCTICO - Modelo completo")
print("EJERCICIO: Participación en mercado laboral")
print("="*50)
mod_mcoT = ols('lwage ~ nwifeinc + educ + I(educ**2) +  exper +  age + kidslt6 + kidsge6', data=mroz).fit()
print(mod_mcoT.summary())

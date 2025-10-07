import pandas as pd
import numpy as np
import wooldridge as woo
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Cargar datos de Mroz
mroz = woo.dataWoo('mroz')
print(mroz.head())
print(f"\nDimensiones: {mroz.shape}")

# ============================================
# 1. Modelo MCO simple - Ecuación salarial
# ============================================
#print("\n" + "="*50)
#print("MODELO 1: Ecuación salarial básica")
#print("="*50)

#mod_mco = ols('lwage ~ educ + exper', data=mroz).fit()
#print(mod_mco.summary())


#mod_mco1 = ols('lwage ~ educ + exper + inlf', data=mroz).fit()
#print(mod_mco1.summary())


# ============================================
# 2. Modelo MCO mejorado
# ============================================
#print("\n" + "="*50)
#print("MODELO 2: Ecuación salarial mejorada")
#print("="*50)

#mod_mco2 = ols('lwage ~ educ + exper + kidslt6 + kidsge6', data=mroz).fit()
#print(mod_mco2.summary())

# ============================================
# 3. EJERCICIO PRÁCTICO - Modelo de Participación Laboral
# ============================================
#print("\n" + "="*50)
#print("EJERCICIO: Participación en mercado laboral")
#print("="*50)

# Paso 1: MCO para inlf
#print("\n1. Modelo MCO (Modelo de Probabilidad Lineal)")
#mod_lpm = ols('inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6', data=mroz).fit()
#print(mod_lpm.summary())

# ============================================
# 4. EJERCICIO PRÁCTICO - Modelo completo
# ============================================
mod_mcoT = ols('lwage ~ inlf + hours + kidslt6 + kidsge6 + age + educ + wage + repwage + hushrs + husage + huseduc + huswage + faminc + mtr + motheduc + fatheduc + unem + city + exper + nwifeinc + lwage + expersq', data=mroz).fit()
print(mod_mcoT.summary())

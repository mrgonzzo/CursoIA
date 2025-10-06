import pandas as pd
import numpy as np
import wooldridge as woo
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Mostrar descripción del dataset (opcional)
woo.dataWoo('mroz', description=True)

# Cargar datos de Mroz
mroz = woo.dataWoo('mroz')  # sin description=True
print(mroz.head())
print(f"\nDimensiones: {mroz.shape}")

# ============================================
# 1. Modelo MCO simple - Ecuación salarial
# ============================================
print("\n" + "="*50)
print("MODELO 1: Ecuación salarial básica")
print("="*50)

mod_mco = ols('lwage ~ educ + exper + city', data=mroz).fit()
print(mod_mco.summary())

# ============================================
# 2. Modelo MCO mejorado
# ============================================
print("\n" + "="*50)
print("MODELO 2: Ecuación salarial mejorada")
print("="*50)

mod_mco2 = ols('lwage ~ educ + exper + kidslt6 + kidsge6', data=mroz).fit()
print(mod_mco2.summary())
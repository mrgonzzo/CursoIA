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
#mod_mco1 = ols('lwage ~ educ + exper + inlf', data=mroz).fit()
#print(mod_mco1.summary())

print("\n" + "!"*50)
print("MODELO 1: Ecuación salarial básica y test het_white")
print("!"*50)
white_test = het_white(mod_mco.resid, exog=mod_mco.model.exog)
print(dict(zip(labels,white_test)))
print("\n" + "!"*50)
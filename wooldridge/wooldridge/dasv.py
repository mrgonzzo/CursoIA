import pandas as pd
import numpy as np
import wooldridge as woo
import statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_white
print("?"*60)
print(f"Tipo de variable woo.dataWoo(): {type(woo.dataWoo())}")
print("?"*60)
print("!"*60)
listado_datasets = woo.dataWoo()
print(f"Tipo de variable listado_datasets: {type(listado_datasets)}")
print("!"*60)
print("-"*60)
i = 0
for e in listado_datasets:
    i = i + 1
    print(i)
    print(e)
    #test[i]=e
print("-"*60)
print("*"*60)
test = list(woo.data())
print("*"*60)
print(f"Tipo de variable test: {type(test)}")
print("="*60)
print(test)
print( "="*60)
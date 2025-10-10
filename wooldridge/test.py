import pandas as pd
import numpy as np
import wooldridge as woo
import statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_white



dataset_names = woo.data()
type(dataset_names)
lista  = list(dataset_names)
type(lista)
print( lista )

import wooldridge as woo
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
hprice1 = woo.dataWoo('hprice1')

# Estimar modelo
modelo = ols('price ~ lotsize + sqrft + bdrms', data=hprice1).fit()

print("="*60)
print("MODELO: price ~ lotsize + sqrft + bdrms")
print("="*60)

# Coeficientes
print("\nCOEFICIENTES:")
print(modelo.summary().tables[1])

print(f"\nR² = {modelo.rsquared:.4f}")
print(f"R² Ajustado = {modelo.rsquared_adj:.4f}")
print(f"N = {int(modelo.nobs)}")

# Residuos
residuos = modelo.resid

# Test de Breusch-Pagan
print("\n" + "="*60)
print("TEST DE BREUSCH-PAGAN (Heterocedasticidad)")
print("="*60)
bp_stat, bp_pval, _, _ = het_breuschpagan(residuos, modelo.model.exog)
print(f"Estadístico LM: {bp_stat:.4f}")
print(f"P-valor: {bp_pval:.4f}")

# Test de Durbin-Watson
print("\n" + "="*60)
print("TEST DE DURBIN-WATSON (Autocorrelación)")
print("="*60)
dw_stat = durbin_watson(residuos)
print(f"Estadístico DW: {dw_stat:.4f}")

# Test de Jarque-Bera
print("\n" + "="*60)
print("TEST DE JARQUE-BERA (Normalidad)")
print("="*60)
jb_stat, jb_pval = jarque_bera(residuos)
print(f"Estadístico JB: {jb_stat:.4f}")
print(f"P-valor: {jb_pval:.4f}")

print("\n" + "="*60)

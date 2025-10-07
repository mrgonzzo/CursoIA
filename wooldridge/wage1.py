#código sencillo para ver el efecto de la educación sobre el logaritmo del salario (lwage)
import wooldridge
import statsmodels.formula.api as smf

df = wooldridge.data('wage1')

# Modelo: ln(wage) = beta_0 + beta_1 * educ + u
resultados = smf.ols('lwage ~ educ', data=df).fit()

print(resultados.summary())
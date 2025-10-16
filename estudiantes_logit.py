import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

df = pd.read_csv(
    r'C:\Users\tarde\Desktop\ProgramacionCursoIA\DATOS\datos_ejercicio1_estudiantes.csv')

X = df[['horas_estudio', 'horas_sueño']]
y = df['aprobado']

# Agregar constante (intercepto)
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Usar Logit de statsmodels
modelo = sm.Logit(y_train, X_train)
resultado = modelo.fit()

# Summary original
print(resultado.summary())

# ===================================================================
# TABLA DE COEFICIENTES CON SIGNIFICANCIA
# ===================================================================
print("\n" + "=" * 80)
print("COEFICIENTES DEL MODELO LOGIT BINARIO")
print("=" * 80 + "\n")

# Crear tabla de coeficientes
coef_table = pd.DataFrame({
    'Coeficiente': resultado.params,
    'Std.Error': resultado.bse,
    'z-value': resultado.tvalues,
    'P>|z|': resultado.pvalues,
    'IC 2.5%': resultado.conf_int()[0],
    'IC 97.5%': resultado.conf_int()[1]
})

coef_table.index = ['Constante', 'Horas Estudio', 'Horas Sueño']


# Agregar significancia
def sig_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    elif pval < 0.1:
        return '.'
    else:
        return ''


coef_table['Sig.'] = coef_table['P>|z|'].apply(sig_stars)

print(coef_table.to_string(float_format='%.6f'))
print("\nNivel de significancia: '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")

# ===================================================================
# ODDS RATIOS
# ===================================================================
print("\n\n" + "=" * 80)
print("ODDS RATIOS (Razones de Momios)")
print("=" * 80 + "\n")

odds_ratios = pd.DataFrame({
    'Odds Ratio': np.exp(resultado.params),
    'IC 2.5%': np.exp(resultado.conf_int()[0]),
    'IC 97.5%': np.exp(resultado.conf_int()[1]),
    'P>|z|': resultado.pvalues
})

odds_ratios.index = ['Constante', 'Horas Estudio', 'Horas Sueño']
odds_ratios['Sig.'] = odds_ratios['P>|z|'].apply(sig_stars)

print(odds_ratios.to_string(float_format='%.6f'))

print("\nInterpretación de Odds Ratios:")
print("  - OR > 1: Aumenta las probabilidades de aprobar")
print("  - OR < 1: Disminuye las probabilidades de aprobar")
print("  - OR = 1: No tiene efecto")

# ===================================================================
# EFECTOS MARGINALES
# ===================================================================
print("\n\n" + "=" * 80)
print("EFECTOS MARGINALES (en la media de las variables)")
print("=" * 80 + "\n")

# Calcular efectos marginales
margeff = resultado.get_margeff(at='mean')

# CORRECCIÓN: manejar correctamente los intervalos de confianza
conf_int_margeff = margeff.conf_int()

margeff_table = pd.DataFrame({
    'dy/dx': margeff.margeff[0],
    'Std.Error': margeff.margeff_se[0],
    'z-value': margeff.margeff[0] / margeff.margeff_se[0],
    'P>|z|': margeff.pvalues[0],
    'IC 2.5%': conf_int_margeff[:, 0],
    'IC 97.5%': conf_int_margeff[:, 1]
})

margeff_table.index = ['Horas Estudio', 'Horas Sueño']
margeff_table['Sig.'] = margeff_table['P>|z|'].apply(sig_stars)

print(margeff_table.to_string(float_format='%.6f'))

print(f"\nEvaluado en:")
print(f"  Horas Estudio = {X_train['horas_estudio'].mean():.2f}")
print(f"  Horas Sueño   = {X_train['horas_sueño'].mean():.2f}")

# ===================================================================
# INTERPRETACIÓN DETALLADA
# ===================================================================
print("\n\n" + "=" * 80)
print("INTERPRETACIÓN DETALLADA DE LOS RESULTADOS")
print("=" * 80 + "\n")

print("1. COEFICIENTES (Log-Odds):")
for var in coef_table.index[1:]:  # Sin constante
    coef = coef_table.loc[var, 'Coeficiente']
    pval = coef_table.loc[var, 'P>|z|']
    sig = coef_table.loc[var, 'Sig.']

    direccion = "positivo" if coef > 0 else "negativo"
    significativo = "significativo" if pval < 0.05 else "NO significativo"

    print(f"\n   {var}:")
    print(f"   - Coeficiente: {coef:.4f} {sig}")
    print(f"   - Efecto {direccion} y {significativo} (p={pval:.4f})")

print("\n\n2. ODDS RATIOS:")
for var in odds_ratios.index[1:]:
    or_val = odds_ratios.loc[var, 'Odds Ratio']
    pval = odds_ratios.loc[var, 'P>|z|']

    if or_val > 1:
        cambio = (or_val - 1) * 100
        print(f"\n   {var}: OR = {or_val:.4f}")
        print(f"   → Aumentar 1 unidad multiplica las probabilidades por {or_val:.2f}")
        print(f"   → Es decir, aumenta las probabilidades en {cambio:.1f}%")
    elif or_val < 1:
        cambio = (1 - or_val) * 100
        print(f"\n   {var}: OR = {or_val:.4f}")
        print(f"   → Aumentar 1 unidad multiplica las probabilidades por {or_val:.2f}")
        print(f"   → Es decir, disminuye las probabilidades en {cambio:.1f}%")

print("\n\n3. EFECTOS MARGINALES (más intuitivos):")
for var in margeff_table.index:
    ef = margeff_table.loc[var, 'dy/dx']
    pval = margeff_table.loc[var, 'P>|z|']
    sig = margeff_table.loc[var, 'Sig.']

    cambio_prob = ef * 100
    direccion = "aumenta" if ef > 0 else "disminuye"

    print(f"\n   {var}: {ef:.6f} {sig}")
    print(f"   → Aumentar 1 unidad {direccion} la probabilidad")
    print(f"     de aprobar en {abs(cambio_prob):.2f} puntos porcentuales")

# ===================================================================
# PREDICCIONES Y EVALUACIÓN
# ===================================================================
print("\n\n" + "=" * 80)
print("EVALUACIÓN DEL MODELO")
print("=" * 80 + "\n")

# Predicciones
y_pred_prob = resultado.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Matriz de confusión
print("\n\nMatriz de Confusión:")
print("-" * 40)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Real: No Aprobado (0)', 'Real: Aprobado (1)'],
    columns=['Pred: No Aprobado (0)', 'Pred: Aprobado (1)']
)

print(cm_df.to_string())

# Reporte de clasificación
print("\n\nReporte de Clasificación:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['No Aprobado', 'Aprobado']))

# ===================================================================
# ESTADÍSTICOS DEL MODELO
# ===================================================================
print("\n" + "=" * 80)
print("ESTADÍSTICOS DE BONDAD DE AJUSTE")
print("=" * 80 + "\n")

stats_table = pd.DataFrame({
    'Estadístico': [
        'Log-Likelihood',
        'AIC',
        'BIC',
        'Pseudo R²',
        'N. Observaciones'
    ],
    'Valor': [
        resultado.llf,
        resultado.aic,
        resultado.bic,
        resultado.prsquared,
        resultado.nobs
    ]
})

print(stats_table.to_string(index=False, float_format='%.4f'))

# ===================================================================
# EJEMPLO DE PREDICCIÓN
# ===================================================================
print("\n\n" + "=" * 80)
print("EJEMPLO DE PREDICCIÓN")
print("=" * 80 + "\n")

# Crear ejemplos
ejemplos = pd.DataFrame({
    'const': [1, 1, 1],
    'horas_estudio': [2, 5, 8],
    'horas_sueño': [6, 7, 8]
})

prob_ejemplos = resultado.predict(ejemplos)

print("Probabilidad de aprobar según horas de estudio y sueño:\n")
for i in range(len(ejemplos)):
    print(f"  Estudio: {ejemplos.iloc[i]['horas_estudio']:.0f}h, "
          f"Sueño: {ejemplos.iloc[i]['horas_sueño']:.0f}h "
          f"→ P(Aprobar) = {prob_ejemplos.iloc[i]:.2%}")

print("\n\n" + "=" * 80)
print("RESUMEN EJECUTIVO")
print("=" * 80)
print("""
Este modelo logit binario predice la probabilidad de aprobar basándose en:
  - Horas de estudio
  - Horas de sueño

Los EFECTOS MARGINALES son la forma más directa de interpretación:
  → Indican el cambio en la PROBABILIDAD (no en odds) de aprobar
  → Se expresan en puntos porcentuales
  → Son más intuitivos que los coeficientes o los odds ratios
""")
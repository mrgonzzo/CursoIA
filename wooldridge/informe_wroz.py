import wooldridge as woo
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def informe_modelo_simple(modelo, nombre_modelo="Modelo MCO"):
    """
    Genera un informe simplificado y didáctico de un modelo MCO
    """
    print("\n" + "=" * 70)
    print(f"📊 INFORME DEL MODELO: {nombre_modelo}")
    print("=" * 70)

    # 1. ECUACIÓN AJUSTADA
    print("\n1️⃣ ECUACIÓN ESTIMADA:")
    print("-" * 70)
    variable_dep = modelo.model.endog_names
    ecuacion = f"{variable_dep} = {modelo.params.iloc[0]:.4f}"

    for i, var in enumerate(modelo.model.exog_names[1:], 1):
        coef = modelo.params.iloc[i]
        signo = "+" if coef >= 0 else ""
        ecuacion += f" {signo} {coef:.4f}*{var}"

    print(ecuacion)

    # 2. COEFICIENTES Y SIGNIFICANCIA
    print("\n2️⃣ COEFICIENTES Y SIGNIFICANCIA:")
    print("-" * 70)
    print(f"{'Variable':<20} {'Coeficiente':>12} {'P-valor':>12} {'Significativo?':>15}")
    print("-" * 70)

    for var in modelo.model.exog_names:
        coef = modelo.params[var]
        pval = modelo.pvalues[var]

        if pval < 0.01:
            sig = "*** (1%)"
        elif pval < 0.05:
            sig = "** (5%)"
        elif pval < 0.10:
            sig = "* (10%)"
        else:
            sig = "NO"

        print(f"{var:<20} {coef:>12.4f} {pval:>12.4f} {sig:>15}")

    print("\nNota: *** = Significativo al 1%, ** = al 5%, * = al 10%")

    # 3. BONDAD DE AJUSTE
    print("\n3️⃣ BONDAD DE AJUSTE:")
    print("-" * 70)
    print(f"R² (R cuadrado)          : {modelo.rsquared:.4f}")
    print(f"R² Ajustado              : {modelo.rsquared_adj:.4f}")
    print(f"Número de observaciones  : {int(modelo.nobs)}")

    interpretacion_r2 = ""
    if modelo.rsquared < 0.3:
        interpretacion_r2 = "Bajo - El modelo explica poco de la variabilidad"
    elif modelo.rsquared < 0.6:
        interpretacion_r2 = "Moderado - El modelo tiene ajuste aceptable"
    else:
        interpretacion_r2 = "Alto - El modelo explica bien la variabilidad"

    print(f"\n💡 Interpretación: {interpretacion_r2}")

    # 4. SUPUESTOS DEL MODELO
    print("\n4️⃣ VERIFICACIÓN DE SUPUESTOS:")
    print("-" * 70)

    residuos = modelo.resid

    # A) Normalidad (Jarque-Bera)
    print("\n📌 A) NORMALIDAD DE RESIDUOS (Test de Jarque-Bera)")
    jb_stat, jb_pval = jarque_bera(residuos)
    print(f"   Estadístico: {jb_stat:.4f}")
    print(f"   P-valor: {jb_pval:.4f}")
    print(f"   H0: Los residuos se distribuyen normalmente")

    if jb_pval > 0.05:
        print(f"   ✅ DECISIÓN: NO se rechaza H0 (p-valor = {jb_pval:.4f} > 0.05)")
        print(f"   📝 Los residuos SÍ son normales")
    else:
        print(f"   ❌ DECISIÓN: Se rechaza H0 (p-valor = {jb_pval:.4f} < 0.05)")
        print(f"   📝 Los residuos NO son normales (puede afectar inferencia)")

    # B) Homocedasticidad (Breusch-Pagan)
    print("\n📌 B) HOMOCEDASTICIDAD (Test de Breusch-Pagan)")
    bp_stat, bp_pval, _, _ = het_breuschpagan(residuos, modelo.model.exog)
    print(f"   Estadístico: {bp_stat:.4f}")
    print(f"   P-valor: {bp_pval:.4f}")
    print(f"   H0: Hay homocedasticidad (varianza constante)")

    if bp_pval > 0.05:
        print(f"   ✅ DECISIÓN: NO se rechaza H0 (p-valor = {bp_pval:.4f} > 0.05)")
        print(f"   📝 HAY homocedasticidad - Supuesto cumplido")
    else:
        print(f"   ❌ DECISIÓN: Se rechaza H0 (p-valor = {bp_pval:.4f} < 0.05)")
        print(f"   📝 HAY heterocedasticidad - Usar errores robustos")

    # C) Autocorrelación (Breusch-Godfrey)
    print("\n📌 C) NO AUTOCORRELACIÓN (Test de Breusch-Godfrey)")
    try:
        bg_test = acorr_breusch_godfrey(modelo, nlags=1)
        bg_stat, bg_pval = bg_test[0], bg_test[1]
        print(f"   Estadístico: {bg_stat:.4f}")
        print(f"   P-valor: {bg_pval:.4f}")
        print(f"   H0: No hay autocorrelación")

        if bg_pval > 0.05:
            print(f"   ✅ DECISIÓN: NO se rechaza H0 (p-valor = {bg_pval:.4f} > 0.05)")
            print(f"   📝 NO hay autocorrelación - Supuesto cumplido")
        else:
            print(f"   ❌ DECISIÓN: Se rechaza H0 (p-valor = {bg_pval:.4f} < 0.05)")
            print(f"   📝 SÍ hay autocorrelación - Considerar errores HAC")
    except:
        print(f"   ⚠️  No se pudo calcular (no aplicable para datos de corte transversal)")

    # 5. GRÁFICO DE RESIDUOS
    print("\n5️⃣ GRÁFICOS DE DIAGNÓSTICO:")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma de residuos
    axes[0].hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Residuos')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(alpha=0.3)

    # Residuos vs valores ajustados
    axes[1].scatter(modelo.fittedvalues, residuos, alpha=0.5, color='steelblue')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Residuos vs Valores Ajustados', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Valores Ajustados')
    axes[1].set_ylabel('Residuos')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("✅ INFORME COMPLETO")
    print("=" * 70 + "\n")


# ============================================
# EJEMPLO DE USO CON TU EJERCICIO
# ============================================

# Cargar datos
mroz = woo.dataWoo('mroz')

# Filtrar solo mujeres que trabajan (para lwage)
mroz_trabajando = mroz[mroz['inlf'] == 1].copy()

print("\n" + "=" * 70)
print("EJERCICIO: Determinantes del salario (log)")
print("=" * 70)

# MODELO 1: Con término cuadrático
print("\n🔹 MODELO 1: Incluyendo educación al cuadrado")
mod1 = ols('lwage ~ nwifeinc + educ + I(educ**2) + exper + age + kidslt6 + kidsge6',
           data=mroz_trabajando).fit()
informe_modelo_simple(mod1, "Modelo con educ²")

# MODELO 2: Sin término cuadrático
print("\n\n🔹 MODELO 2: Sin educación al cuadrado")
mod2 = ols('lwage ~ nwifeinc + educ + exper + age + kidslt6 + kidsge6',
           data=mroz_trabajando).fit()
informe_modelo_simple(mod2, "Modelo lineal simple")
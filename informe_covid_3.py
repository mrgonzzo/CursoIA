import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera

# ================== RUTAS ==================
CSV_PATH = r"C:/Users/tarde/Desktop/ProgramacionCursoIA/DATOS/covid_data.csv"
OUTPUT_DIR = r"C:/Users/tarde/Desktop/ProgramacionCursoIA/graficos_covid"
INFORME_PATH = r"C:/Users/tarde/Desktop/ProgramacionCursoIA/informe_covid.txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== CARGA DATOS ==================
df = pd.read_csv(CSV_PATH)
print("Columnas disponibles:", df.columns.tolist())

# ================== PARSEO FECHAS EN ESPAÑOL ==================
meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}


def parse_fecha_es(fecha_texto):
    """Convierte '6 de enero de 2020' a datetime"""
    try:
        partes = fecha_texto.strip().split(' de ')
        dia = int(partes[0])
        mes = meses[partes[1].lower()]
        anio = int(partes[2])
        return datetime.datetime(anio, mes, dia)
    except:
        return pd.NaT


# Detectar columna de fecha y convertir
fecha_col = None
for col in ['Fecha_r', 'Fecha', 'Fecha_num']:
    if col in df.columns:
        fecha_col = col
        break
if fecha_col is None:
    raise ValueError("No se encontró ninguna columna de fecha en el CSV")

if df[fecha_col].dtype == object:
    df['Fecha_dt'] = df[fecha_col].apply(parse_fecha_es)
else:
    df['Fecha_dt'] = pd.to_datetime(df[fecha_col], errors='coerce', dayfirst=True)

if df['Fecha_dt'].isna().sum() > 0:
    print("Advertencia: algunas fechas no se pudieron convertir a datetime")

# ================== VARIABLES ==================
for col in ['Hay_vacuna', 'Hay_Confinamiento', 'Año']:
    if col in df.columns:
        df[col] = df[col].astype(int)
    else:
        df[col] = 0


# ================== FUNCIONES ==================
def ajustar_ols(dep_var, indep_vars):
    X = df[indep_vars]
    X = sm.add_constant(X)
    y = df[dep_var]
    modelo = sm.OLS(y, X).fit()
    return modelo


def evaluar_significancia(pval):
    if pval < 0.01:
        return '*** (1%)'
    elif pval < 0.05:
        return '** (5%)'
    elif pval < 0.10:
        return '* (10%)'
    else:
        return 'NO'


def generar_informe_modelo(modelo, dep_var, nombre_modelo):
    informe = []
    informe.append("=" * 70)
    informe.append(f"📊 INFORME DEL MODELO: {nombre_modelo}")
    informe.append("=" * 70)
    informe.append("\n1️⃣ ECUACIÓN ESTIMADA:")
    informe.append("-" * 70)
    eq = f"{dep_var} = "
    for i, coef in enumerate(modelo.params):
        var = modelo.params.index[i]
        signo = '+' if coef >= 0 and i > 0 else ''
        eq += f"{signo}{coef:.4f}*{var} "
    informe.append(eq)

    informe.append("\n2️⃣ COEFICIENTES Y SIGNIFICANCIA:")
    informe.append("-" * 70)
    informe.append(f"{'Variable':<20}{'Coeficiente':>12}  {'P-valor':>8}  {'Significativo?':>15}")
    informe.append("-" * 70)
    for var in modelo.params.index:
        coef = modelo.params[var]
        pval = modelo.pvalues[var]
        signif = evaluar_significancia(pval)
        informe.append(f"{var:<20}{coef:12.4f}  {pval:8.4f}  {signif:>15}")
    informe.append("\nNota: *** = 1%, ** = 5%, * = 10%")

    informe.append("\n3️⃣ BONDAD DE AJUSTE:")
    informe.append("-" * 70)
    informe.append(f"R² (R cuadrado)          : {modelo.rsquared:.4f}")
    informe.append(f"R² Ajustado              : {modelo.rsquared_adj:.4f}")
    informe.append(f"Número de observaciones  : {int(modelo.nobs)}")

    informe.append("\n4️⃣ VERIFICACIÓN DE SUPUESTOS:")
    informe.append("-" * 70)
    jb_stat, jb_p, _, _ = jarque_bera(modelo.resid)
    decision_norm = "✅ NO se rechaza H0" if jb_p > 0.05 else f"❌ Se rechaza H0 (p-valor = {jb_p:.4f} < 0.05)"
    informe.append(
        f"\n📌 A) NORMALIDAD DE RESIDUOS (Jarque-Bera)\n   Estadístico: {jb_stat:.4f}\n   P-valor: {jb_p:.4f}\n   H0: Los residuos se distribuyen normalmente\n   {decision_norm}")
    bp_test = het_breuschpagan(modelo.resid, modelo.model.exog)
    bp_stat, bp_p = bp_test[0], bp_test[1]
    decision_hom = "✅ NO se rechaza H0" if bp_p > 0.05 else f"❌ Se rechaza H0 (p-valor = {bp_p:.4f} < 0.05)"
    informe.append(
        f"\n📌 B) HOMOCEDASTICIDAD (Breusch-Pagan)\n   Estadístico: {bp_stat:.4f}\n   P-valor: {bp_p:.4f}\n   H0: Hay homocedasticidad (varianza constante)\n   {decision_hom}")
    bg_test = acorr_breusch_godfrey(modelo, nlags=1)
    bg_stat, bg_p = bg_test[0], bg_test[1]
    decision_autocor = "✅ NO se rechaza H0" if bg_p > 0.05 else f"❌ Se rechaza H0 (p-valor = {bg_p:.4f} < 0.05)"
    informe.append(
        f"\n📌 C) NO AUTOCORRELACIÓN (Breusch-Godfrey)\n   Estadístico: {bg_stat:.4f}\n   P-valor: {bg_p:.4f}\n   H0: No hay autocorrelación\n   {decision_autocor}")

    informe.append("\n5️⃣ GRÁFICOS DE DIAGNÓSTICO:")
    informe.append("-" * 70)
    return "\n".join(informe)


def graficar_serie_ols(dep_var, nombre_archivo):
    y = df[dep_var].values
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    tendencia = a + b * x
    plt.figure(figsize=(12, 6))
    plt.plot(df['Fecha_dt'], y, label=dep_var, color='blue')
    plt.plot(df['Fecha_dt'], tendencia, '--', color='orange', label='Tendencia OLS')
    plt.title(f"{dep_var} con línea de tendencia")
    plt.xlabel("Fecha")
    plt.ylabel(dep_var)
    plt.legend()
    plt.grid(True)
    ruta = os.path.join(OUTPUT_DIR, nombre_archivo)
    plt.savefig(ruta)
    plt.close()
    return ruta


def graficar_eventos_visual():
    plt.figure(figsize=(14, 7))
    plt.plot(df['Fecha_dt'], df['Ingresos_UCI_diarios'], label='Ingresos UCI', color='blue')
    plt.plot(df['Fecha_dt'], df['Fallecidos_diarios'], label='Fallecidos', color='red')

    x_ols = np.arange(len(df))
    y_uci = df['Ingresos_UCI_diarios'].values
    coef = np.polyfit(x_ols, y_uci, 1)
    tendencia_uci = coef[0] + coef[1] * x_ols
    plt.plot(df['Fecha_dt'], tendencia_uci, '--', color='orange', label='Tendencia UCI')

    # Eventos Vacuna
    vacunas = df[df['Hay_vacuna'] == 1]['Fecha_dt']
    for f in vacunas:
        if pd.notna(f):
            plt.axvspan(f, f + pd.Timedelta(days=1), color='green', alpha=0.3)
            plt.text(f, plt.ylim()[1] * 0.95, 'Vacuna', rotation=90, verticalalignment='top', color='green')

    # Eventos Confinamiento
    confin = df[df['Hay_Confinamiento'] == 1]['Fecha_dt']
    for f in confin:
        if pd.notna(f):
            plt.axvspan(f, f + pd.Timedelta(days=1), color='orange', alpha=0.3)
            plt.text(f, plt.ylim()[1] * 0.9, 'Confinamiento', rotation=90, verticalalignment='top', color='orange')

    plt.title("COVID-19: Ingresos UCI y Fallecidos diarios con eventos")
    plt.xlabel("Fecha")
    plt.ylabel("Casos diarios")
    plt.legend()
    plt.grid(True)

    ruta = os.path.join(OUTPUT_DIR, "grafico_eventos_visual.png")
    plt.savefig(ruta)
    plt.close()

    return ruta


# ================== AJUSTAR MODELOS ==================
modelo_UCI = ajustar_ols('Ingresos_UCI_diarios', ['Hay_vacuna', 'Hay_Confinamiento', 'Año'])
modelo_Fall = ajustar_ols('Fallecidos_diarios', ['Hay_vacuna', 'Hay_Confinamiento', 'Año'])

# ================== GENERAR INFORME ==================
informe_texto = []
informe_texto.append("=" * 70)
informe_texto.append("EJERCICIO: Análisis COVID-19 (UCI y Fallecidos)")
informe_texto.append("=" * 70)

informe_texto.append("\n🔹 MODELO 1: Ingresos UCI diarios")
informe_texto.append(generar_informe_modelo(modelo_UCI, 'Ingresos_UCI_diarios', 'Modelo UCI'))

informe_texto.append("\n🔹 MODELO 2: Fallecidos diarios")
informe_texto.append(generar_informe_modelo(modelo_Fall, 'Fallecidos_diarios', 'Modelo Fallecidos'))

informe_texto.append("\n" + "=" * 70)
informe_texto.append("✅ INFORME COMPLETO")
informe_texto.append("=" * 70)

# Guardar en archivo
with open(INFORME_PATH, "w", encoding='utf-8') as f:
    f.write("\n".join(informe_texto))

# Mostrar en consola
print("\n" + "=" * 70)
print("🔹 INFORME COMPLETO COVID-19 (UCI y Fallecidos) 🔹")
print("=" * 70 + "\n")
print("\n".join(informe_texto))
print("\n" + "=" * 70)

# ================== CREAR GRÁFICOS ==================
graf_UCI = graficar_serie_ols('Ingresos_UCI_diarios', 'grafico_UCI.png')
graf_Fall = graficar_serie_ols('Fallecidos_diarios', 'grafico_fallecidos.png')
graf_Eventos = graficar_eventos_visual()

print(f"\n✅ Gráficos guardados en: {OUTPUT_DIR}")
print(f" - {graf_UCI}")
print(f" - {graf_Fall}")
print(f" - {graf_Eventos}")

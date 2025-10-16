import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")
pdf_path = os.path.join(output_path, "Reporte_UCI_Fallecidos_UltraCompleto.pdf")

# --- CARGA DE DATOS ---
df = pd.read_csv(data_path)
df["Fecha_r"] = pd.to_datetime(df["Fecha_r"], errors="coerce")

features = ["Hay_vacuna", "Hay_Confinamiento"]
targets = ["Ingresos_UCI_diarios", "Fallecidos_diarios"]
X = df[features]
X = sm.add_constant(X)

# --- RESUMEN ESTADÍSTICO DEL DATASET ---
desc = df.describe()

# --- CREACIÓN DEL PDF ---
with PdfPages(pdf_path) as pdf:
    # Página 1: Resumen estadístico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("Resumen Estadístico del Dataset", fontsize=14, fontweight='bold')
    tabla = ax.table(cellText=desc.round(2).values,
                     rowLabels=desc.index,
                     colLabels=desc.columns,
                     cellLoc='center',
                     loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)
    pdf.savefig()
    plt.close()

    # Página 2: Gráfico histórico de Ingresos UCI
    plt.figure(figsize=(10, 5))
    plt.plot(df["Fecha_r"], df["Ingresos_UCI_diarios"], label="Ingresos UCI diarios")
    plt.xlabel("Fecha")
    plt.ylabel("Ingresos UCI diarios")
    plt.title("Histórico de Ingresos UCI diarios")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Página 3: Gráfico histórico de Fallecidos diarios
    plt.figure(figsize=(10, 5))
    plt.plot(df["Fecha_r"], df["Fallecidos_diarios"], label="Fallecidos diarios", color="red")
    plt.xlabel("Fecha")
    plt.ylabel("Fallecidos diarios")
    plt.title("Histórico de Fallecidos diarios")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# --- MODELOS OLS MULTISALIDA ---
model_dict = {}
coef_list = []

for target in targets:
    y = df[target]
    model = sm.OLS(y, X).fit()
    model_dict[target] = model
    for var in model.params.index:
        coef_list.append({
            "Variable_objetivo": target,
            "Predictor": var,
            "Coeficiente": model.params[var],
            "p_value": model.pvalues[var]
        })

df_coefs = pd.DataFrame(coef_list)

# --- AGREGAR MODELOS Y GRÁFICOS DE PREDICCIÓN AL PDF ---
with PdfPages(pdf_path, 'a') as pdf:  # 'a' para añadir
    # Página 4: Tabla coeficientes y p-values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("Coeficientes y p-values (Efectos Marginales)", fontsize=14, fontweight='bold')
    tabla = ax.table(cellText=df_coefs.round(4).values,
                     colLabels=df_coefs.columns,
                     cellLoc='center',
                     loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.5)
    pdf.savefig()
    plt.close()

    # Página 5 y 6: Predicciones vs reales
    for target in targets:
        model = model_dict[target]
        y_pred = model.predict(X)
        plt.figure(figsize=(8, 5))
        plt.scatter(df[target], y_pred, alpha=0.7)
        plt.plot([df[target].min(), df[target].max()],
                 [df[target].min(), df[target].max()], 'r--')
        plt.xlabel(f"{target} reales")
        plt.ylabel(f"{target} predichos")
        plt.title(f"Predicciones vs reales ({target})")
        plt.tight_layout()
        pdf.savefig()
        plt.close()


# --- FUNCIONES DE PREDICCIÓN NUEVA CON INTERVALOS DE CONFIANZA ---
def predecir_nuevos(hay_vacuna, hay_confinamiento, alpha=0.05):
    X_new = pd.DataFrame([[1, hay_vacuna, hay_confinamiento]], columns=["const"] + features)
    resultados = {}
    for target, model in model_dict.items():
        pred = model.get_prediction(X_new)
        pred_mean = pred.predicted_mean[0]
        ci_lower, ci_upper = pred.conf_int(alpha=alpha)[0]
        resultados[target] = {"Predicción": pred_mean, "CI_lower": ci_lower, "CI_upper": ci_upper}
    print(f"Predicción → Vacuna: {hay_vacuna}, Confinamiento: {hay_confinamiento} → {resultados}")
    return resultados


# --- EJEMPLOS DE PREDICCIÓN ---
predecir_nuevos(0, 0)
predecir_nuevos(1, 0)
predecir_nuevos(1, 1)

# --- APLICACIÓN A CSV REAL CON INTERVALOS DE CONFIANZA ---
csv_real = os.path.join(output_path, "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
if os.path.exists(csv_real):
    df_real = pd.read_csv(csv_real)
    X_real = df_real[features]
    X_real = sm.add_constant(X_real)
    for target, model in model_dict.items():
        pred_real = model.get_prediction(X_real)
        df_real[f"{target}_pred"] = pred_real.predicted_mean
        df_real[f"{target}_CI_lower"] = pred_real.conf_int()[:, 0]
        df_real[f"{target}_CI_upper"] = pred_real.conf_int()[:, 1]
    df_real.to_csv(os.path.join(output_path, "uci_fallecidos_predicciones_ultracompleto.csv"), index=False)
    print("✅ Predicciones guardadas en 'uci_fallecidos_predicciones_ultracompleto.csv'")

print(f"\n✅ Reporte PDF ultra completo generado en: {pdf_path}")
print("=== FIN DEL PROCESO ===")

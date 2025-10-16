import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")

# --- CARGA DE DATOS ---
print("=== CARGANDO ARCHIVO REAL ===")
df = pd.read_csv(data_path)
df["Fecha_r"] = pd.to_datetime(df["Fecha_r"], errors="coerce")
print(df.head(), "\n")

# --- VARIABLES ---
features = ["Hay_vacuna", "Hay_Confinamiento"]
targets = ["Ingresos_UCI_diarios", "Fallecidos_diarios"]

X = df[features]
X = sm.add_constant(X)  # Añadimos intercepto

# --- MODELOS OLS MULTISALIDA ---
model_dict = {}
coef_list = []

for target in targets:
    y = df[target]
    model = sm.OLS(y, X).fit()
    model_dict[target] = model
    # Guardar coeficientes y p-values
    for var in model.params.index:
        coef_list.append({
            "Variable_objetivo": target,
            "Predictor": var,
            "Coeficiente": model.params[var],
            "p_value": model.pvalues[var]
        })

    # Gráfico real vs predicción
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel(f"{target} reales")
    plt.ylabel(f"{target} predichos")
    plt.title(f"Predicciones vs reales ({target})")
    plt.tight_layout()
    filename = f"{target}_pred_vs_real.png"
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.show()
    print(f"✅ Gráfico guardado como '{filename}'\n")

# --- RESUMEN CONSOLIDADO DE EFECTOS MARGINALES ---
df_coefs = pd.DataFrame(coef_list)
print("=== RESUMEN CONSOLIDADO DE EFECTOS MARGINALES ===")
print(df_coefs)


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
print("=== PREDICCIONES NUEVAS ===")
predecir_nuevos(0, 0)
predecir_nuevos(1, 0)
predecir_nuevos(1, 1)

# --- APLICACIÓN A CSV REAL FUTURO ---
csv_real = os.path.join(output_path, "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
if os.path.exists(csv_real):
    print("\n=== APLICANDO MODELO AL CSV REAL ===")
    df_real = pd.read_csv(csv_real)
    X_real = df_real[features]
    X_real = sm.add_constant(X_real)
    for target, model in model_dict.items():
        pred_real = model.get_prediction(X_real)
        df_real[f"{target}_pred"] = pred_real.predicted_mean
        df_real[f"{target}_CI_lower"] = pred_real.conf_int()[:, 0]
        df_real[f"{target}_CI_upper"] = pred_real.conf_int()[:, 1]
    df_real.to_csv(os.path.join(output_path, "uci_fallecidos_predicciones_final.csv"), index=False)
    print("✅ Predicciones guardadas en 'uci_fallecidos_predicciones_final.csv'")

print("\n=== FIN DEL PROCESO ===")

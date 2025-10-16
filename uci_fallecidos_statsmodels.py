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
target = "Ingresos_UCI_diarios"

X = df[features]
X = sm.add_constant(X)  # añade el intercepto
y = df[target]

# --- AJUSTE DEL MODELO OLS ---
model = sm.OLS(y, X).fit()

# --- RESUMEN DEL MODELO ---
print("=== RESUMEN DEL MODELO ===")
print(model.summary())

# --- EFECTOS MARGINALES ---
# En OLS, los efectos marginales = coeficientes de las variables
marginal_effects = model.params[1:]  # excluye const
p_values = model.pvalues[1:]
print("\n=== EFECTOS MARGINALES ===")
for var in features:
    print(f"{var}: coef={marginal_effects[var]:.3f}, p-value={p_values[var]:.4f}")

# --- PLOT: REAL VS PREDICCIÓN ---
y_pred = model.predict(X)
plt.figure(figsize=(8,5))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Ingresos UCI reales")
plt.ylabel("Ingresos UCI predichos")
plt.title("Predicciones vs Valores reales (Ingresos UCI)")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "uci_pred_vs_real_statsmodels.png"), dpi=300)
plt.show()
print("\n✅ Gráfico guardado como 'uci_pred_vs_real_statsmodels.png'\n")

# --- FUNCIONES DE PREDICCIÓN NUEVA ---
def predecir_uci(hay_vacuna, hay_confinamiento):
    X_new = pd.DataFrame([[1, hay_vacuna, hay_confinamiento]], columns=["const"] + features)
    pred = model.predict(X_new)[0]
    print(f"Predicción Ingresos UCI → Vacuna: {hay_vacuna}, Confinamiento: {hay_confinamiento} → {pred:.1f}")
    return pred

# --- EJEMPLOS DE PREDICCIÓN ---
print("=== PREDICCIONES NUEVAS ===")
predecir_uci(0,0)
predecir_uci(1,0)
predecir_uci(1,1)

# --- APLICACIÓN A CSV REAL FUTURO ---
csv_real = os.path.join(output_path, "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
if os.path.exists(csv_real):
    print("\n=== APLICANDO MODELO AL CSV REAL ===")
    df_real = pd.read_csv(csv_real)
    X_real = df_real[features]
    X_real = sm.add_constant(X_real)
    df_real["Ingresos_UCI_pred"] = model.predict(X_real)
    df_real.to_csv(os.path.join(output_path, "uci_predicciones_statsmodels.csv"), index=False)
    print("✅ Predicciones guardadas en 'uci_predicciones_statsmodels.csv'")

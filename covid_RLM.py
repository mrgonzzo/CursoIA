import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")

# --- CARGA DE DATOS ---
print("=== CARGANDO ARCHIVO REAL ===")
df = pd.read_csv(data_path)
print(df.head(), "\n")

# --- LIMPIEZA Y PREPARACIÓN ---
df["Fecha_r"] = pd.to_datetime(df["Fecha_r"], errors="coerce")

# Variables predictoras
features = ["Hay_vacuna", "Hay_Confinamiento"]
target_uci = "Ingresos_UCI_diarios"
target_fallecidos = "Fallecidos_diarios"

X = df[features]
y_uci = df[target_uci]
y_fallecidos = df[target_fallecidos]

# --- ESCALADO ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- DIVISIÓN DE DATOS ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_uci, test_size=0.3, random_state=42)

# --- MODELO REGRESIÓN LINEAL (UCI) ---
model_uci = LinearRegression()
model_uci.fit(X_train, y_train)

# --- GUARDAR MODELO Y SCALER ---
joblib.dump(model_uci, os.path.join(output_path, "modelo_uci.pkl"))
joblib.dump(scaler, os.path.join(output_path, "scaler_uci.pkl"))
print("✅ Modelo y scaler UCI guardados como 'modelo_uci.pkl' y 'scaler_uci.pkl'\n")

# --- EVALUACIÓN ---
y_pred = model_uci.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== EVALUACIÓN MODELO INGRESOS UCI ===")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.3f}")

# --- PLOT: REAL VS PREDICHO ---
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Ingresos UCI reales")
plt.ylabel("Ingresos UCI predichos")
plt.title("Predicciones vs Valores reales (Ingresos UCI)")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "uci_pred_vs_real.png"), dpi=300)
plt.show()
print("✅ Gráfico guardado como 'uci_pred_vs_real.png'\n")

# --- FUNCIONES DE PREDICCIÓN NUEVA ---
def predecir_uci(hay_vacuna, hay_confinamiento):
    X_new = pd.DataFrame([[hay_vacuna, hay_confinamiento]], columns=features)
    X_new_scaled = scaler.transform(X_new)
    pred = model_uci.predict(X_new_scaled)[0]
    print(f"Predicción Ingresos UCI → Vacuna: {hay_vacuna}, Confinamiento: {hay_confinamiento} → {pred:.1f}")
    return pred

# --- EJEMPLOS DE PREDICCIÓN ---
print("=== PREDICCIONES NUEVAS ===")
predecir_uci(0, 0)
predecir_uci(1, 0)
predecir_uci(1, 1)
print("\n=== FIN DEL PROCESO ===")

# --- APLICACIÓN A CSV REAL FUTURO ---
csv_real = os.path.join(output_path, "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
if os.path.exists(csv_real):
    print("\n=== APLICANDO MODELO AL CSV REAL ===")
    df_real = pd.read_csv(csv_real)
    X_real = df_real[features]
    X_real_scaled = scaler.transform(X_real)
    df_real["Ingresos_UCI_pred"] = model_uci.predict(X_real_scaled)
    df_real.to_csv(os.path.join(output_path, "uci_predicciones.csv"), index=False)
    print("✅ Predicciones guardadas en 'uci_predicciones.csv'")

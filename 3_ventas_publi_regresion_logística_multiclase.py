import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "ventasypubl.csv")
output_path = os.path.join(base_dir, "DATOS")

# --- CARGA DE DATOS ---
print("=== CARGANDO ARCHIVO REAL ===")
df = pd.read_csv(data_path)
print(df.head(), "\n")

# --- LIMPIEZA Y PREPARACIÓN ---
df["dateid01"] = pd.to_datetime(df["dateid01"], errors="coerce")
df["dateid"] = pd.to_datetime(df["dateid"], errors="coerce")

# Crear variable de clasificación basada en ventas y publicaciones
def clasificar_periodo(row):
    if row["vtas"] >= 1000 and row["pub"] >= 550:
        return "Alto"
    elif row["vtas"] >= 950 and row["pub"] >= 500:
        return "Medio"
    else:
        return "Bajo"

df["nivel_ventas"] = df.apply(clasificar_periodo, axis=1)

# --- VARIABLES ---
X = df[["pub", "vtas"]]
y = df["nivel_ventas"]

# --- ESCALADO ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- DIVISIÓN DE DATOS ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# --- MODELO DE REGRESIÓN LOGÍSTICA ---
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000)
model.fit(X_train, y_train)

# --- GUARDAR MODELO Y SCALER ---
joblib.dump(model, os.path.join(output_path, "modelo_ventas.pkl"))
joblib.dump(scaler, os.path.join(output_path, "scaler_ventas.pkl"))
print("✅ Modelo y scaler guardados como 'modelo_ventas.pkl' y 'scaler_ventas.pkl'\n")

# --- EVALUACIÓN ---
y_pred = model.predict(X_test)

print("=== EVALUACIÓN DEL MODELO ===")
print("Precisión general: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, zero_division=0))

# --- MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(7,5))
disp.plot(cmap="Blues", ax=plt.gca())
plt.title("Matriz de Confusión - Clasificación de Ventas")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "matriz_confusion_ventas.png"), dpi=300)
plt.show()
print("✅ Matriz de confusión guardada como 'matriz_confusion_ventas.png'\n")

# --- FUNCIONES DE PREDICCIÓN NUEVA ---
def predecir_nuevos(pub, vtas):
    """Predice el nivel de ventas usando el modelo guardado"""
    X_new = pd.DataFrame([[pub, vtas]], columns=["pub", "vtas"])
    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)[0]
    prob = np.max(model.predict_proba(X_new_scaled))
    print(f"Publicaciones: {pub}, Ventas: {vtas} → Nivel estimado: {pred} (confianza: {prob*100:.1f}%)")
    return pred

# --- EJEMPLOS DE PREDICCIÓN ---
print("=== PREDICCIONES NUEVAS ===")
predecir_nuevos(400, 900)
predecir_nuevos(600, 1100)
predecir_nuevos(550, 970)
print("\n=== FIN DEL PROCESO ===")

# --- APLICACIÓN A CSV REAL FUTURO ---
csv_real = os.path.join(output_path, "ventasypubl.csv")  # mismo CSV u otro
if os.path.exists(csv_real):
    print("\n=== APLICANDO MODELO AL CSV REAL ===")
    df_real = pd.read_csv(csv_real)
    X_real = df_real[["pub", "vtas"]]
    X_real_scaled = scaler.transform(X_real)
    df_real["nivel_ventas_pred"] = model.predict(X_real_scaled)
    df_real.to_csv(os.path.join(output_path, "ventasypubl_predicciones.csv"), index=False)
    print("✅ Predicciones guardadas en 'ventasypubl_predicciones.csv'")

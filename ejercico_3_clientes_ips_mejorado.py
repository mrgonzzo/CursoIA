# -*- coding: utf-8 -*-
"""
Clasificación de clientes con Regresión Logística Multiclase + Detección de Casos Fuera de Rango
Autor: Vincent & ChatGPT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ============================
# 1. Generar datos sintéticos
# ============================
N = 1000
np.random.seed(42)

velocidad_mbps = np.random.uniform(5, 500, N)
datos_gb_mes = np.random.uniform(10, 1000, N)
dispositivos = np.random.randint(1, 16, N)


def clasificar_cliente(v, d, disp):
    if v >= 300 and d >= 500 and disp >= 10:
        return "Empresarial"
    elif v >= 100 and d >= 300 and disp < 10:
        return "Premium"
    elif v >= 50 and d >= 100:
        return "Estándar"
    else:
        return "Básico"


perfil = [clasificar_cliente(v, d, disp) for v, d, disp in zip(velocidad_mbps, datos_gb_mes, dispositivos)]

df = pd.DataFrame({
    "velocidad_mbps": velocidad_mbps,
    "datos_gb_mes": datos_gb_mes,
    "dispositivos": dispositivos,
    "perfil": perfil
})

print("=== MUESTRA DE DATOS GENERADOS ===")
print(df.head(), "\n")

os.makedirs("resultados_modelo", exist_ok=True)

# ============================
# 2. Preparar datos
# ============================
X = df[["velocidad_mbps", "datos_gb_mes", "dispositivos"]]
y = df["perfil"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42,
                                                    stratify=y_encoded)

# ============================
# 3. Entrenar modelo
# ============================
modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)

# ============================
# 4. Evaluar modelo
# ============================
y_pred = modelo.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("=== EVALUACIÓN DEL MODELO ===")
print(f"Precisión general: {acc * 100:.2f} %\n")
print(pd.DataFrame(report).transpose())

# ============================
# 5. Validación cruzada
# ============================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X_scaled, y_encoded, cv=kfold, scoring='accuracy')

print("\n=== VALIDACIÓN CRUZADA (K-FOLD) ===")
print(f"Resultados por fold: {np.round(scores * 100, 2)} %")
print(f"Precisión promedio: {np.round(scores.mean() * 100, 2)} %")
print(f"Desviación estándar: {np.round(scores.std() * 100, 2)} %")

# ============================
# 6. Guardar resultados
# ============================
plt.figure(figsize=(7, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Regresión Logística Multiclase")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("resultados_modelo/matriz_confusion.png", dpi=300)
plt.close()

joblib.dump(modelo, "resultados_modelo/modelo.pkl")
joblib.dump(scaler, "resultados_modelo/scaler.pkl")
joblib.dump(le, "resultados_modelo/label_encoder.pkl")

print("\n✅ Archivos guardados en la carpeta 'resultados_modelo/'")


# ============================
# 7. Predicción robusta
# ============================
def predecir_perfil(vel, datos, disp):
    # Verificación de rango válido
    if not (5 <= vel <= 500) or not (10 <= datos <= 1000) or not (1 <= disp <= 15):
        print(f"\n⚠️ Valores fuera de rango: Vel={vel}, Datos={datos}, Disp={disp}")
        print("Perfil estimado: Indeterminado")
        return

    # Verificación de compatibilidad lógica
    if (vel < 5) or (datos < 10):
        print(f"\n⚠️ Datos incoherentes detectados ({vel} Mbps / {datos} GB)")
        print("Perfil estimado: Indeterminado")
        return

    X_nuevo = pd.DataFrame([[vel, datos, disp]], columns=["velocidad_mbps", "datos_gb_mes", "dispositivos"])
    X_nuevo_scaled = scaler.transform(X_nuevo)
    probas = modelo.predict_proba(X_nuevo_scaled)[0]
    max_prob = np.max(probas)

    # Umbral de confianza
    if max_prob < 0.6:
        print(f"\n⚠️ Baja confianza en la predicción ({max_prob * 100:.1f}%)")
        print(f"Perfil estimado: Indeterminado")
        return

    pred = modelo.predict(X_nuevo_scaled)
    perfil_pred = le.inverse_transform(pred)[0]
    print(f"\nPredicción -> Velocidad: {vel} Mbps, Datos: {datos} GB, Dispositivos: {disp}")
    print(f"Perfil estimado: {perfil_pred} (confianza: {max_prob * 100:.1f}%)")


# ============================
# 8. Ejemplos de predicción
# ============================
predecir_perfil(400, 800, 12)  # Empresarial
predecir_perfil(150, 400, 5)  # Estándar
predecir_perfil(60, 200, 4)  # Básico
predecir_perfil(25, 50, 1)  # Básico
predecir_perfil(600, 2000, 20)  # Fuera de rango
predecir_perfil(120, 250, 25)  # Dispositivos fuera de rango
predecir_perfil(80, 90, 8)  # Baja confianza

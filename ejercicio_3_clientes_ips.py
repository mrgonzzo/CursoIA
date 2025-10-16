# -*- coding: utf-8 -*-
"""
https://chatgpt.com/share/68f11e00-94e0-8007-a4e6-81834d89ac53

Programa: Clasificación de perfiles de clientes con Regresión Logística Multiclase
Autor: Vincent & ChatGPT
Descripción:
Genera datos sintéticos de clientes (velocidad, consumo de datos, dispositivos)
y entrena un modelo de regresión logística para clasificar los perfiles:
Básico, Estándar, Premium y Empresarial.
"""

# ============================
# Importar librerías necesarias
# ============================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import random

# ============================
# 1. Generar datos sintéticos
# ============================

# Configuración
N = 1000  # cantidad de registros a generar

np.random.seed(42)

# Variables
velocidad_mbps = np.random.uniform(5, 500, N)
datos_gb_mes = np.random.uniform(10, 1000, N)
dispositivos = np.random.randint(1, 16, N)

# Función de clasificación según tus reglas
def clasificar_cliente(v, d, disp):
    if v >= 300 and d >= 500 and disp >= 10:
        return "Empresarial"
    elif v >= 100 and d >= 300 and disp < 10:
        return "Premium"
    elif v >= 50 and d >= 100:
        return "Estándar"
    else:
        return "Básico"

# Aplicar la clasificación
perfil = [clasificar_cliente(v, d, disp) for v, d, disp in zip(velocidad_mbps, datos_gb_mes, dispositivos)]

# Crear DataFrame
df = pd.DataFrame({
    "velocidad_mbps": velocidad_mbps,
    "datos_gb_mes": datos_gb_mes,
    "dispositivos": dispositivos,
    "perfil": perfil
})

# Mostrar muestra inicial
print("=== MUESTRA DE DATOS GENERADOS ===")
print(df.head(), "\n")

# Guardar opcionalmente en CSV
df.to_csv("clientes.csv", index=False, encoding="utf-8")

# ============================
# 2. Preparar datos para el modelo
# ============================

X = df[["velocidad_mbps", "datos_gb_mes", "dispositivos"]]
y = df["perfil"]

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ============================
# 3. Entrenar el modelo
# ============================

modelo = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
modelo.fit(X_train, y_train)

# ============================
# 4. Evaluar el modelo
# ============================

y_pred = modelo.predict(X_test)

print("=== EVALUACIÓN DEL MODELO ===")
print("Precisión general:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================
# 5. Probar nuevas predicciones
# ============================

def predecir_perfil(vel, datos, disp):
    X_nuevo = np.array([[vel, datos, disp]])
    pred = modelo.predict(X_nuevo)
    perfil_pred = le.inverse_transform(pred)[0]
    print(f"\nPredicción -> Velocidad: {vel} Mbps, Datos: {datos} GB, Dispositivos: {disp}")
    print(f"Perfil estimado: {perfil_pred}")

# Ejemplo de predicciones
predecir_perfil(400, 800, 12)   # Empresarial
predecir_perfil(150, 400, 5)    # Premium
predecir_perfil(60, 200, 4)     # Estándar
predecir_perfil(25, 50, 1)      # Básico

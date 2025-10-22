# python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import StringIO
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE NOMBRES Y SALIDA ---
NAME_POLICE = 'Police_Incidents_GeoTime'
NAME_SOCIO = 'SocioEcon_Censo'
NAME_URBAN = 'Urbanisticos_SIG'
NAME_CONTEXT = 'Contextual_Eventos'
OUTPUT_DIR = 'reporte_criminalidad_urbana'
PDF_FILENAME = os.path.join(OUTPUT_DIR, 'Reporte_Predictivo_Inicial.pdf')
PLOT_FILENAME = os.path.join(OUTPUT_DIR, 'Incidencia_UsoSuelo.png')

# Crear el directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 0. DATASETS HETEROGÉNEOS Y SIMULACIÓN DE FUSIÓN ---

data_police_str = """Celda_ID,Hora_Dia,Dia_Semana,Tasa_Incidencia
C001,Tarde,Miercoles,5.2
C002,Mañana,Lunes,1.1
C003,Noche,Viernes,8.9
C004,Tarde,Jueves,3.5
C005,Noche,Viernes,6.1
C006,Mañana,Lunes,2.0
C007,Noche,Sabado,7.5
C008,Tarde,Miercoles,4.0
C009,Noche,Sabado,10.0
C010,Mañana,Martes,2.5
"""
df_police = pd.read_csv(StringIO(data_police_str))

data_socio_str = """Celda_ID,Renta_Media_Hogar,Densidad_Poblacion
C001,25000,5200
C002,45000,3100
C003,18000,1500
C004,38000,4500
C005,22000,6000
C006,50000,2800
C007,19500,1800
C008,35000,4100
C009,15000,6500
C010,42000,3300
"""
df_socio = pd.read_csv(StringIO(data_socio_str))

data_urban_str = """Celda_ID,Iluminacion_Media,Tipo_Uso_Suelo
C001,0.85,Comercial
C002,0.98,Residencial
C003,0.55,Industrial
C004,0.72,Residencial
C005,0.68,Comercial
C006,0.95,Residencial
C007,0.60,Industrial
C008,0.78,Residencial
C009,0.45,Comercial
C010,0.90,Residencial
"""
df_urban = pd.read_csv(StringIO(data_urban_str))

data_context_str = """Celda_ID,Evento_Programado
C001,No
C002,No
C003,Si
C004,No
C005,Si
C006,No
C007,Si
C008,No
C009,No
C010,No
"""
df_context = pd.read_csv(StringIO(data_context_str))

# Fusión
df = df_police.merge(df_socio, on='Celda_ID', how='left')
df = df.merge(df_urban, on='Celda_ID', how='left')
df = df.merge(df_context, on='Celda_ID', how='left')

print("--- 0. DataSets Heterogéneos Utilizados y Fusión Simulada (ETL) ---")
print(f"1. **{NAME_POLICE}** (Objetivo): {list(df_police.columns)}")
print(f"2. **{NAME_SOCIO}**: {list(df_socio.columns)}")
print(f"3. **{NAME_URBAN}**: {list(df_urban.columns)}")
print(f"4. **{NAME_CONTEXT}**: {list(df_context.columns)}")
print("\nResultado: DataFrame único por Cuadrícula Espacial/Temporal ('Celda_ID' + Temporales).")
print("=" * 80 + "\n")

# --- 1. DATOS UNIFICADOS Y PREPARACIÓN PARA $sklearn$ ---

TARGET = 'Tasa_Incidencia'
y = df[TARGET]
X = df.drop(columns=[TARGET, 'Celda_ID'])

# Bloque de preprocesamiento con ColumnTransformer
numerical_features = ['Iluminacion_Media', 'Renta_Media_Hogar', 'Densidad_Poblacion']
categorical_features = ['Tipo_Uso_Suelo', 'Evento_Programado', 'Hora_Dia', 'Dia_Semana']

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

X_processed = preprocessor.fit_transform(X)

feature_names_out = (
        numerical_features +
        list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
)

X_final = pd.DataFrame(X_processed, columns=feature_names_out)

print("--- 1. DataFrame Final Post-Preprocesamiento ($sklearn$) ---")
print("Dimensiones (Filas x Columnas):", X_final.shape)
print("\n" + "=" * 80 + "\n")

# --- 2. GENERACIÓN DE GRÁFICOS (Matplotlib) ---

# Análisis de Tasa de Incidencia Media por Tipo de Uso de Suelo
avg_incidencia_uso = df.groupby('Tipo_Uso_Suelo')['Tasa_Incidencia'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
avg_incidencia_uso.plot(kind='bar', color=['#CC0000', '#FF9933', '#0099CC'])
plt.title('Tasa de Incidencia Media por Uso de Suelo (EDA)')
plt.ylabel('Tasa de Incidencia (simulada)')
plt.xlabel('Tipo de Uso de Suelo (Urbanístico)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig(PLOT_FILENAME)
plt.close()

print(f"--- 2. Gráfico generado: '{PLOT_FILENAME}' (Análisis Urbanístico) ---")
print("\n" + "=" * 80 + "\n")

# --- 3. RECOLECCIÓN DE EXPLICACIONES Y DEDUCCIONES ---

# CORRECCIÓN: Usamos to_string() para la muestra del DataFrame y eliminamos \_ para evitar SyntaxWarning
df_muestra_str = X_final.head().to_string()

explicaciones = [
    "## 1. Fusión de DataSets Heterogéneos",
    f"El sistema se basa en la fusión de 4 fuentes clave:",
    f"- 🚨 **{NAME_POLICE}** (Datos delictivos)",
    f"- 💰 **{NAME_SOCIO}** (Renta, Densidad)",
    f"- 🏗️ **{NAME_URBAN}** (Iluminación, Uso del Suelo)",
    f"- 🗓️ **{NAME_CONTEXT}** (Eventos, Temporalidad)",
    f"El dataset resultante tiene {X_final.shape[0]} registros (cuadrículas x tiempo) y {X_final.shape[1]} características.",

    "\n## 2. Preprocesamiento $sklearn$ y Deducciones Clave",
    "**DEDUCCIÓN 1 (Escalamiento):** La **StandardScaler** aplicada a la 'Renta_Media_Hogar' y 'Densidad_Poblacion' asegura que las variables numéricas contribuyan sin sesgo por magnitud, fundamental para la **Regresión Logística**.",

    f"**DEDUCCIÓN 2 (Codificación Categórica):** El **OneHotEncoder** ($sklearn$) convirtió las variables discretas (como 'Hora_Dia', 'Tipo_Uso_Suelo') en un espacio de características expandido de {X_final.shape[1]} columnas, permitiendo que los modelos de **Random Forest** capturen las complejas interacciones.",

    "\n## 3. Análisis Gráfico (Anexo Imagen)",
    f"El gráfico '{os.path.basename(PLOT_FILENAME)}' demuestra que las zonas **{avg_incidencia_uso.index[0]}** tienen la mayor tasa de incidencia. Esta es una deducción crucial de **Ingeniería de Características**.",

    "\n## 4. Datos Procesados (Muestra)",
    "Se incluye una muestra del DataFrame final listo para el ML (escalado y codificado):",
    df_muestra_str  # Insertamos la tabla formateada con to_string()
]

# --- 4. SIMULACIÓN DE CREACIÓN DE PDF ---

print(f"--- 4. Generación Simulada del Reporte PDF: '{PDF_FILENAME}' ---")

# Guardamos el contenido del reporte en un archivo de texto para simular la estructura del PDF.
report_content = "\n".join(explicaciones)

try:
    # Usamos .txt para la simulación
    with open(PDF_FILENAME.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ARQUITECTURA Y PROCESAMIENTO DE DATOS\n")
        f.write("-" * 50 + "\n")
        f.write(report_content)
        f.write("\n\n(En el PDF real, se insertaría el gráfico Incidencia_UsoSuelo.png aquí.)")
    print(f"✅ Éxito: Contenido del reporte guardado en '{PDF_FILENAME.replace('.pdf', '.txt')}'")
    print(f"✅ Éxito: Gráfico de análisis guardado en '{PLOT_FILENAME}'")

except Exception as e:
    print(f"❌ Error al intentar guardar el archivo (simulado): {e}")

print("\n" + "=" * 80)
print("El script ahora funciona sin la librería 'tabulate' y sin SyntaxWarning.")
print(
    "Para generar el PDF real, recuerde instalar 'tabulate' (opción 1) o usar librerías de generación de PDF como ReportLab o Fpdf (opción 2).")
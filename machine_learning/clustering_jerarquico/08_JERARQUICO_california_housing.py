# python
import pandas as pd
import numpy as np
import seaborn as sns
from io import StringIO # AÑADIDO: Importación explícita de StringIO para manejar strings como archivos
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing, load_diabetes
import matplotlib.pyplot as plt
import os

# --- 0. CARGA, RENOMBRADO Y ALINEACIÓN DE DATASETS HETEROGÉNEOS ---

# 0.1. DATASET 1: California Housing
cali_housing = fetch_california_housing(as_frame=True)
df_cali = cali_housing.frame[['MedInc', 'AveRooms', 'HouseAge']].head(100).copy()
df_cali.columns = ['Renta_Media', 'Promedio_Habitaciones', 'Antiguedad_Vivienda']
df_cali['ID'] = range(len(df_cali))

# 0.2. DATASET 2: USArrests (Simulando Carga)
# CORRECCIÓN: Se asegura que el string multilinea sea válido.
data_arrests_str = """
Murder,Assault,Rape,UrbanPop
13.2,236,21.2,58
10.0,263,44.5,48
8.1,294,31.0,80
8.8,190,19.5,50
9.0,276,40.6,91
"""
df_arrests_base = pd.read_csv(StringIO(data_arrests_str))
df_arrests = pd.concat([df_arrests_base] * 20, ignore_index=True).head(100).copy()
df_arrests['Tasa_Delincuencia'] = df_arrests['Murder'] + df_arrests['Rape']
df_arrests['ID'] = range(len(df_arrests))
df_arrests = df_arrests[['ID', 'Tasa_Delincuencia', 'UrbanPop']]

# 0.3. DATASET 3: Diabetes
diabetes = load_diabetes(as_frame=True)
df_diabetes = diabetes.frame[['bmi', 'bp']].head(100).copy()
df_diabetes.columns = ['IMC', 'Presion_Sanguinea']
df_diabetes['ID'] = range(len(df_diabetes))

# 0.4. DATASET 4: Tips
df_tips_base = sns.load_dataset('tips').head(100).copy()
df_tips_base = df_tips_base[['day', 'time', 'size']]
df_tips_base.columns = ['Dia_Semana', 'Momento_Dia', 'Tamano_Grupo']
df_tips_base['ID'] = range(len(df_tips_base))

# --- 1. FUSIÓN DE DATASETS (Usando 'ID' como clave de unión) ---

df_fused = df_cali.merge(df_arrests, on='ID', how='left')
df_fused = df_fused.merge(df_diabetes, on='ID', how='left')
df_fused = df_fused.merge(df_tips_base, on='ID', how='left')

# Definir el objetivo (simulado) y las características
TARGET = 'Tasa_Delincuencia'
y = df_fused[TARGET]
X = df_fused.drop(columns=[TARGET, 'ID'])

print("--- 1. DataSets Fusionados y Variables Identificadas ---")
print(f"Objetivo (Y): {TARGET}")
print(f"Características (X) Totales: {list(X.columns)}")
print("-" * 70)
print(X.head().to_string())
print("\n" + "="*80 + "\n")

# --- 2. CONFIGURACIÓN DEL PIPELINE DE PREPROCESAMIENTO ($sklearn$) ---

# Identificación de variables por tipo
numerical_features = ['Renta_Media', 'Promedio_Habitaciones', 'Antiguedad_Vivienda',
                      'UrbanPop', 'IMC', 'Presion_Sanguinea', 'Tamano_Grupo']
categorical_features = ['Dia_Semana', 'Momento_Dia']

# Transformadores
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Aplicar el preprocesamiento
X_processed = preprocessor.fit_transform(X)

# Recuperar los nombres de las columnas para el DataFrame final
feature_names_out = (
    numerical_features +
    list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
)
X_final = pd.DataFrame(X_processed, columns=feature_names_out)


print("--- 2. DataFrame Final Post-Preprocesamiento ($sklearn$) ---")
print("Dimensiones Finales (Filas x Columnas):", X_final.shape)
print("\n" + "="*80 + "\n")

# --- 3. DEDUCCIONES CLAVE DE LA FUSIÓN HETEROGÉNEA ---

print("--- 3. Deducciones de la Integración de DataSets Canónicos ---")

print("DEDUCCIÓN 1 (Escalamiento y Homogeneización):")
print(f"La **StandardScaler** ($sklearn$) fue vital para homogeneizar las escalas masivamente diferentes: 'Renta_Media' (miles, de California Housing) y 'IMC' (cercano a cero, de Diabetes). Esta acción es crucial para que el modelo no priorice la magnitud del dato sobre su importancia predictiva real.")

print("\nDEDUCCIÓN 2 (Ingeniería Contextual y Codificación):")
print(f"Variables categóricas como 'Dia_Semana' y 'Momento_Dia' (de Tips) fueron transformadas con **OneHotEncoder** ($sklearn$). Estas *features* temporales y contextuales introducen la dinámica social en el modelo, y su correcta codificación es necesaria, resultando en un *feature space* de **{X_final.shape[1]} columnas**.")

print("\nDEDUCCIÓN 3 (Relevancia y Fusión):")
print(f"La fusión ha unido datos de criminalidad ('Tasa_Delincuencia' de USArrests) con datos de vivienda ('Antiguedad_Vivienda' de California Housing) y datos biométricos ('IMC' de Diabetes). El **ColumnTransformer** crea un esquema de datos unificado y listo para el ML, fundamental para la arquitectura de datos heterogéneos.")

print("\n--- Muestra del DataFrame Final (Listo para ML) ---")
print(X_final.head().to_string())
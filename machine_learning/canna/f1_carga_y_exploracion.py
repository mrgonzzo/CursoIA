import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

print("--- f1_carga_y_exploracion.py: Carga, Estandarización y Guardado ---")

# 1. Creación de un Dataset de Ejemplo de Cepas de Cannabis (Simulado)
data = {
    'Strain': ['Amnesia Haze', 'Blue Dream', 'Girl Scout Cookies', 'White Rhino', 'Granddaddy Purple', 'Northern Lights', 'Sour Diesel'],
    'Type': ['Sativa', 'Hybrid', 'Hybrid', 'Indica', 'Indica', 'Indica', 'Sativa'],
    'THC_Level': [22.5, 18.0, 24.0, 16.5, 17.5, 20.0, 25.5],
    'Myrcene': [0.2, 0.4, 0.1, 1.2, 0.9, 1.1, 0.3],
    'Limonene': [0.8, 0.3, 0.4, 0.05, 0.1, 0.08, 0.75],
    'Caryophyllene': [0.15, 0.3, 0.25, 0.7, 0.6, 0.55, 0.2],
    'Relaxed': [3, 4, 5, 5, 5, 5, 2],
    'Happy': [5, 4, 3, 2, 3, 3, 5],
    'Uplifted': [5, 3, 3, 1, 1, 1, 5],
    'Sleepy': [1, 2, 3, 5, 4, 4, 1],
    'Image_URL': ['url_ahaze.jpg', 'url_bdream.jpg', 'url_gsc.jpg', 'url_wrr.jpg', 'url_gdp.jpg', 'url_nl.jpg', 'url_sd.jpg']
}
df_original = pd.DataFrame(data).set_index('Strain')

# Guardar el dataset original
df_original.to_csv('cannabis_data_original.csv')
print("✅ Datos originales guardados en 'cannabis_data_original.csv'.")

# 2. Estandarización de Datos Numéricos
features_to_scale = df_original.select_dtypes(include=np.number).columns
df_numeric = df_original[features_to_scale].copy()

scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_numeric)

df_scaled = pd.DataFrame(df_scaled_array, columns=features_to_scale, index=df_original.index)

# Guardar el modelo de escalado y el dataset escalado
joblib.dump(scaler, 'scaler.pkl')
df_scaled.to_csv('cannabis_data_scaled.csv')
print("✅ Modelo de escalado y datos escalados guardados.")
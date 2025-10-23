from statsmodels.tsa.stattools import adfuller
import pandas as pd

# 1️⃣ Crear una serie de ejemplo
datos = [100, 110, 115, 120, 130, 150, 160]

# 2️⃣ Crear el DataFrame
df = pd.DataFrame({'valor': datos})

# 3️⃣ Calcular la primera diferencia
df['diferencia_1'] = df['valor'].diff()

# 4️⃣ Eliminar valores NaN antes de aplicar la prueba
serie_diferenciada = df['diferencia_1'].dropna()

# 5️⃣ Aplicar la prueba ADF (Augmented Dickey-Fuller)
resultado = adfuller(serie_diferenciada)

# 6️⃣ Mostrar resultados
print('Estadístico ADF:', resultado[0])
print('p-valor:', resultado[1])

# (Opcional) mostrar el DataFrame para ver las diferencias
print(df)



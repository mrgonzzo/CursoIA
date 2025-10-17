import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# CARGAR (usa el nombre de tu dataset: 'servidor_data.csv','hardware_faults.csv', etc.)
df = pd.read_csv('servidor_data.csv')
print(f"Shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe())
# EXPLORAR DESBALANCE columna objetivo
print('='*10)
print('EXPLORAR DESBALANCE columna objetivo: Servidor_Saturado ')
print('='*10)
print(df['Servidor_Saturado'].value_counts())
sns.countplot(x='Servidor_Saturado', data=df)
plt.show()
# REVISAR FALTANTES
print('='*10)
print('REVISAR FALTANTES')
print('='*10)
print(df.isnull().sum())
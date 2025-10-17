import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('servidor_data.csv')
# Reemplaza 'target' por el nombre de tu columna objetivo
X = df.drop('Servidor_Saturado', axis=1)
y = df['Servidor_Saturado']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Train: {X_train_scaled.shape}")
print(f"Test: {X_test_scaled.shape}")
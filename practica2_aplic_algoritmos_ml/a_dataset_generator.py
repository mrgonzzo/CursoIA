#from sklearn.datasets import make_classification
#import pandas as pd
#X, y = make_classification(n_samples=500, n_features=5,
#n_informative=4, random_state=42)
#df = pd.DataFrame(X, columns=['CPU_%', 'Memoria_%', 'Conexiones','Ancho_banda_MB', 'Latencia_ms'])
#df['Servidor_Saturado'] = y
#df.to_csv('servidor_data.csv', index=False)
#print(df.head())
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=500,
                           n_features=5,
                           n_informative=4,
                           n_redundant=0,  # Evita exceder el límite
                           random_state=42)

df = pd.DataFrame(X, columns=['CPU_%', 'Memoria_%', 'Conexiones', 'Ancho_banda_MB', 'Latencia_ms'])
df['Servidor_Saturado'] = y
df.to_csv('servidor_data.csv', index=False)
print(df.head())
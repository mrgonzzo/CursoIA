import wooldridge as woo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analizar_dataset(nombre_dataset):
    """Analiza cualquier dataset de wooldridge con gráficos"""
    df = woo.dataWoo(nombre_dataset)

    print(f"\n{'=' * 60}")
    print(f"DATASET: {nombre_dataset}")
    print(f"{'=' * 60}\n")

    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    print("VARIABLES:", df.columns.tolist())
    print("\nVALORES FALTANTES:\n", df.isnull().sum())
    print("\nESTADÍSTICAS DESCRIPTIVAS:\n", df.describe())

    # Seleccionar solo columnas numéricas
    df_num = df.select_dtypes(include=['number'])
    num_vars = len(df_num.columns)

    if num_vars > 0:
        # Gráfico 1: Histogramas
        fig, axes = plt.subplots(nrows=(num_vars + 2) // 3, ncols=3, figsize=(15, 4 * ((num_vars + 2) // 3)))
        axes = axes.flatten() if num_vars > 1 else [axes]

        for i, col in enumerate(df_num.columns):
            axes[i].hist(df_num[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Frecuencia')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # Gráfico 2: Boxplots
        fig, axes = plt.subplots(nrows=(num_vars + 2) // 3, ncols=3, figsize=(15, 4 * ((num_vars + 2) // 3)))
        axes = axes.flatten() if num_vars > 1 else [axes]

        for i, col in enumerate(df_num.columns):
            axes[i].boxplot(df_num[col].dropna())
            axes[i].set_title(f'{col}')
            axes[i].set_ylabel('Valor')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # Gráfico 3: Matriz de correlación
        if num_vars > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_num.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Matriz de Correlación')
            plt.tight_layout()
            plt.show()

    return df


# CAMBIAR SOLO ESTA LÍNEA:
listado_datasets = woo.dataWoo()

print(f"¿Cuantos datasets hay en el paquete Wooldridge? \n Hay {len(listado_datasets)} mi señor")

for dataset in listado_datasets:
     print(f"este es el data sel leido en cada iteracion : {dataset} \n")
     #df = analizar_dataset(dataset)

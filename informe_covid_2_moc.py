"""
Análisis de datos COVID-19 (Ingresos UCI y Fallecidos diarios)
Autor: Asistente (analista de datos)
Objetivo: aplicar búsqueda, ordenación y regresión lineal (OLS)
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Callable, Any
import numpy as np

# ----------------------------- CARGA Y PREPARACIÓN -----------------------------

def cargar_datos() -> pd.DataFrame:
    """Carga el CSV desde la ruta especificada y convierte las fechas."""
    path = r'C:\\Users\\tarde\\Desktop\\ProgramacionCursoIA\\DATOS\\covid_data.csv'
    df = pd.read_csv(path)
    if 'Fecha_r' in df.columns:
        df['Fecha_dt'] = pd.to_datetime(df['Fecha_r'], errors='coerce')
    else:
        df['Fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True)
    return df


def columna_a_lista(df: pd.DataFrame, campo: str) -> List[Tuple[datetime, Any]]:
    return [(row['Fecha_dt'], row.get(campo, None)) for _, row in df.iterrows()]

# ----------------------------- REGRESIÓN OLS -----------------------------

def calcular_recta_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calcula los coeficientes de la recta por mínimos cuadrados ordinarios (OLS).
    Devuelve (pendiente b, intercepto a)
    """
    n = len(x)
    media_x = np.mean(x)
    media_y = np.mean(y)
    b = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x)**2)
    a = media_y - b * media_x
    return a, b

# ----------------------------- VISUALIZACIONES -----------------------------

def graficar_con_tendencia(df: pd.DataFrame, campo: str, nombre_salida: str):
    """
    Grafica la serie temporal y añade la recta de tendencia calculada por OLS.
    Exporta el resultado a PNG.
    """
    df_sorted = df.sort_values('Fecha_dt').dropna(subset=[campo])
    x = np.arange(len(df_sorted))
    y = df_sorted[campo].values

    a, b = calcular_recta_ols(x, y)
    tendencia = a + b * x

    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['Fecha_dt'], y, label=campo, color='blue')
    plt.plot(df_sorted['Fecha_dt'], tendencia, color='orange', linestyle='--', label='Tendencia (OLS)')

    plt.title(f"Tendencia (OLS) para {campo}")
    plt.xlabel("Fecha")
    plt.ylabel("Casos diarios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ruta_guardado = f'C:/Users/tarde/Desktop/ProgramacionCursoIA/{nombre_salida}.png'
    plt.savefig(ruta_guardado)
    plt.close()

    print(f"\nGráfico '{campo}' con tendencia OLS guardado en:\n{ruta_guardado}")
    print(f"  - Pendiente (b): {b:.4f}")
    print(f"  - Intercepto (a): {a:.4f}")

# ----------------------------- EJECUCIÓN PRINCIPAL -----------------------------

if __name__ == '__main__':
    print("=== Inicio del análisis de COVID-19 ===")
    df = cargar_datos()
    print(f"Datos cargados correctamente: {len(df)} filas")

    # Gráficos con tendencia OLS
    graficar_con_tendencia(df, 'Ingresos_UCI_diarios', 'tendencia_UCI')
    graficar_con_tendencia(df, 'Fallecidos_diarios', 'tendencia_fallecidos')

    print("\n=== Fin del análisis ===")

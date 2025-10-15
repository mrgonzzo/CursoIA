"""
Análisis de datos COVID-19 (Ingresos UCI y Fallecidos diarios)
Autor: Asistente (analista de datos)
Propósito: Implementar algoritmos de búsqueda y ordenación, generar visualizaciones
y marcar eventos clave (vacuna y confinamiento) en las series temporales.

Rutas de trabajo:
- CSV: C:\\Users\\tarde\\Desktop\\ProgramacionCursoIA\\DATOS\\covid_data.csv
- Script: C:\\Users\\tarde\\Desktop\\ProgramacionCursoIA\\Analisis_COVID_UCI_fallecidos.py

Ejecución:
    python Analisis_COVID_UCI_fallecidos.py

Contenido:
1) Carga y limpieza de datos (pandas)
2) Algoritmos de búsqueda y ordenación (explicativos)
3) Visualizaciones (gráficos y exportación PNG)
4) Marcadores de eventos (vacuna y confinamiento)
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Callable, Any

# ----------------------------- CARGA Y PREPARACIÓN -----------------------------

def cargar_datos() -> pd.DataFrame:
    """Carga el CSV desde la ruta especificada y convierte las fechas."""
    path = r'C:\\Users\\tarde\\Desktop\\ProgramacionCursoIA\\DATOS\\ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv'
    df = pd.read_csv(path)
    if 'Fecha_r' in df.columns:
        df['Fecha_dt'] = pd.to_datetime(df['Fecha_r'], errors='coerce')
    else:
        df['Fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce', dayfirst=True)
    return df


def columna_a_lista(df: pd.DataFrame, campo: str) -> List[Tuple[datetime, Any]]:
    return [(row['Fecha_dt'], row.get(campo, None)) for _, row in df.iterrows()]

# ----------------------------- BÚSQUEDA -----------------------------

def busqueda_lineal(arr: List[Tuple[datetime, Any]], predicate: Callable[[Tuple[datetime, Any]], bool]) -> int:
    for i, item in enumerate(arr):
        if predicate(item):
            return i
    return -1

def busqueda_binaria(arr: List[Tuple[datetime, Any]], target: datetime) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid][0] == target:
            return mid
        elif arr[mid][0] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# ----------------------------- ORDENACIÓN -----------------------------

def insertion_sort(arr: List[Tuple[datetime, Any]], key=lambda x: x[1]):
    a = arr.copy()
    for i in range(1, len(a)):
        current = a[i]
        j = i - 1
        while j >= 0 and key(a[j]) > key(current):
            a[j+1] = a[j]
            j -= 1
        a[j+1] = current
    return a

def merge_sort(arr: List[Tuple[datetime, Any]], key=lambda x: x[1]):
    if len(arr) <= 1:
        return arr.copy()
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)
    return _merge(left, right, key)

def _merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr: List[Tuple[datetime, Any]], key=lambda x: x[1]):
    if len(arr) <= 1:
        return arr.copy()
    pivot = key(arr[len(arr)//2])
    left = [x for x in arr if key(x) < pivot]
    middle = [x for x in arr if key(x) == pivot]
    right = [x for x in arr if key(x) > pivot]
    return quick_sort(left, key) + middle + quick_sort(right, key)

# ----------------------------- FUNCIONES DE ALTO NIVEL -----------------------------

def ordenar_por_campo(df: pd.DataFrame, campo: str, metodo: str = 'merge') -> List[Tuple[datetime, Any]]:
    lista = columna_a_lista(df, campo)
    if metodo == 'insertion':
        return insertion_sort(lista, key=lambda x: (x[1] if x[1] is not None else float('inf')))
    elif metodo == 'quick':
        return quick_sort(lista, key=lambda x: (x[1] if x[1] is not None else float('inf')))
    else:
        return merge_sort(lista, key=lambda x: (x[1] if x[1] is not None else float('inf')))

def top_k_por_campo(df: pd.DataFrame, campo: str, k: int = 5, metodo_sort: str = 'merge') -> List[Tuple[datetime, Any]]:
    ordenada = ordenar_por_campo(df, campo, metodo_sort)
    if k <= 0:
        return []
    return ordenada[-k:][::-1]

def buscar_primer_dia_condicion(df: pd.DataFrame, predicate: Callable[[pd.Series], bool]) -> Tuple[pd.Timestamp, int]:
    df_sorted = df.sort_values('Fecha_dt')
    for _, row in df_sorted.iterrows():
        if predicate(row):
            return row['Fecha_dt'], int(row.get('Ingresos_UCI_diarios', 0))
    return None, None

# ----------------------------- VISUALIZACIONES -----------------------------

def graficar_series_temporales(df: pd.DataFrame):
    df_sorted = df.sort_values('Fecha_dt')

    # Identificar fechas de eventos para marcarlas
    fechas_vacuna = df_sorted.loc[df_sorted['Hay_vacuna'] == 1, 'Fecha_dt']
    fechas_confinamiento = df_sorted.loc[df_sorted['Hay_Confinamiento'] == 1, 'Fecha_dt']

    # Gráfico comparativo con eventos
    plt.figure(figsize=(12,6))
    plt.plot(df_sorted['Fecha_dt'], df_sorted['Ingresos_UCI_diarios'], label='Ingresos UCI', color='blue')
    plt.plot(df_sorted['Fecha_dt'], df_sorted['Fallecidos_diarios'], label='Fallecidos', color='red')

    # Marcar eventos importantes con líneas verticales
    for f in fechas_vacuna:
        plt.axvline(x=f, color='green', linestyle='--', alpha=0.4)
    for f in fechas_confinamiento:
        plt.axvline(x=f, color='orange', linestyle=':', alpha=0.5)

    plt.title('Evolución COVID-19 con eventos: Vacuna (verde) y Confinamiento (naranja)')
    plt.xlabel('Fecha')
    plt.ylabel('Casos diarios')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(r'C:\\Users\\tarde\\Desktop\\ProgramacionCursoIA\\grafico_eventos.png')
    plt.close()

    print('\nSe generó el gráfico con marcadores de eventos:')
    print('- C:/Users/tarde/Desktop/ProgramacionCursoIA/grafico_eventos.png')

# ----------------------------- EJECUCIÓN PRINCIPAL -----------------------------

if __name__ == '__main__':
    print("=== Inicio: Análisis educativo de datos COVID - Ingresos UCI y Fallecidos ===")
    df = cargar_datos()
    print('Datos cargados: filas =', len(df))

    top7_uci = top_k_por_campo(df, 'Ingresos_UCI_diarios', k=7)
    print('\nTop 7 días con más Ingresos_UCI_diarios:')
    for fecha, valor in top7_uci:
        print(fecha.date(), int(valor))

    top7_fall = top_k_por_campo(df, 'Fallecidos_diarios', k=7)
    print('\nTop 7 días con más Fallecidos_diarios:')
    for fecha, valor in top7_fall:
        print(fecha.date(), int(valor))

    fecha_vac_uci, uci_val = buscar_primer_dia_condicion(df, lambda r: (r.get('Hay_vacuna', 0) == 1) and (r.get('Ingresos_UCI_diarios', 0) > 50))
    print('\nPrimer día con vacuna y UCI>50:')
    if fecha_vac_uci is not None:
        print(fecha_vac_uci.date(), 'Ingresos UCI =', uci_val)
    else:
        print('No encontrado.')

    graficar_series_temporales(df)

    print('\n=== Fin del análisis y generación de gráficos con eventos ===')

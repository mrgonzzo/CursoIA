#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo de Series Temporales con ARIMA (Plotly + Matplotlib)
IMPLEMENTA: Estructura Box-Jenkins, Carga Dinámica, Gráficos Plotly,
GENERACIÓN DE INFORME PDF, AJUSTE INTERACTIVO DE RANGOS (p, d, q),
Y EMULACIÓN DE SALIDA CONSOLA/GRÁFICOS MATPLOTLIB DEL SEGUNDO SCRIPT.
"""

import wooldridge as woo
import pandas as pd
import numpy as np
import sys
import io
import os
import re
from datetime import datetime

# Librerías estadísticas
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Librerías de visualización y reportes (Plotly y Matplotlib)
import warnings
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns # Para estilo seaborn en Matplotlib

# Requiere: pip install fpdf2
try:
    from fpdf import FPDF
except ImportError:
    print("⚠️ ADVERTENCIA: La librería 'fpdf2' no está instalada. El informe PDF no se generará. Ejecuta: pip install fpdf2")
    class FPDF:  # Dummy class
        def __init__(self, *args, **kwargs): pass
        def add_page(self): pass
        def set_font(self, *args, **kwargs): pass
        def multi_cell(self, *args, **kwargs): pass
        def image(self, *args, **kwargs): pass
        def output(self, *args, **kwargs): pass

# Deshabilitar advertencias
warnings.filterwarnings('ignore')

# Configuración de estilo Matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN DE RUTAS Y CAPTURA DE SALIDA
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_PATH / "data"
OUTPUT_DIR = SCRIPT_PATH / "arima_interact"
OUTPUT_DIR.mkdir(exist_ok=True)

DOCUMENTO_TEXTO = []
IMAGENES_REPORT = []
TS_NAME = "Serie Temporal"

# Archivos de datos (para referencia en el menú)
FILE_TS = 'TS.csv'
FILE_VP = 'ventasypubl.csv'
FILE_XLSX = 'datos_adicionales.xlsx'  # Placeholder


# Clase para capturar la salida de la consola pero también imprimirla
class ConsoleAndCapture:
    def __init__(self, document_list):
        self.document_list = document_list
        self.original_stdout = sys.stdout

    def __enter__(self):
        self.buffer = io.StringIO()
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        # Asegurarse de que el buffer capturado se añada al documento al salir
        captured_lines = self.buffer.getvalue().splitlines()
        self.document_list.extend(captured_lines)


    def write(self, text):
        self.original_stdout.write(text)  # Imprime en la consola real
        self.buffer.write(text) # Captura en el buffer

    def flush(self):
        self.original_stdout.flush() # Asegura que la salida se vacía inmediatamente


# ============================================================================
# FUNCIONES AUXILIARES Y MÉTRICAS
# ============================================================================

def test_adf(serie, nombre):
    """Test de Dickey-Fuller Aumentado"""
    result = adfuller(serie.dropna())
    return {
        'nombre': nombre,
        'estadistico': result[0],
        'pvalor': result[1],
        'es_estacionaria': result[1] <= 0.05
    }


def calcular_coeficiente_theil(actual, prediccion):
    """Calcula el Coeficiente de Desigualdad de Theil (Theil's U) y otras métricas"""
    actual = np.array(actual)
    prediccion = np.array(prediccion)
    mse = np.mean((actual - prediccion) ** 2)
    rmse = np.sqrt(mse)
    denominador = np.sqrt(np.mean(actual ** 2)) + np.sqrt(np.mean(prediccion ** 2))
    theil_u = rmse / denominador if denominador != 0 else np.inf
    mae = np.mean(np.abs(actual - prediccion))
    mape = np.mean(np.abs((actual - prediccion) / actual)) * 100 if np.all(actual != 0) else np.inf

    # Descomposición de Theil (añadido para emular el segundo script)
    mean_actual = np.mean(actual)
    mean_pred = np.mean(prediccion)
    bias = (mean_pred - mean_actual) ** 2
    var_actual = np.var(actual)
    var_pred = np.var(prediccion)
    bias_proportion = bias / mse if mse != 0 else 0
    # Modificación para evitar errores si var_actual o var_pred son cero (varianzas iguales, o series constantes)
    variance_proportion = (np.sqrt(var_pred) - np.sqrt(var_actual)) ** 2 / mse if mse != 0 else 0
    covariance_proportion = 1 - bias_proportion - variance_proportion if (bias_proportion + variance_proportion) <= 1 else 0


    return {'Theil_U': theil_u, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
            'MSE': mse, 'Bias_Proportion': bias_proportion,
            'Variance_Proportion': variance_proportion, 'Covariance_Proportion': covariance_proportion}


# ============================================================================
# FUNCIONES DE CARGA Y PARÁMETROS INTERACTIVOS
# ============================================================================

def solicitar_seleccion():
    """Muestra el menú de series temporales disponibles y solicita la elección del usuario."""

    print("\n" + "=" * 80)
    print("SELECCIÓN DE LA SERIE TEMPORAL")
    print("El análisis se detendrá si no se encuentra el archivo .csv o .xlsx.")
    print("=" * 80)
    print("Por favor, elija la serie temporal a analizar:")
    print("  1) Importaciones de Bario Chino (Mensual, Wooldridge)")
    print(f"  2) Serie 'x' (Mensual, Archivo: data/{FILE_TS})")
    print(f"  3) Ventas 'vtas' (Trimestral, Archivo: data/{FILE_VP})")
    print(f"  4) Publicidad 'pub' (Trimestral, Archivo: data/{FILE_VP})")
    print(f"  5) Datos Adicionales (Placeholder, Archivo: data/{FILE_XLSX})")

    while True:
        try:
            choice = input("Ingrese el número de su elección (1-5): ")
            choice = int(choice)
            if 1 <= choice <= 5:
                return choice
            else:
                print("Opción no válida. Por favor, ingrese un número entre 1 y 5.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número entero.")


def cargar_datos_seleccionados(choice):
    """Carga y prepara la serie temporal elegida."""
    ts = None;
    freq = None;
    file_path = None

    try:
        if choice == 1:
            df = woo.dataWoo('barium')
            serie_val = df['chnimp'].values
            n = len(serie_val)
            fechas = pd.date_range(start='1978-02', periods=n, freq='MS')
            ts = pd.Series(serie_val, index=fechas, name='Importaciones de Bario Chino (Wooldridge)')
            freq = 'MS'
            # Emulación del segundo script:
            # Estos prints son redundantes si ya se está capturando la salida principal.
            # Los he dejado aquí para que el flujo de la emulación sea más claro, pero
            # se eliminarán del DOCUMENTO_TEXTO si la captura principal ya los incluye.

        elif choice == 2:
            file_path = DATA_DIR / FILE_TS
            df_ts = pd.read_csv(file_path)
            serie_val = df_ts['x'].values
            n = len(serie_val)
            fechas = pd.date_range(start='2010-01', periods=n, freq='MS')
            ts = pd.Series(serie_val, index=fechas, name=f"Serie 'x' de {FILE_TS} (Mensual)")
            freq = 'MS'

        elif choice == 3:
            file_path = DATA_DIR / FILE_VP
            df_vp = pd.read_csv(file_path, parse_dates=['dateid01'])
            df_vp = df_vp.set_index('dateid01')
            ts = df_vp['vtas']
            ts.name = f"Ventas 'vtas' de {FILE_VP} (Trimestral)"
            freq = 'QS'

        elif choice == 4:
            file_path = DATA_DIR / FILE_VP
            df_vp = pd.read_csv(file_path, parse_dates=['dateid01'])
            df_vp = df_vp.set_index('dateid01')
            ts = df_vp['pub']
            ts.name = f"Publicidad 'pub' de {FILE_VP} (Trimestral)"
            freq = 'QS'

        elif choice == 5:
            file_path = DATA_DIR / FILE_XLSX
            print(
                f"\n⚠️ Advertencia: El archivo '{FILE_XLSX}' no ha sido cargado/no existe en la ruta {file_path}. Saliendo.")
            return None, None

        print(f"   ✓ Datos cargados desde: {file_path if file_path else 'Wooldridge'}")
        global TS_NAME
        TS_NAME = ts.name.split('(')[0].strip()
        return ts, freq

    except FileNotFoundError:
        print(f"\n❌ Error de Archivo: No se encontró el archivo necesario en la ruta esperada: {file_path.absolute()}")
        return None, None
    except Exception as e:
        print(f"\n❌ Error al cargar los datos: {e}")
        return None, None


def solicitar_rangos_arima():
    """Solicita al usuario los órdenes máximos para el Grid Search."""
    print("\n" + "=" * 80)
    print("AJUSTE INTERACTIVO DE PARÁMETROS ARIMA (Grid Search)")
    print("Defina los órdenes máximos para p (AR), d (I) y q (MA).")
    print("El Grid Search probará todos los órdenes desde 0 hasta el valor máximo especificado.")
    print("=" * 80)

    # Valores por defecto para p, q y d
    p_max_def = 2
    q_max_def = 2
    d_max_def = 1

    def get_input_int(prompt, default_val, max_val=4):
        while True:
            try:
                user_input = input(f"{prompt} (0 a {max_val}) [Default: {default_val}]: ")
                if not user_input:
                    return default_val
                val = int(user_input)
                if 0 <= val <= max_val:
                    return val
                else:
                    print(f"Valor fuera de rango (0 a {max_val}). Inténtelo de nuevo.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero.")

    p_max = get_input_int("Orden Máximo AR (p)", p_max_def)
    q_max = get_input_int("Orden Máximo MA (q)", q_max_def)
    d_max = get_input_int("Orden Máximo de Diferenciación (d)", d_max_def, max_val=2)

    return range(0, p_max + 1), range(0, d_max + 1), range(0, q_max + 1)


# ============================================================================
# FUNCIONES DE PLOTLY Y MATPLOTLIB (Gráficos)
# ============================================================================

# Variable global para almacenar la serie original para Matplotlib en plot_transformaciones
# dado que no se pasa explicitamente en el script original del segundo tipo de output.
_global_ts_for_mpl_transforms = None

def guardar_grafico(fig, filename_base, title, show_plot=False):
    """Guarda el gráfico de Plotly como PNG y HTML, lo añade al reporte."""
    if not title.startswith("PASO"):
        title = f"Gráfico - {title}"

    png_path = OUTPUT_DIR / f"{filename_base}.png"
    html_path = OUTPUT_DIR / f"{filename_base}.html"

    try:
        fig.write_image(str(png_path), scale=1)
        IMAGENES_REPORT.append({'title': title, 'path': png_path})
    except Exception as e:
        # Imprimir directamente en la consola real para asegurar que se vea el error
        sys.__stdout__.write(f"❌ Error al guardar PNG para Plotly (¿Falta 'kaleido'?): {e}\n")

    fig.write_html(str(html_path))
    if show_plot: # Esto es opcional, Plotly puede abrirse en navegador
        pass # fig.show()


def plot_serie_y_distribucion(ts_data):
    # Plotly
    fig_plotly = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Serie Temporal Original: {ts_data.name}', 'Distribución de la Serie'))
    fig_plotly.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Serie Original', line=dict(color='#2E86AB')),
                  row=1, col=1)
    fig_plotly.add_trace(go.Histogram(x=ts_data.values, name='Distribución', marker_color='#A23B72'), row=1, col=2)
    fig_plotly.update_layout(height=500, title_text="PASO 1: Análisis Descriptivo (Visualización Inicial)", showlegend=False)
    guardar_grafico(fig_plotly, "01_Serie_Distribucion", "Visualización y Distribución")

    # Matplotlib (emulación del segundo script)
    fig_mpl, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(ts_data, linewidth=2, color='#2E86AB')
    axes[0].set_title(f'Serie Temporal Original: {ts_data.name.split("(")[0].strip()}', # Nombre sin paréntesis
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fecha', fontsize=11)
    axes[0].set_ylabel('Importaciones (toneladas)', fontsize=11) # Hardcoded, ajustar si la serie es diferente
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(ts_data, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].set_title('Distribución de la Serie', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Importaciones (toneladas)', fontsize=11) # Hardcoded
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_descomposicion(ts_data, periodo):
    decomposition = seasonal_decompose(ts_data, model='additive', period=periodo)

    # Plotly
    fig_plotly = make_subplots(rows=4, cols=1,
                        subplot_titles=('Serie Original', 'Tendencia', 'Componente Estacional', 'Residuos'),
                        shared_xaxes=True)
    fig_plotly.add_trace(
        go.Scatter(x=ts_data.index, y=decomposition.observed, mode='lines', name='Observado', line=dict(color='#2E86AB')),
        row=1, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=ts_data.index, y=decomposition.trend, mode='lines', name='Tendencia', line=dict(color='#F18F01')),
        row=2, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=ts_data.index, y=decomposition.seasonal, mode='lines', name='Estacional', line=dict(color='#C73E1D')),
        row=3, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=ts_data.index, y=decomposition.resid, mode='lines', name='Residuos', line=dict(color='#6A994E')), row=4,
        col=1)
    fig_plotly.update_layout(height=800, title_text=f"PASO 2: Descomposición Estacional Aditiva (Período={periodo})",
                      showlegend=False)
    guardar_grafico(fig_plotly, "02_Descomposicion", "Descomposición Clásica")

    # Matplotlib (emulación del segundo script)
    fig_mpl, axes = plt.subplots(4, 1, figsize=(14, 10))
    decomposition.observed.plot(ax=axes[0], color='#2E86AB', linewidth=2)
    axes[0].set_title('Serie Original', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valor', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    decomposition.trend.plot(ax=axes[1], color='#F18F01', linewidth=2)
    axes[1].set_title('Tendencia', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Tendencia', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    decomposition.seasonal.plot(ax=axes[2], color='#C73E1D', linewidth=2)
    axes[2].set_title('Componente Estacional', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Estacionalidad', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    decomposition.resid.plot(ax=axes[3], color='#6A994E', linewidth=1)
    axes[3].set_title('Residuos', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Residuos', fontsize=10)
    axes[3].set_xlabel('Fecha', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_transformaciones(original_ts, transformaciones_dict):
    global _global_ts_for_mpl_transforms # Para que Matplotlib pueda dibujar la original

    # Almacenar la serie original para Matplotlib en plot_transformaciones
    _global_ts_for_mpl_transforms = original_ts

    valid_trans = {k: v for k, v in transformaciones_dict.items() if not v.empty and len(v.dropna()) > 10}
    n_plots = len(valid_trans)
    if n_plots == 0: return

    # Plotly
    rows_plotly = (n_plots + 1) // 2 if n_plots > 0 else 1
    fig_plotly = make_subplots(rows=rows_plotly, cols=2, subplot_titles=list(valid_trans.keys()), shared_xaxes=False)

    for idx, (nombre, serie_trans) in enumerate(valid_trans.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        fig_plotly.add_trace(
            go.Scatter(x=serie_trans.index, y=serie_trans.values, mode='lines', name=nombre, line=dict(width=1.5)),
            row=row, col=col)

    fig_plotly.update_layout(height=400 * rows_plotly, title_text=f"PASO 4: Visualización de Transformaciones para Estacionariedad",
                      showlegend=False)
    guardar_grafico(fig_plotly, "04_Transformaciones", "Visualización de Transformaciones")

    # Matplotlib (emulación del segundo script)
    # El segundo script muestra hasta 7 transformaciones + la original, en un grid 4x2
    fig_mpl, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.ravel()

    # Primera subplot es la serie original
    axes[0].plot(original_ts, linewidth=1.5, color='#2E86AB')
    axes[0].set_title('Serie Original', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Valor', fontsize=9)

    for idx, (nombre, serie_trans) in enumerate(valid_trans.items(), 1): # Empieza en 1 para la siguiente subplot
        if idx >= len(axes): # Evitar desbordamiento si hay más transformaciones que subplots disponibles
            break
        axes[idx].plot(serie_trans, linewidth=1.5, color=plt.cm.tab10(idx))
        axes[idx].set_title(nombre, fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylabel('Valor', fontsize=9)
        if idx >= 6: # Para las últimas dos filas (índices 6 y 7)
            axes[idx].set_xlabel('Fecha', fontsize=9)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_acf_pacf(serie, nombre, lags):
    # Plotly
    acf_values = acf(serie.dropna(), nlags=lags, fft=False)[1:]
    pacf_values = pacf(serie.dropna(), nlags=lags)[1:]
    conf_limit = 1.96 / np.sqrt(len(serie.dropna()))
    lags_index = np.arange(1, lags + 1)

    fig_plotly = make_subplots(rows=2, cols=1, subplot_titles=(f'ACF - {nombre} (sin lag 0)', f'PACF - {nombre} (sin lag 0)'))

    fig_plotly.add_trace(go.Bar(x=lags_index, y=acf_values, name='ACF', marker_color='#2E86AB'), row=1, col=1)
    fig_plotly.add_trace(go.Scatter(x=lags_index, y=[conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1)), row=1, col=1)
    fig_plotly.add_trace(go.Scatter(x=lags_index, y=[-conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=1, col=1)

    fig_plotly.add_trace(go.Bar(x=lags_index, y=pacf_values, name='PACF', marker_color='#C73E1D'), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(x=lags_index, y=[conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=2, col=1)
    fig_plotly.add_trace(go.Scatter(x=lags_index, y=[-conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=2, col=1)

    fig_plotly.update_layout(height=600, title_text="PASO 5: Autocorrelación y Autocorrelación Parcial (Identificación p, q)",
                      showlegend=False)
    fig_plotly.update_xaxes(title_text="Rezagos", row=2, col=1)
    guardar_grafico(fig_plotly, "05_ACF_PACF", "ACF y PACF para Identificación")

    # Matplotlib (emulación del segundo script)
    fig_mpl, axes = plt.subplots(2, 1, figsize=(14, 8))
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Importar aquí para evitar conflicto si es necesario
    plot_acf(serie, lags=lags, ax=axes[0], color='#2E86AB', alpha=0.7, zero=False)
    axes[0].set_title(f'ACF - {nombre} (sin lag 0)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rezagos', fontsize=10)
    axes[0].set_ylabel('Autocorrelación', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plot_pacf(serie, lags=lags, ax=axes[1], color='#C73E1D', alpha=0.7, zero=False)
    axes[1].set_title(f'PACF - {nombre} (sin lag 0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Rezagos', fontsize=10)
    axes[1].set_ylabel('Autocorrelación Parcial', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_diagnostico_residuos(residuos, orden, ts_name):
    # Plotly
    fig_plotly = make_subplots(rows=2, cols=2,
                        subplot_titles=('Serie de Residuos', 'Distribución de Residuos', 'ACF de Residuos (sin lag 0)',
                                        'Q-Q Plot'))

    fig_plotly.add_trace(
        go.Scatter(x=residuos.index, y=residuos.values, mode='lines', name='Residuos', line=dict(color='#2E86AB')),
        row=1, col=1)
    fig_plotly.add_trace(go.Scatter(x=residuos.index, y=[0] * len(residuos), mode='lines', name='Cero',
                             line=dict(color='red', dash='dash')), row=1, col=1)

    fig_plotly.add_trace(go.Histogram(x=residuos.values, name='Distribución', marker_color='#A23B72'), row=1, col=2)

    acf_values = acf(residuos, nlags=30, fft=False)[1:]
    conf_limit = 1.96 / np.sqrt(len(residuos))
    lags_index = np.arange(1, 31)
    fig_plotly.add_trace(go.Bar(x=lags_index, y=acf_values, name='ACF Residuos', marker_color='#F18F01'), row=2, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=lags_index, y=[conf_limit] * 30, mode='lines', line=dict(dash='dash', color='red', width=1),
                   showlegend=False), row=2, col=1)
    fig_plotly.add_trace(
        go.Scatter(x=lags_index, y=[-conf_limit] * 30, mode='lines', line=dict(dash='dash', color='red', width=1),
                   showlegend=False), row=2, col=1)

    qq_data = stats.probplot(residuos, dist="norm", fit=True)
    fig_plotly.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Puntos', marker=dict(color='#C73E1D')),
        row=2, col=2)
    fig_plotly.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1], mode='lines', name='Línea Teórica',
                   line=dict(color='gray', dash='dash')), row=2, col=2)

    fig_plotly.update_layout(height=700, title_text=f"PASO 8: Diagnóstico de Residuos - ARIMA{orden} ({ts_name})",
                      showlegend=False)
    guardar_grafico(fig_plotly, "08_Diagnostico_Residuos", "Diagnóstico de Residuos")

    # Matplotlib (emulación del segundo script)
    fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(residuos, linewidth=1, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0, 0].set_title('Residuos del Modelo', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Observación', fontsize=10)
    axes[0, 0].set_ylabel('Residuo', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].hist(residuos, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residuo', fontsize=10)
    axes[0, 1].set_ylabel('Frecuencia', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    from statsmodels.graphics.tsaplots import plot_acf # Se importa de nuevo por si acaso el scope
    plot_acf(residuos, lags=30, ax=axes[1, 0], color='#F18F01', alpha=0.7, zero=False)
    axes[1, 0].set_title('ACF de Residuos (sin lag 0)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    stats.probplot(residuos, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_prediccion_final(ts_hist, forecast_values, forecast_ci_df, mejor_orden, metricas_test, mejor_modelo, test_set_values, predicciones_test_set):
    # Plotly
    fig_plotly = go.Figure()

    fig_plotly.add_trace(go.Scatter(x=forecast_ci_df.index, y=forecast_ci_df.iloc[:, 1], mode='lines', name='IC Sup 95%',
                             line=dict(width=0), fillcolor='rgba(199, 62, 29, 0.2)', fill='tonexty'))
    fig_plotly.add_trace(go.Scatter(x=forecast_ci_df.index, y=forecast_ci_df.iloc[:, 0], mode='lines', name='IC Inf 95%',
                             line=dict(width=0), fill='tonexty', showlegend=False))

    fig_plotly.add_trace(go.Scatter(x=ts_hist.index, y=ts_hist.values, mode='lines', name='Serie Histórica',
                             line=dict(color='#2E86AB', width=2)))
    fig_plotly.add_trace(go.Scatter(x=test_set_values.index, y=test_set_values.values, mode='lines', name='Valores Reales (Test)',
                             line=dict(color='#6A994E', width=2, dash='dot')))
    fig_plotly.add_trace(go.Scatter(x=ts_hist.index, y=ts_hist.values, mode='lines', name='Serie Histórica',
                                    line=dict(color='#2E86AB', width=2)))
    fig_plotly.add_trace(
        go.Scatter(x=test_set_values.index, y=test_set_values.values, mode='lines', name='Valores Reales (Test)',
                   line=dict(color='#6A994E', width=2, dash='dot')))
    fig_plotly.add_trace(
        go.Scatter(x=predicciones_test_set.index, y=predicciones_test_set.values, mode='lines', name='Predicciones (Test)',
                   line=dict(color='#A23B72', width=2, dash='dash')))
    fig_plotly.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values.values, mode='lines', name='Pronóstico',
                                    line=dict(color='#F18F01', width=3)))

    fig_plotly.update_layout(
        title_text=f"PASO 9: Pronóstico Final ARIMA{mejor_orden} ({TS_NAME})",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        height=600,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
    )
    guardar_grafico(fig_plotly, "09_Prediccion_Final", "Pronóstico Final")

    # Matplotlib (emulación del segundo script)
    plt.figure(figsize=(16, 8))
    plt.plot(ts_hist, label='Serie Histórica', color='#2E86AB', linewidth=2)
    plt.plot(test_set_values, label='Valores Reales (Test)', color='#6A994E', linewidth=2, linestyle='dotted')
    plt.plot(predicciones_test_set, label='Predicciones (Test)', color='#A23B72', linewidth=2, linestyle='dashed')
    plt.plot(forecast_values, label='Pronóstico Futuro', color='#F18F01', linewidth=3)
    plt.fill_between(forecast_ci_df.index, forecast_ci_df.iloc[:, 0], forecast_ci_df.iloc[:, 1],
                     color='#C73E1D', alpha=0.1, label='IC 95%')
    plt.title(f'Pronóstico Final ARIMA{mejor_orden} ({TS_NAME})', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ============================================================================
# FUNCIONES DE REPORTE PDF
# ============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, f'Análisis de Serie Temporal: {TS_NAME}', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, f"Fecha del reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_image_to_report(self, title, image_path):
        self.chapter_title(title)
        self.image(str(image_path), x=10, w=self.w - 20)  # Ajustar ancho a la página
        self.ln(5)


def generar_reporte_pdf(documento_texto, imagenes_report, ts_name, mejor_orden, metricas_test):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.chapter_title("1. Introducción y Configuración")
    pdf.chapter_body(
        f"Este informe presenta un análisis detallado de la serie temporal '{ts_name}' utilizando el modelo ARIMA (AutoRegressive Integrated Moving Average).")
    pdf.chapter_body(
        f"El análisis sigue la metodología de Box-Jenkins, incluyendo visualización inicial, descomposición, pruebas de estacionariedad, identificación de órdenes p, d, q, ajuste del modelo, diagnóstico de residuos y pronóstico.")
    pdf.chapter_body(
        f"Los parámetros interactivos para el Grid Search de ARIMA se establecieron por el usuario durante la ejecución del script.")

    pdf.add_page()
    pdf.chapter_title("2. Salida de Consola del Análisis")
    pdf.set_font('Courier', '', 8)
    # Limitar el número de líneas o caracteres si la salida es muy extensa
    for line in documento_texto:
        pdf.multi_cell(0, 4, line)
    pdf.ln()

    # Añadir gráficos
    for img_info in imagenes_report:
        pdf.add_page()
        pdf.add_image_to_report(img_info['title'], img_info['path'])

    pdf.add_page()
    pdf.chapter_title(f"3. Resumen del Mejor Modelo y Métricas (ARIMA{mejor_orden})")
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, f"El mejor modelo ARIMA encontrado, basado en el criterio AIC, fue ARIMA{mejor_orden}.")
    pdf.ln(2)
    pdf.chapter_body("Métricas de rendimiento en el conjunto de prueba:")
    for metric, value in metricas_test.items():
        pdf.chapter_body(f"- {metric}: {value:.4f}")
    pdf.ln()
    pdf.chapter_body(
        "El Coeficiente de Theil (U) es particularmente útil para comparar la precisión del pronóstico con un pronóstico ingenuo. Valores cercanos a 0 indican un modelo muy bueno, mientras que valores mayores a 1 sugieren que un pronóstico ingenuo sería mejor.")
    pdf.chapter_body(
        "La descomposición de Theil (proporciones de Sesgo, Varianza y Covarianza) permite entender las fuentes del error de pronóstico: el sesgo indica errores sistemáticos, la varianza indica la capacidad del modelo para replicar la variabilidad de la serie, y la covarianza es el error no sistemático.")

    pdf_filename = OUTPUT_DIR / f"Reporte_ARIMA_{ts_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    try:
        pdf.output(str(pdf_filename))
        print(f"\n✅ Informe PDF generado exitosamente: {pdf_filename}")
    except Exception as e:
        sys.__stdout__.write(f"❌ Error al generar el informe PDF: {e}\n")


# ============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS BOX-JENKINS
# ============================================================================

def box_jenkins_arima(ts_original, freq_str):
    global TS_NAME
    # Renombrar la serie para evitar problemas con nombres muy largos/especiales
    if ' ' in ts_original.name or '(' in ts_original.name:
        ts_original.name = TS_NAME

    DOCUMENTO_TEXTO.append("\n" + "=" * 80)
    DOCUMENTO_TEXTO.append(f"INICIO DEL ANÁLISIS ARIMA PARA: {ts_original.name}")
    DOCUMENTO_TEXTO.append("=" * 80 + "\n")

    # PASO 1: Visualización Inicial y Descriptiva
    DOCUMENTO_TEXTO.append("\nPASO 1: Análisis Descriptivo (Visualización Inicial)")
    DOCUMENTO_TEXTO.append("--------------------------------------------------")
    DOCUMENTO_TEXTO.append(f"Serie: {ts_original.name}")
    DOCUMENTO_TEXTO.append(f"Número de observaciones: {len(ts_original)}")
    DOCUMENTO_TEXTO.append(f"Fecha de inicio: {ts_original.index.min().strftime('%Y-%m-%d')}")
    DOCUMENTO_TEXTO.append(f"Fecha de fin: {ts_original.index.max().strftime('%Y-%m-%d')}")
    DOCUMENTO_TEXTO.append(f"Resumen estadístico:\n{ts_original.describe().to_string()}")
    plot_serie_y_distribucion(ts_original)
    DOCUMENTO_TEXTO.append("Gráfico 01_Serie_Distribucion generado.")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 2: Descomposición de la Serie Temporal
    DOCUMENTO_TEXTO.append("PASO 2: Descomposición Estacional Clásica")
    DOCUMENTO_TEXTO.append("------------------------------------------")
    periodo_map = {'MS': 12, 'QS': 4, 'AS': 1}  # Mapping de frecuencia a período
    periodo = periodo_map.get(freq_str, 1)  # Por defecto 1 si no es mensual o trimestral
    if len(ts_original) > periodo * 2:  # Necesitamos al menos dos ciclos para la descomposición
        plot_descomposicion(ts_original, periodo)
        DOCUMENTO_TEXTO.append(f"Descomposición estacional realizada con período = {periodo}.")
        DOCUMENTO_TEXTO.append("Gráfico 02_Descomposicion generado.")
    else:
        DOCUMENTO_TEXTO.append(f"No hay suficientes datos para la descomposición estacional con período = {periodo}.")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 3: Prueba de Estacionariedad (ADF Test) - Serie Original
    DOCUMENTO_TEXTO.append("PASO 3: Prueba de Estacionariedad (ADF Test) - Serie Original")
    DOCUMENTO_TEXTO.append("-------------------------------------------------------------")
    adf_original = test_adf(ts_original, 'Serie Original')
    DOCUMENTO_TEXTO.append(f"Resultados ADF (Serie Original):\n"
                           f"  Estadístico ADF: {adf_original['estadistico']:.4f}\n"
                           f"  P-valor: {adf_original['pvalor']:.4f}\n"
                           f"  ¿Es estacionaria (p <= 0.05)?: {adf_original['es_estacionaria']}")
    if not adf_original['es_estacionaria']:
        DOCUMENTO_TEXTO.append("  La serie original NO es estacionaria. Se requerirá diferenciación.")
    else:
        DOCUMENTO_TEXTO.append("  La serie original ES estacionaria.")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 4: Transformaciones para Estacionariedad
    DOCUMENTO_TEXTO.append("PASO 4: Aplicación de Transformaciones para Lograr Estacionariedad")
    DOCUMENTO_TEXTO.append("------------------------------------------------------------------")
    transformaciones = {
        'Diferencia de Primer Orden': ts_original.diff(1).dropna(),
        'Diferencia Logarítmica': np.log(ts_original).diff(1).dropna() if np.all(ts_original > 0) else pd.Series(),
        'Diferencia Estacional': ts_original.diff(periodo).dropna() if periodo > 1 else pd.Series(),
        'Diferencia Estacional y de Primer Orden': (
            ts_original.diff(periodo).diff(1)).dropna() if periodo > 1 else pd.Series(),
        'Diferencia Logarítmica y Estacional': (np.log(ts_original).diff(periodo)).dropna() if np.all(
            ts_original > 0) and periodo > 1 else pd.Series(),
    }

    adf_results_transf = []
    for name, transformed_ts in transformaciones.items():
        if not transformed_ts.empty and len(transformed_ts) > 10:
            adf_result = test_adf(transformed_ts, name)
            adf_results_transf.append(adf_result)
            DOCUMENTO_TEXTO.append(f"Resultados ADF ({name}):\n"
                                   f"  Estadístico ADF: {adf_result['estadistico']:.4f}\n"
                                   f"  P-valor: {adf_result['pvalor']:.4f}\n"
                                   f"  ¿Es estacionaria (p <= 0.05)?: {adf_result['es_estacionaria']}")
        else:
            DOCUMENTO_TEXTO.append(f"Transformación '{name}' no aplicada o insuficiente para análisis.")

    plot_transformaciones(ts_original, transformaciones)
    DOCUMENTO_TEXTO.append("Gráfico 04_Transformaciones generado.")
    DOCUMENTO_TEXTO.append("\n")

    # Determinar la serie a usar para ACF/PACF (la más estacionaria o la primera diferencia)
    best_stationary_ts = ts_original
    best_stationary_name = "Serie Original"
    diff_order = 0

    for result in adf_results_transf:
        if result['es_estacionaria']:
            if "Diferencia de Primer Orden" in result['nombre'] and diff_order < 1:
                best_stationary_ts = transformaciones['Diferencia de Primer Orden']
                best_stationary_name = "Diferencia de Primer Orden"
                diff_order = 1
                break  # Usamos la primera diferencia si es estacionaria
            elif "Diferencia Estacional" in result['nombre'] and diff_order < 1:
                best_stationary_ts = transformaciones['Diferencia Estacional']
                best_stationary_name = "Diferencia Estacional"
                diff_order = 1
                break
            elif "Diferencia Estacional y de Primer Orden" in result['nombre'] and diff_order < 2:
                best_stationary_ts = transformaciones['Diferencia Estacional y de Primer Orden']
                best_stationary_name = "Diferencia Estacional y de Primer Orden"
                diff_order = 2
                break
            elif "Diferencia Logarítmica" in result['nombre'] and diff_order < 1 and np.all(ts_original > 0):
                best_stationary_ts = transformaciones['Diferencia Logarítmica']
                best_stationary_name = "Diferencia Logarítmica"
                diff_order = 1
                break
            elif "Diferencia Logarítmica y Estacional" in result['nombre'] and diff_order < 2 and np.all(
                    ts_original > 0):
                best_stationary_ts = transformaciones['Diferencia Logarítmica y Estacional']
                best_stationary_name = "Diferencia Logarítmica y Estacional"
                diff_order = 2
                break

    if not adf_original[
        'es_estacionaria'] and diff_order == 0:  # Si la original no es estacionaria y no encontramos una mejor, usamos la primera diferencia
        if not transformaciones['Diferencia de Primer Orden'].empty:
            best_stationary_ts = transformaciones['Diferencia de Primer Orden']
            best_stationary_name = "Diferencia de Primer Orden (Por Defecto)"
            diff_order = 1
        else:
            DOCUMENTO_TEXTO.append(
                "ADVERTENCIA: No se pudo encontrar una serie estacionaria adecuada. Usando la serie original.")
            best_stationary_ts = ts_original
            best_stationary_name = "Serie Original (No estacionaria)"
            diff_order = 0

    DOCUMENTO_TEXTO.append(
        f"Se utilizará '{best_stationary_name}' para identificar los órdenes p y q (d={diff_order}).")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 5: Gráficos de ACF y PACF
    DOCUMENTO_TEXTO.append("PASO 5: Identificación de Órdenes p (AR) y q (MA) con ACF/PACF")
    DOCUMENTO_TEXTO.append("----------------------------------------------------------------")
    lags = min(40, len(best_stationary_ts) // 2 - 1)  # No más de la mitad de las observaciones
    if lags <= 1:
        DOCUMENTO_TEXTO.append("No hay suficientes datos para calcular ACF/PACF significativos.")
    else:
        plot_acf_pacf(best_stationary_ts, best_stationary_name, lags)
        DOCUMENTO_TEXTO.append(f"Gráficos 05_ACF_PACF generados para '{best_stationary_name}' con {lags} rezagos.")
        DOCUMENTO_TEXTO.append("Interpretación de ACF y PACF para identificar posibles órdenes p y q.")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 6: División del dataset y Grid Search para ARIMA
    DOCUMENTO_TEXTO.append("PASO 6: Entrenamiento y Selección del Mejor Modelo ARIMA (Grid Search)")
    DOCUMENTO_TEXTO.append("--------------------------------------------------------------------")

    # Dividir datos en entrenamiento y prueba
    train_size = int(len(ts_original) * 0.8)
    train_data, test_data = ts_original[0:train_size], ts_original[train_size:]
    DOCUMENTO_TEXTO.append(
        f"Datos divididos: Entrenamiento ({len(train_data)} observaciones), Prueba ({len(test_data)} observaciones).")

    # Solicitar rangos de p, d, q
    p_values, d_values, q_values = solicitar_rangos_arima()
    DOCUMENTO_TEXTO.append(f"Rangos de búsqueda: p={list(p_values)}, d={list(d_values)}, q={list(q_values)}")

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(train_data, order=order)
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = order
                        best_model = model_fit
                    DOCUMENTO_TEXTO.append(f'ARIMA{order} - AIC: {model_fit.aic:.2f}')
                except Exception as e:
                    DOCUMENTO_TEXTO.append(f'ARIMA{order} - Error: {e}')
                    continue

    if best_order:
        DOCUMENTO_TEXTO.append(f"\nMEJOR MODELO ENCONTRADO:\n"
                               f"  Orden ARIMA: {best_order}\n"
                               f"  AIC Mínimo: {best_aic:.2f}")
    else:
        DOCUMENTO_TEXTO.append("No se pudo encontrar un modelo ARIMA óptimo. Revisar los datos o rangos.")
        return  # Salir si no hay un modelo válido

    DOCUMENTO_TEXTO.append("\n")

    # PASO 7: Predicción en el conjunto de prueba y evaluación
    DOCUMENTO_TEXTO.append("PASO 7: Predicción y Evaluación en el Conjunto de Prueba")
    DOCUMENTO_TEXTO.append("--------------------------------------------------------")

    if best_model and not test_data.empty:
        # Predecir sobre el conjunto de prueba
        start_index = len(train_data)
        end_index = len(ts_original) - 1
        predictions_test_set = best_model.predict(start=start_index, end=end_index, dynamic=False)
        predictions_test_set.index = test_data.index  # Asegurar que el índice es correcto

        # Calcular métricas de evaluación
        metricas_test = calcular_coeficiente_theil(test_data.values, predictions_test_set.values)
        DOCUMENTO_TEXTO.append("Métricas de rendimiento en el conjunto de prueba:")
        for metric, value in metricas_test.items():
            DOCUMENTO_TEXTO.append(f"  {metric}: {value:.4f}")
    else:
        DOCUMENTO_TEXTO.append("No se pudo realizar la predicción en el conjunto de prueba.")
        metricas_test = {}

    DOCUMENTO_TEXTO.append("\n")

    # PASO 8: Diagnóstico de Residuos
    DOCUMENTO_TEXTO.append("PASO 8: Diagnóstico de Residuos del Mejor Modelo")
    DOCUMENTO_TEXTO.append("-----------------------------------------------")
    if best_model:
        residuos = pd.Series(best_model.resid, index=train_data.index)
        DOCUMENTO_TEXTO.append(f"Análisis de los residuos del modelo ARIMA{best_order}:")
        DOCUMENTO_TEXTO.append(f"  Media de residuos: {residuos.mean():.4f}")
        DOCUMENTO_TEXTO.append(f"  Desviación estándar de residuos: {residuos.std():.4f}")
        jb_test = stats.jarque_bera(residuos.dropna())
        DOCUMENTO_TEXTO.append(
            f"  Test Jarque-Bera (Normalidad): Estadístico={jb_test[0]:.4f}, P-valor={jb_test[1]:.4f} "
            f"(p <= 0.05 indica no normalidad)")
        lb_test = best_model.get_prediction(start=0, end=len(train_data) - 1).summary_frame()['std_errors'].iloc[
            1:].autocorr(1)  # Un proxy simple
        # Ljung-Box test
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box_results = acorr_ljungbox(residuos.dropna(), lags=[1, 5, 10], return_df=True)
        DOCUMENTO_TEXTO.append("  Test de Ljung-Box (Autocorrelación de Residuos):")
        DOCUMENTO_TEXTO.append(f"{ljung_box_results.to_string()}")
        DOCUMENTO_TEXTO.append("  (p-valores bajos (<=0.05) sugieren autocorrelación en los residuos)")

        plot_diagnostico_residuos(residuos, best_order, ts_original.name)
        DOCUMENTO_TEXTO.append("Gráfico 08_Diagnostico_Residuos generado.")
    DOCUMENTO_TEXTO.append("\n")

    # PASO 9: Pronóstico Final
    DOCUMENTO_TEXTO.append("PASO 9: Pronóstico a Futuro")
    DOCUMENTO_TEXTO.append("---------------------------")
    forecast_steps = 12  # Pronóstico para 12 períodos futuros
    if best_model:
        forecast_obj = best_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int(alpha=0.05)
        forecast_ci_df = pd.DataFrame(forecast_ci, index=forecast_values.index, columns=['Lower CI', 'Upper CI'])

        DOCUMENTO_TEXTO.append(f"Pronóstico para los próximos {forecast_steps} períodos:")
        DOCUMENTO_TEXTO.append(f"{forecast_values.to_string()}")
        DOCUMENTO_TEXTO.append(f"Intervalo de Confianza (95%) del pronóstico:")
        DOCUMENTO_TEXTO.append(f"{forecast_ci_df.to_string()}")

        # Visualizar pronóstico
        plot_prediccion_final(ts_original, forecast_values, forecast_ci_df, best_order, metricas_test, best_model,
                              test_data, predictions_test_set)
        DOCUMENTO_TEXTO.append("Gráfico 09_Prediccion_Final generado.")
    DOCUMENTO_TEXTO.append("\n" + "=" * 80)
    DOCUMENTO_TEXTO.append(f"FIN DEL ANÁLISIS ARIMA PARA: {ts_original.name}")
    DOCUMENTO_TEXTO.append("=" * 80 + "\n")

    return best_order, metricas_test, best_model


# ============================================================================
# CÓDIGO PRINCIPAL DE EJECUCIÓN
# ============================================================================

def main():
    # Asegurarse de que el directorio de datos existe
    DATA_DIR.mkdir(exist_ok=True)
    # Crear archivos dummy si no existen para que el menú funcione sin errores de ruta.
    # En un caso real, estos archivos deberían contener datos válidos.
    if not (DATA_DIR / FILE_TS).exists():
        pd.DataFrame({'date': pd.date_range(start='2010-01-01', periods=100, freq='MS'),
                      'x': np.random.randn(100) * 10 + 50}).to_csv(DATA_DIR / FILE_TS, index=False)
    if not (DATA_DIR / FILE_VP).exists():
        pd.DataFrame({'dateid01': pd.date_range(start='2000-01-01', periods=80, freq='QS'),
                      'vtas': np.random.randn(80) * 100 + 1000,
                      'pub': np.random.randn(80) * 5 + 20}).to_csv(DATA_DIR / FILE_VP, index=False)
    if not (DATA_DIR / FILE_XLSX).exists():
        pd.DataFrame({'col1': [1, 2, 3]}).to_excel(DATA_DIR / FILE_XLSX, index=False)

    # Capturar toda la salida de la consola
    with ConsoleAndCapture(DOCUMENTO_TEXTO):
        print("Iniciando análisis de series temporales...")
        choice = solicitar_seleccion()
        ts_selected, freq_selected = cargar_datos_seleccionados(choice)

        if ts_selected is None:
            print("No se pudo cargar la serie temporal. Terminando el script.")
            return

        # Ajustar la frecuencia de la serie para statsmodels si es necesario (ej. 'MS' -> 'MS')
        # statsmodels requiere que el índice de tiempo tenga una frecuencia definida si se usa el parámetro `freq` en ARIMA
        if ts_selected.index.freq is None and freq_selected:
            try:
                ts_selected = ts_selected.asfreq(freq_selected)
            except ValueError:
                print(
                    f"No se pudo inferir la frecuencia '{freq_selected}' del índice. Continuando sin establecer la frecuencia explícita en el índice.")
        elif ts_selected.index.freq is None:
            print(
                "Advertencia: La frecuencia del índice de la serie no pudo ser inferida o definida. Esto podría afectar los modelos estacionales.")

        best_order, metricas_test, final_model_fit = box_jenkins_arima(ts_selected, freq_selected)

    # Generar el reporte PDF fuera del bloque de captura de la consola
    # para asegurar que todos los prints y errores se incluyan si ocurren después.
    if best_order:
        generar_reporte_pdf(DOCUMENTO_TEXTO, IMAGENES_REPORT, TS_NAME, best_order, metricas_test)
    else:
        sys.__stdout__.write("No se pudo generar el reporte PDF porque no se encontró un modelo ARIMA óptimo.\n")

    # Cerrar todos los gráficos de Matplotlib al finalizar
    plt.close('all')
    print("\nAnálisis completado. Todos los gráficos de Matplotlib se cerrarán ahora.")


if __name__ == "__main__":
    main()
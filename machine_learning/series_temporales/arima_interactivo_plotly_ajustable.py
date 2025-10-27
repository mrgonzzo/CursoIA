#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo de Series Temporales con ARIMA (Plotly)
IMPLEMENTA: Estructura Box-Jenkins, Carga Dinámica, Gráficos Plotly,
GENERACIÓN DE INFORME PDF, y AJUSTE INTERACTIVO DE RANGOS (p, d, q).
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

# Librerías de visualización y reportes
import warnings
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# Clase para capturar la salida de la consola
class ConsoleCapture:
    def __init__(self, document_list):
        self.document_list = document_list
        self.original_stdout = sys.stdout

    def __enter__(self):
        self.buffer = io.StringIO()
        sys.stdout = self.buffer
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        self.document_list.extend(self.buffer.getvalue().splitlines())


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

    return {'Theil_U': theil_u, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


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
            # USA input() para detener la ejecución y solicitar datos.
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
                # USA input() para detener la ejecución y solicitar datos.
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
# FUNCIONES DE PLOTLY (Gráficos Interactivos)
# ============================================================================
# *NOTA: Las funciones de Plotly se mantienen idénticas a la versión anterior.*

def guardar_grafico(fig, filename_base, title, show_plot=True):
    """Guarda el gráfico como PNG y HTML, lo añade al reporte y lo muestra."""
    if not title.startswith("PASO"):
        title = f"Gráfico - {title}"

    png_path = OUTPUT_DIR / f"{filename_base}.png"
    html_path = OUTPUT_DIR / f"{filename_base}.html"

    # print(f"\n[GRÁFICO] Guardando {title} como {png_path.name} (estático) y {html_path.name} (interactivo)...")

    try:
        # Intenta guardar el PNG (requiere kaleido)
        fig.write_image(str(png_path), scale=1)
        IMAGENES_REPORT.append({'title': title, 'path': png_path})
    except Exception as e:
        print(f"❌ Error al guardar PNG (¿Falta 'kaleido'?): {e}")

    fig.write_html(str(html_path))
    if show_plot:
        # fig.show() (Esta línea se mantiene comentada para evitar que se abran demasiadas ventanas/navegadores automáticamente)
        pass

def plot_serie_y_distribucion(ts):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Serie Temporal Original: {ts.name}', 'Distribución de la Serie'))
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Serie Original', line=dict(color='#2E86AB')),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=ts.values, name='Distribución', marker_color='#A23B72'), row=1, col=2)
    fig.update_layout(height=500, title_text="PASO 1: Análisis Descriptivo (Visualización Inicial)", showlegend=False)
    guardar_grafico(fig, "01_Serie_Distribucion", "Visualización y Distribución", show_plot=False)


def plot_descomposicion(ts, periodo):
    decomposition = seasonal_decompose(ts, model='additive', period=periodo)
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=('Serie Original', 'Tendencia', 'Componente Estacional', 'Residuos'),
                        shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=ts.index, y=decomposition.observed, mode='lines', name='Observado', line=dict(color='#2E86AB')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=ts.index, y=decomposition.trend, mode='lines', name='Tendencia', line=dict(color='#F18F01')),
        row=2, col=1)
    fig.add_trace(
        go.Scatter(x=ts.index, y=decomposition.seasonal, mode='lines', name='Estacional', line=dict(color='#C73E1D')),
        row=3, col=1)
    fig.add_trace(
        go.Scatter(x=ts.index, y=decomposition.resid, mode='lines', name='Residuos', line=dict(color='#6A994E')), row=4,
        col=1)
    fig.update_layout(height=800, title_text=f"PASO 2: Descomposición Estacional Aditiva (Período={periodo})",
                      showlegend=False)
    guardar_grafico(fig, "02_Descomposicion", "Descomposición Clásica", show_plot=False)


def plot_transformaciones(transformaciones):
    valid_trans = {k: v for k, v in transformaciones.items() if not v.empty and len(v.dropna()) > 10}
    n_plots = len(valid_trans)
    if n_plots == 0: return

    rows = (n_plots + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=list(valid_trans.keys()), shared_xaxes=False)

    for idx, (nombre, serie_trans) in enumerate(valid_trans.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        fig.add_trace(
            go.Scatter(x=serie_trans.index, y=serie_trans.values, mode='lines', name=nombre, line=dict(width=1.5)),
            row=row, col=col)

    fig.update_layout(height=400 * rows, title_text=f"PASO 4: Visualización de Transformaciones para Estacionariedad",
                      showlegend=False)
    guardar_grafico(fig, "04_Transformaciones", "Visualización de Transformaciones", show_plot=False)


def plot_acf_pacf(serie, nombre, lags):
    acf_values = acf(serie.dropna(), nlags=lags, fft=False)[1:]
    pacf_values = pacf(serie.dropna(), nlags=lags)[1:]
    conf_limit = 1.96 / np.sqrt(len(serie.dropna()))
    lags_index = np.arange(1, lags + 1)

    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'ACF - {nombre} (sin lag 0)', f'PACF - {nombre} (sin lag 0)'))

    fig.add_trace(go.Bar(x=lags_index, y=acf_values, name='ACF', marker_color='#2E86AB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lags_index, y=[conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=lags_index, y=[-conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=lags_index, y=pacf_values, name='PACF', marker_color='#C73E1D'), row=2, col=1)
    fig.add_trace(go.Scatter(x=lags_index, y=[conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=lags_index, y=[-conf_limit] * lags, mode='lines', name='IC 95%',
                             line=dict(dash='dash', color='red', width=1), showlegend=False), row=2, col=1)

    fig.update_layout(height=600, title_text="PASO 5: Autocorrelación y Autocorrelación Parcial (Identificación p, q)",
                      showlegend=False)
    fig.update_xaxes(title_text="Rezagos", row=2, col=1)
    guardar_grafico(fig, "05_ACF_PACF", "ACF y PACF para Identificación", show_plot=False)


def plot_diagnostico_residuos(residuos, orden, ts_name):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Serie de Residuos', 'Distribución de Residuos', 'ACF de Residuos (sin lag 0)',
                                        'Q-Q Plot'))

    fig.add_trace(
        go.Scatter(x=residuos.index, y=residuos.values, mode='lines', name='Residuos', line=dict(color='#2E86AB')),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=residuos.index, y=[0] * len(residuos), mode='lines', name='Cero',
                             line=dict(color='red', dash='dash')), row=1, col=1)

    fig.add_trace(go.Histogram(x=residuos.values, name='Distribución', marker_color='#A23B72'), row=1, col=2)

    acf_values = acf(residuos, nlags=30, fft=False)[1:]
    conf_limit = 1.96 / np.sqrt(len(residuos))
    lags_index = np.arange(1, 31)
    fig.add_trace(go.Bar(x=lags_index, y=acf_values, name='ACF Residuos', marker_color='#F18F01'), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=lags_index, y=[conf_limit] * 30, mode='lines', line=dict(dash='dash', color='red', width=1),
                   showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=lags_index, y=[-conf_limit] * 30, mode='lines', line=dict(dash='dash', color='red', width=1),
                   showlegend=False), row=2, col=1)

    qq_data = stats.probplot(residuos, dist="norm", fit=True)
    fig.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Puntos', marker=dict(color='#C73E1D')),
        row=2, col=2)
    fig.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1], mode='lines', name='Línea Teórica',
                   line=dict(color='gray', dash='dash')), row=2, col=2)

    fig.update_layout(height=700, title_text=f"PASO 8: Diagnóstico de Residuos - ARIMA{orden} ({ts_name})",
                      showlegend=False)
    guardar_grafico(fig, "08_Diagnostico_Residuos", "Diagnóstico de Residuos", show_plot=False)


def plot_prediccion_final(ts, forecast, forecast_ci, mejor_orden, metricas_test, mejor_modelo):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', name='IC Sup 95%',
                             line=dict(width=0), fillcolor='rgba(199, 62, 29, 0.2)', fill='tonexty'))
    fig.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', name='IC Inf 95%',
                             line=dict(width=0), fill='tonexty', showlegend=False))

    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Serie Histórica',
                             line=dict(color='#2E86AB', width=2)))

    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', name=f'Predicción ARIMA{mejor_orden}',
                   line=dict(color='#C73E1D', width=2.5, dash='dash'), marker=dict(size=5, symbol='square')))

    fig.add_vline(x=ts.index[-1], line_width=2, line_dash="dash", line_color="gray",
                  annotation_text="Inicio de Predicción", annotation_position="top left")

    text_metrics = (f"AIC: {mejor_modelo.aic:.2f} | BIC: {mejor_modelo.bic:.2f}<br>"
                    f"Theil U: {metricas_test['Theil_U']:.4f} | RMSE: {metricas_test['RMSE']:.2f}")

    fig.update_layout(title=f'PASO 10: Predicción a Futuro - ARIMA{mejor_orden} ({ts.name})', title_x=0.5, height=650,
                      xaxis_title="Fecha", yaxis_title="Valor de la Serie", hovermode="x unified",
                      annotations=[
                          dict(x=0.01, y=0.99, xref="paper", yref="paper", text=text_metrics, showarrow=False,
                               bordercolor="black", borderwidth=1, borderpad=4, bgcolor="rgba(255, 255, 255, 0.8)",
                               align="left")
                      ]
                      )
    guardar_grafico(fig, "10_Prediccion_Final", "Predicción y Intervalos de Confianza", show_plot=False)


# ============================================================================
# FUNCIÓN DE GENERACIÓN DE PDF
# ============================================================================

def generar_informe_pdf(pdf_filename):
    """Crea el informe PDF a partir del texto capturado y las imágenes guardadas."""
    try:
        pdf = FPDF(unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"INFORME DE ANÁLISIS ARIMA: {TS_NAME}", 0, 1, "C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        pdf.ln(5)

        current_image_index = 0

        for line in DOCUMENTO_TEXTO:
            if line.startswith('---'): continue

            if line.startswith('PASO') or line.startswith('--- FASE'):
                pdf.ln(3)
                pdf.set_font("Arial", "B", 12)
                pdf.multi_cell(0, 5, line, border='B')
                pdf.ln(1)
                pdf.set_font("Arial", "", 10)

            elif line.startswith('[GRÁFICO]') and current_image_index < len(IMAGENES_REPORT):
                pdf.ln(2)
                img_data = IMAGENES_REPORT[current_image_index]

                pdf.set_font("Arial", "I", 10)
                pdf.multi_cell(0, 5, f"Figura {current_image_index + 1}: {img_data['title']}", 0, 1, "C")
                pdf.ln(1)

                img_width = 180
                try:
                    pdf.image(img_data['path'], x=pdf.get_x() + 10, w=img_width)
                    pdf.ln(img_width * 0.75 + 5)
                except Exception:
                    pdf.set_font("Arial", "B", 10)
                    pdf.multi_cell(0, 5, f"ERROR al insertar imagen {img_data['path'].name}. Asegúrese de tener 'kaleido' instalado para la generación de PNG.")
                    pdf.set_font("Arial", "", 10)

                current_image_index += 1
                pdf.ln(5)

            elif line.strip().startswith('coef') or line.strip().startswith('const') or re.match(
                    r'^(L\d\.\w\d|\s*\bconst\b)', line):
                pdf.set_font("Courier", "", 8)
                pdf.multi_cell(0, 3, line)

            else:
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 5, line)

        pdf.output(pdf_filename)
        return True

    except Exception as e:
        # Se imprime directamente a la consola real (sys.stdout original)
        sys.stdout = sys.__stdout__
        print(f"\n❌ ERROR CRÍTICO al generar el PDF: {e}")
        return False


# ============================================================================
# ANÁLISIS PRINCIPAL
# ============================================================================

def ejecutar_analisis_arima():

    # 1. SELECCIÓN Y CARGA DE DATOS (INTERACTIVO)
    choice = solicitar_seleccion()
    ts, freq = cargar_datos_seleccionados(choice)

    if ts is None:
        print("\nAnálisis cancelado.")
        return

    # 2. AJUSTE INTERACTIVO DE RANGOS
    p_range, d_range, q_range = solicitar_rangos_arima()

    # Iniciar la captura de la salida de la consola (a partir de aquí, la salida va al informe)
    with ConsoleCapture(DOCUMENTO_TEXTO):

        # --- CONFIGURACIÓN DE PARÁMETROS INTERNOS ---
        if freq == 'QS':
            periodo_estacional = 4
            n_forecast = 8
        else:
            periodo_estacional = 12
            n_forecast = 24

        n = len(ts)

        print("\n" + "=" * 80)
        print(f"INICIO DEL PROCESO DE ANÁLISIS AUTOMATIZADO ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"Carpeta de Resultados: {OUTPUT_DIR.absolute()}")
        print(f"\nANÁLISIS DE LA SERIE: {ts.name}")
        print(f"Frecuencia: {freq}, Observaciones: {n}, Período Estacional: {periodo_estacional}")
        print(f"RANGOS DE GRID SEARCH: p:{list(p_range)}, d:{list(d_range)}, q:{list(q_range)}")
        print(f"{'=' * 80}")

        # ----------------------------------------------------------------------------
        # FASE 1: ESTUDIO DESCRIPTIVO Y DESCOMPOSICIÓN
        # ----------------------------------------------------------------------------
        print("\n--- FASE 1: ESTUDIO DESCRIPTIVO Y DESCOMPOSICIÓN ---")

        print("PASO 1: Visualización inicial (Gráficos Plotly)...")
        plot_serie_y_distribucion(ts)
        print(
            "EXPLICACIÓN: Este gráfico identifica si la serie tiene tendencia clara y permite inspeccionar la forma de su distribución, buscando posibles transformaciones (ej. logaritmo si la varianza crece con la media).")

        print("\nPASO 2: Descomposición estacional (Gráfico Plotly)...")
        plot_descomposicion(ts, periodo_estacional)
        print(
            "EXPLICACIÓN: La descomposición aditiva separa la serie en Tendencia, Estacionalidad y Residuos. Esto confirma visualmente la necesidad de diferenciación (para la tendencia) y de componentes SARIMA (para la estacionalidad).")

        # ----------------------------------------------------------------------------
        # FASE 2: ESTACIONARIEDAD E IDENTIFICACIÓN (d, p, q)
        # ----------------------------------------------------------------------------
        print("\n--- FASE 2: ESTACIONARIEDAD E IDENTIFICACIÓN (d, p, q) ---")

        print("PASO 3: Test ADF (Dickey-Fuller Aumentado)")
        adf_original = test_adf(ts, "Serie Original")
        print(
            f"  ADF Original (p-valor): {adf_original['pvalor']:.4f} -> {'ESTACIONARIA' if adf_original['es_estacionaria'] else 'NO ESTACIONARIA (Requiere diferenciación: d>0)'}")
        print(
            "EXPLICACIÓN: Se usa el Test ADF para determinar formalmente si la serie tiene raíz unitaria (No Estacionaria), lo cual define el parámetro 'd' del ARIMA. Si p-valor > 0.05, no se rechaza la hipótesis nula de No Estacionariedad.")

        # Generar transformaciones para el análisis
        ts_diff1 = ts.diff().dropna()
        ts_diff_s = ts.diff(periodo_estacional).dropna()
        ts_diff1_diff_s = ts_diff1.diff(periodo_estacional).dropna()

        transformaciones = {
            'Diferencia Regular (d=1)': ts_diff1,
            f'Diferencia Estacional (s={periodo_estacional})': ts_diff_s,
            f'Dif. Reg. + Est.': ts_diff1_diff_s
        }

        print("\nPASO 4: Visualización de transformaciones clave (Gráfico Plotly)...")
        plot_transformaciones(transformaciones)
        print(
            "EXPLICACIÓN: Se inspeccionan las series diferenciadas. Generalmente, la combinación de Diferencia Regular y Estacional es necesaria para lograr la estacionariedad completa en series económicas (d=1, D=1).")

        print(f"\nPASO 5: ACF y PACF (Identificación p, q) sobre serie 'Dif. Reg. + Est.'...")
        adf_diff = test_adf(ts_diff1_diff_s, "Diferencia Estacionaria Final")
        print(
            f"  ADF Final (p-valor): {adf_diff['pvalor']:.4f} -> {'ESTACIONARIA' if adf_diff['es_estacionaria'] else 'AÚN NO ESTACIONARIA'}")

        plot_acf_pacf(ts_diff1_diff_s, f"Diferencia Regular + Estacional (d=1, D=1, s={periodo_estacional})",
                      lags=periodo_estacional * 2)
        print(
            "EXPLICACIÓN: Los gráficos de Autocorrelación (ACF) y Autocorrelación Parcial (PACF) de la serie estacionaria (o asumida como tal) se usan para 'identificar' los órdenes p (AR) y q (MA). Los picos significativos más allá de los intervalos de confianza sugieren valores iniciales para p y q.")

        # ----------------------------------------------------------------------------
        # FASE 3: ESTIMACIÓN Y SELECCIÓN DEL MODELO
        # ----------------------------------------------------------------------------
        print("\n--- FASE 3: ESTIMACIÓN Y SELECCIÓN DEL MODELO ---")
        print(
            f"PASO 6: Estimación de Múltiples Modelos ARIMA (Grid Search con rangos p:{list(p_range)}, d:{list(d_range)}, q:{list(q_range)})...")

        train_size = int(len(ts) * 0.85)
        train, test = ts[:train_size], ts[train_size:]
        resultados_modelos = []

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        # Nota: Aquí se está probando un ARIMA no estacional. Un SARIMA completo requiere más complejidad en el Grid Search.
                        modelo = ARIMA(train, order=(p, d, q), trend='c' if d == 0 else 'n',
                                       enforce_stationarity=False, enforce_invertibility=False)
                        modelo_fit = modelo.fit(disp=False)

                        resultados_modelos.append({
                            'orden': (p, d, q),
                            'AIC': modelo_fit.aic,
                            'BIC': modelo_fit.bic,
                            'modelo': modelo_fit
                        })
                    except:
                        # Ignorar modelos que no convergen o fallan la estimación
                        continue

        if not resultados_modelos:
            print("\nError: No se pudo ajustar ningún modelo con los rangos seleccionados. Análisis detenido.")
            return

        df_resultados = pd.DataFrame([
            {'Orden': f"({r['orden'][0]},{r['orden'][1]},{r['orden'][2]})", 'AIC': r['AIC'], 'BIC': r['BIC'], 'idx': i}
            for i, r in enumerate(resultados_modelos)
        ]).sort_values('AIC').reset_index(drop=True)

        print("\nTop 5 modelos por AIC:")
        print(df_resultados.head(5)[['Orden', 'AIC', 'BIC']].to_string(index=False))
        print(
            "EXPLICACIÓN: Se utiliza el Criterio de Información de Akaike (AIC) y el Criterio de Información Bayesiano (BIC) para la selección. Estos criterios penalizan la complejidad del modelo (más parámetros) y favorecen el mejor ajuste. El modelo con el AIC más bajo es seleccionado por defecto.")

        idx_mejor = df_resultados.iloc[0]['idx']
        mejor_resultado = resultados_modelos[idx_mejor]
        mejor_orden = mejor_resultado['orden']
        mejor_modelo = mejor_resultado['modelo']

        print("\nPASO 7: Mejor Modelo Seleccionado")
        print(f"  Mejor Modelo: ARIMA{mejor_orden}")
        print(f"  AIC: {mejor_resultado['AIC']:.4f} | BIC: {mejor_resultado['BIC']:.4f}")
        print("\nRESUMEN ESTADÍSTICO DEL MEJOR MODELO AJUSTADO:")
        print(mejor_modelo.summary().as_text())

        # ----------------------------------------------------------------------------
        # FASE 4: DIAGNOSIS Y VALIDACIÓN
        # ----------------------------------------------------------------------------
        print("\n--- FASE 4: DIAGNOSIS Y VALIDACIÓN ---")
        print("PASO 8: Diagnóstico de Residuos (Gráficos Plotly)...")
        plot_diagnostico_residuos(mejor_modelo.resid, mejor_orden, ts.name)
        print(
            "EXPLICACIÓN: Se evalúa si los residuos (errores) del modelo son Ruido Blanco. Los residuos deben ser incorrelacionados (ACF cero), tener media cero y seguir una distribución normal (Q-Q Plot). Un buen diagnóstico confirma la validez del modelo ARIMA.")

        print("\nPASO 9: Evaluación y Coeficiente de Theil (U) en Test Set...")
        predicciones_test = mejor_modelo.forecast(steps=len(test))
        metricas_test = calcular_coeficiente_theil(test.values, predicciones_test.values)

        print(f"  RMSE (Error Cuadrático Medio): {metricas_test['RMSE']:.4f}")
        print(f"  MAPE (Error Absoluto Porcentual Medio): {metricas_test['MAPE']:.2f}%")
        print(f"  Coeficiente de Theil (U): {metricas_test['Theil_U']:.6f}")
        print(
            "EXPLICACIÓN: El Coeficiente de Theil (U) evalúa la precisión de la predicción. Valores de U cercanos a cero (< 1) indican que el modelo es mejor que una simple predicción 'naive' (cambio sin cambio). El RMSE y el MAPE miden la magnitud del error.")

        # Reajustar con todos los datos para la predicción final
        modelo_final = ARIMA(ts, order=mejor_orden, trend='c' if mejor_orden[1] == 0 else 'n',
                             enforce_stationarity=False, enforce_invertibility=False)
        modelo_final_fit = modelo_final.fit()

        # ----------------------------------------------------------------------------
        # FASE 5: PREDICCIÓN
        # ----------------------------------------------------------------------------
        print("\n--- FASE 5: PREDICCIÓN ---")
        print(f"PASO 10: Predicción para {n_forecast} períodos (Gráfico Plotly)...")

        forecast_result = modelo_final_fit.get_forecast(steps=n_forecast)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=0.05)

        offset_params = {'months': 1} if freq == 'MS' else {'quarters': 1}
        forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(**offset_params),
                                       periods=n_forecast, freq=freq)

        plot_prediccion_final(ts, forecast, forecast_ci, mejor_orden, metricas_test, mejor_modelo)

        print("\nRESULTADO FINAL: PREDICCIONES FUTURAS CON INTERVALOS DE CONFIANZA:")
        for fecha, valor, ic_inf, ic_sup in zip(forecast_index, forecast, forecast_ci.iloc[:, 0],
                                                forecast_ci.iloc[:, 1]):
            fecha_str = fecha.strftime('%Y-%m') if freq != 'QS' else fecha.strftime('%Y-Q%q')
            print(f"  {fecha_str}: {valor:8.2f}  [IC 95%: {ic_inf:8.2f} - {ic_sup:8.2f}]")

        print("\n" + "=" * 80)
        print("✅ ANÁLISIS COMPLETO Y REPORTE GENERADO")
        print("=" * 80)

    # ----------------------------------------------------------------------------
    # GENERAR PDF FUERA DE LA CAPTURA DE CONSOLA
    # ----------------------------------------------------------------------------
    sys.stdout = sys.__stdout__  # Asegurar que esta parte se muestre en consola
    pdf_filename = OUTPUT_DIR / f'INFORME_ARIMA_{TS_NAME.replace(" ", "_")}.pdf'
    if generar_informe_pdf(str(pdf_filename)):
        print(f"\n[FINAL] Informe PDF generado exitosamente en: {pdf_filename.absolute()}")
    else:
        print("\n[FINAL] La generación del PDF falló. Asegúrate de que 'fpdf2' y 'kaleido' estén instalados.")

    print("\n[FINAL] Los gráficos interactivos (.html) también están disponibles en la carpeta arima_interact.")


if __name__ == '__main__':
    ejecutar_analisis_arima()
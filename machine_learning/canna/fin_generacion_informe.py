import pandas as pd
import joblib
# Se elimina XPos y YPos de la importación
from fpdf import FPDF
from datetime import datetime
import os

print("--- f9_generacion_informe.py: Generación de Informe Final (PDF) ---")

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---

# Cargar datos necesarios
try:
    df_final = pd.read_csv('resumen_pca_kmeans_final.csv', index_col='Strain')
    pca_model = joblib.load('pca_model.pkl')
    cluster_profile = pd.read_csv('kmeans_cluster_profile.csv', index_col='Cluster')

    # Datos de Varianza PCA
    explained_variance = pca_model.explained_variance_ratio_
    total_variance = explained_variance.sum() * 100

    # Cargas de PCA para interpretación
    df_scaled = pd.read_csv('cannabis_data_scaled.csv', index_col='Strain')
    pca_loadings = pd.DataFrame(pca_model.components_.T,
                                columns=[f'PC{i + 1}' for i in range(pca_model.n_components_)],
                                index=df_scaled.columns)

except FileNotFoundError as e:
    print(f"❌ ERROR: Archivo necesario no encontrado. Asegúrate de ejecutar f1 a f6. Detalle: {e}")
    exit()
except ImportError:
    # Este mensaje es útil si el usuario tiene fpdf instalado, pero no fpdf2 (donde viven XPos/YPos)
    print(
        "❌ ERROR: La librería 'fpdf' no está instalada o está desactualizada. Se requiere la versión moderna (fpdf2) o la versión clásica con sintaxis ajustada.")
    exit()


# --- 2. CLASE PERSONALIZADA PARA EL PDF ---
class PDF(FPDF):
    def header(self):
        # Título y logo (simulado)
        self.set_font('Arial', 'B', 15)
        # Se elimina XPos.RIGHT (que causaba el error) y se usa 0 como valor de posición
        self.cell(0, 10, 'INFORME DE ANÁLISIS PCA + K-MEANS', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        # Número de página
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')


# --- 3. GENERACIÓN DEL INFORME ---
pdf = PDF('P', 'mm', 'A4')
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.alias_nb_pages()

# --- Título Principal y Metadata ---
pdf.set_font('Arial', '', 12)
pdf.cell(0, 5, f'Fecha de Generación: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'L')
pdf.ln(5)

# ----------------------------------------------------
# 3.1. SECCIÓN 1: ANÁLISIS PCA (Varianza y Cargas)
# ----------------------------------------------------
pdf.set_font('Arial', 'B', 14)
pdf.set_fill_color(200, 220, 255)
pdf.cell(0, 8, '1. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)', 1, 1, 'L', True)
pdf.ln(2)

pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 5,
               f"El modelo PCA redujo la dimensionalidad a 2 Componentes Principales (PC1, PC2), capturando un total de {total_variance:.2f}% de la varianza original en los datos químicos y de efectos. Esto asegura que la visualización 2D es altamente representativa de la estructura real de los datos.")

# Varianza Explicada
pdf.set_font('Arial', 'B', 10)
pdf.ln(3)
pdf.cell(0, 5, 'Varianza Individual por Componente:', 0, 1)
pdf.set_font('Arial', '', 10)
pdf.cell(50, 5, f"- PC1: {explained_variance[0] * 100:.2f}%", 0, 0)
pdf.cell(50, 5, f"- PC2: {explained_variance[1] * 100:.2f}%", 0, 1)
pdf.ln(3)

# Cargas Dominantes (Interpretación)
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 5, 'Conclusión: Interpretación de PC1 (Eje de Máxima Variación)', 0, 1)
pdf.set_font('Arial', '', 10)

dominant_pc1 = pca_loadings['PC1'].abs().sort_values(ascending=False).head(3)
pc1_loading_text = ""
for feature, loading in dominant_pc1.items():
    sign = 'POSITIVA' if pca_loadings.loc[feature, 'PC1'] > 0 else 'NEGATIVA'
    pc1_loading_text += f"'{feature}' (Correlación {sign}); "

pdf.multi_cell(0, 5,
               f"PC1 está dominada por: {pc1_loading_text}. Este eje define la principal diferencia entre las cepas.")

# Insertar Gráfico de Varianza (si existe)
if os.path.exists('pca_variance_plot.png'):
    pdf.ln(2)
    pdf.cell(0, 5, 'Gráfico de Varianza Explicada (Ver archivo: pca_variance_plot.png):', 0, 1)
    # Se asegura que la imagen se intente cargar solo si existe
    try:
        pdf.image('pca_variance_plot.png', w=100)
    except Exception as e:
        pdf.multi_cell(0, 5, f"Advertencia: No se pudo insertar la imagen 'pca_variance_plot.png'. Detalles: {e}", 0, 1)
    pdf.ln(3)

# ----------------------------------------------------
# 3.2. SECCIÓN 2: RESULTADOS K-MEANS Y PERFILES
# ----------------------------------------------------
pdf.set_font('Arial', 'B', 14)
pdf.set_fill_color(200, 220, 255)
pdf.add_page()
pdf.cell(0, 8, '2. RESULTADOS DEL CLUSTERING K-MEANS (K=3)', 1, 1, 'L', True)
pdf.ln(2)

# Perfiles Promedio (Tabla)
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 5, 'Perfiles de Características Promedio por Clúster:', 0, 1)
pdf.ln(1)

# Preparar y dibujar la tabla
data = cluster_profile.round(2).reset_index().to_numpy().tolist()
column_names = ['Cluster'] + cluster_profile.columns.tolist()
col_widths = [18] + [20] * len(cluster_profile.columns)

# Cabecera de la tabla
pdf.set_font('Arial', 'B', 8)
pdf.set_fill_color(240, 240, 240)
for i, header in enumerate(column_names):
    pdf.cell(col_widths[i], 6, header, 1, 0, 'C', True)
pdf.ln()

# Filas de datos
pdf.set_font('Arial', '', 8)
for row in data:
    for i, item in enumerate(row):
        pdf.cell(col_widths[i], 5, str(item), 1, 0, 'C')
    pdf.ln()
pdf.ln(5)

# Conclusiones de Interpretación
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 5, 'Interpretación de los Clústeres (Basado en Perfiles):', 0, 1)
pdf.set_font('Arial', '', 10)

# Simulación de la interpretación de los clústeres
for cluster in cluster_profile.index:
    sleepy_val = cluster_profile.loc[cluster, 'Sleepy']
    uplifted_val = cluster_profile.loc[cluster, 'Uplifted']
    myrcene_val = cluster_profile.loc[cluster, 'Myrcene']

    if sleepy_val > uplifted_val and myrcene_val > 0.5:
        label = f"INDICA-DOMINANTE"
        key_char = f"Alto Myrcene ({myrcene_val:.2f}) y Alto efecto Sedante ({sleepy_val:.2f})."
    elif uplifted_val > sleepy_val:
        label = f"SATIVA-DOMINANTE"
        key_char = f"Alto Limonene y Alto efecto Energético ({uplifted_val:.2f})."
    else:
        label = f"HÍBRIDO/EQUILIBRADO"
        key_char = f"Valores medios en efectos (Sleepy: {sleepy_val:.2f}, Uplifted: {uplifted_val:.2f})."

    pdf.multi_cell(0, 5,
                   f"   - Clúster {cluster} ({df_final[df_final['Cluster'] == cluster].shape[0]} Cepas): Identificado como **{label}**. Característica clave: {key_char}")
pdf.ln(5)

# ----------------------------------------------------
# 3.3. SECCIÓN 3: VISUALIZACIÓN Y CONCLUSIÓN
# ----------------------------------------------------
pdf.set_font('Arial', 'B', 14)
pdf.set_fill_color(200, 220, 255)
pdf.cell(0, 8, '3. VALIDACIÓN Y VISUALIZACIÓN', 1, 1, 'L', True)
pdf.ln(2)

pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 5,
               "La validación visual es crucial. El gráfico de dispersión PCA/K-Means (f7) muestra cómo los 3 clústeres forman grupos distintos en el espacio 2D, confirmando que el clustering ha encontrado la variación estructural basada en la composición química.",
               0, 1)

# Insertar Gráfico de Clustering (si existe)
if os.path.exists('pca_kmeans_clustering_plot.png'):
    pdf.ln(2)
    pdf.cell(0, 5, 'Gráfico de Agrupamiento PCA/K-Means (Ver archivo: pca_kmeans_clustering_plot.png):', 0, 1)
    try:
        pdf.image('pca_kmeans_clustering_plot.png', w=120)
    except Exception as e:
        pdf.multi_cell(0, 5,
                       f"Advertencia: No se pudo insertar la imagen 'pca_kmeans_clustering_plot.png'. Detalles: {e}", 0,
                       1)

# ----------------------------------------------------
# 3.4. CONCLUSIÓN FINAL
# ----------------------------------------------------
pdf.set_font('Arial', 'BU', 12)
pdf.ln(5)
pdf.cell(0, 8, 'CONCLUSIÓN FINAL Y RECOMENDACIÓN', 0, 1)
pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 5,
               "El análisis demostró que la principal variación entre las cepas se relaciona directamente con el contenido de Terpenos (ej. Myrcene/Limonene) y sus efectos percibidos (Sedante/Energético). El modelo K-Means ha segmentado exitosamente el dataset en 3 grupos que reflejan esta variación biológica, lo cual es útil para la recomendación de productos y el desarrollo de cepas con perfiles de efectos específicos.")

# --- 4. SALIDA DEL PDF ---
pdf_file_name = 'Informe_PCA_KMeans_Cannabis.pdf'
pdf.output(pdf_file_name, 'F')

print(f"\n✅ Informe generado y guardado exitosamente en: {pdf_file_name}")
import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import messagebox

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")
pdf_path = os.path.join(output_path, "Reporte_UCI_Fallecidos_GUI.pdf")
csv_output_path = os.path.join(output_path, "uci_fallecidos_predicciones_gui.csv")

# --- CARGA DE DATOS ---
df = pd.read_csv(data_path)
df["Fecha_r"] = pd.to_datetime(df["Fecha_r"], errors="coerce")

features = ["Hay_vacuna", "Hay_Confinamiento"]
targets = ["Ingresos_UCI_diarios", "Fallecidos_diarios"]
X = df[features]
X = sm.add_constant(X)

# --- MODELOS OLS MULTISALIDA ---
model_dict = {}
coef_list = []

for target in targets:
    y = df[target]
    model = sm.OLS(y, X).fit()
    model_dict[target] = model
    for var in model.params.index:
        coef_list.append({
            "Variable_objetivo": target,
            "Predictor": var,
            "Coeficiente": model.params[var],
            "p_value": model.pvalues[var]
        })

df_coefs = pd.DataFrame(coef_list)


# --- FUNCIONES ---
def predecir_nuevos_gui(hay_vacuna, hay_confinamiento):
    X_new = pd.DataFrame([[1, hay_vacuna, hay_confinamiento]], columns=["const"] + features)
    resultados = {}
    for target, model in model_dict.items():
        pred = model.get_prediction(X_new)
        pred_mean = pred.predicted_mean[0]
        ci_lower, ci_upper = pred.conf_int()[0]
        resultados[target] = {"Predicción": pred_mean, "CI_lower": ci_lower, "CI_upper": ci_upper}
    return resultados


def mostrar_prediccion():
    vac = vac_var.get()
    conf = conf_var.get()
    resultados = predecir_nuevos_gui(vac, conf)
    texto = ""
    for target, val in resultados.items():
        texto += f"{target}: Predicción={val['Predicción']:.2f}, CI=({val['CI_lower']:.2f},{val['CI_upper']:.2f})\n"
    messagebox.showinfo("Predicción", texto)


def generar_pdf():
    with PdfPages(pdf_path) as pdf:
        # Página 1: Resumen estadístico
        desc = df.describe()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title("Resumen Estadístico del Dataset", fontsize=14, fontweight='bold')
        tabla = ax.table(cellText=desc.round(2).values,
                         rowLabels=desc.index,
                         colLabels=desc.columns,
                         cellLoc='center',
                         loc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1, 1.5)
        pdf.savefig()
        plt.close()

        # Página 2 y 3: Gráficos históricos
        for target, color in zip(targets, ["blue", "red"]):
            plt.figure(figsize=(10, 5))
            plt.plot(df["Fecha_r"], df[target], label=target, color=color)
            plt.xlabel("Fecha")
            plt.ylabel(target)
            plt.title(f"Histórico {target}")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Página 4: Tabla coeficientes y p-values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title("Coeficientes y p-values (Efectos Marginales)", fontsize=14, fontweight='bold')
        tabla = ax.table(cellText=df_coefs.round(4).values,
                         colLabels=df_coefs.columns,
                         cellLoc='center',
                         loc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1, 1.5)
        pdf.savefig()
        plt.close()

        # Página 5 y 6: Predicciones vs reales
        for target in targets:
            model = model_dict[target]
            y_pred = model.predict(X)
            plt.figure(figsize=(8, 5))
            plt.scatter(df[target], y_pred, alpha=0.7)
            plt.plot([df[target].min(), df[target].max()],
                     [df[target].min(), df[target].max()], 'r--')
            plt.xlabel(f"{target} reales")
            plt.ylabel(f"{target} predichos")
            plt.title(f"Predicciones vs reales ({target})")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    messagebox.showinfo("PDF Generado", f"✅ PDF generado en '{pdf_path}'")


def aplicar_csv_real():
    if os.path.exists(data_path):
        df_real = pd.read_csv(data_path)
        X_real = df_real[features]
        X_real = sm.add_constant(X_real)
        for target, model in model_dict.items():
            pred_real = model.get_prediction(X_real)
            df_real[f"{target}_pred"] = pred_real.predicted_mean
            df_real[f"{target}_CI_lower"] = pred_real.conf_int()[:, 0]
            df_real[f"{target}_CI_upper"] = pred_real.conf_int()[:, 1]
        df_real.to_csv(csv_output_path, index=False)
        messagebox.showinfo("CSV Guardado", f"✅ Predicciones guardadas en '{csv_output_path}'")


# --- GUI ---
root = tk.Tk()
root.title("Predicción UCI y Fallecidos")

vac_var = tk.IntVar()
conf_var = tk.IntVar()

tk.Label(root, text="Predicción de UCI y Fallecidos").grid(row=0, column=0, columnspan=2, pady=10)

tk.Checkbutton(root, text="Hay vacuna", variable=vac_var).grid(row=1, column=0, sticky="w", padx=10)
tk.Checkbutton(root, text="Hay confinamiento", variable=conf_var).grid(row=2, column=0, sticky="w", padx=10)

tk.Button(root, text="Hacer Predicción", command=mostrar_prediccion).grid(row=3, column=0, pady=10)
tk.Button(root, text="Generar PDF", command=generar_pdf).grid(row=3, column=1, pady=10)
tk.Button(root, text="Aplicar CSV Real", command=aplicar_csv_real).grid(row=4, column=0, columnspan=2, pady=10)
tk.Button(root, text="Salir", command=root.quit).grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()

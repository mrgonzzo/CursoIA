import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")
pdf_path = os.path.join(output_path, "Reporte_UCI_Fallecidos_Dashboard.pdf")
csv_output_path = os.path.join(output_path, "uci_fallecidos_predicciones_dashboard.csv")

# --- CARGA DE DATOS ---
df = pd.read_csv(data_path)
df["Fecha_r"] = pd.to_datetime(df["Fecha_r"], errors="coerce")

features = ["Hay_vacuna", "Hay_Confinamiento"]
targets = ["Ingresos_UCI_diarios", "Fallecidos_diarios"]
X = df[features]
X = sm.add_constant(X)

# --- MODELOS ---
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


# --- FUNCIONES DE PREDICCIÓN ---
def predecir_nuevos_gui(hay_vacuna, hay_confinamiento):
    X_new = pd.DataFrame([[1, hay_vacuna, hay_confinamiento]], columns=["const"] + features)
    resultados = {}
    for target, model in model_dict.items():
        pred = model.get_prediction(X_new)
        pred_mean = pred.predicted_mean[0]
        ci_lower, ci_upper = pred.conf_int()[0]
        resultados[target] = {"Predicción": pred_mean, "CI_lower": ci_lower, "CI_upper": ci_upper}
    return resultados


# --- FUNCIONES DE GRAFICOS DINÁMICOS ---
def actualizar_graficos():
    vac = vac_var.get()
    conf = conf_var.get()
    resultados = predecir_nuevos_gui(vac, conf)

    for i, target in enumerate(targets):
        ax[i].cla()
        # Datos históricos
        ax[i].plot(df["Fecha_r"], df[target], label="Datos históricos", color="blue" if i == 0 else "red")
        # Predicción nueva
        ax[i].scatter(pd.Timestamp.now(), resultados[target]["Predicción"], color="green", s=100,
                      label="Predicción nueva")
        # Intervalo de confianza
        ax[i].fill_between([pd.Timestamp.now()], resultados[target]["CI_lower"], resultados[target]["CI_upper"],
                           color="green", alpha=0.3, label="CI")
        ax[i].set_title(target)
        ax[i].legend()
    canvas.draw()


# --- FUNCIONES DE PDF ---
def generar_pdf():
    with PdfPages(pdf_path) as pdf:
        # Tabla coeficientes
        fig, ax_pdf = plt.subplots(figsize=(10, 6))
        ax_pdf.axis('off')
        ax_pdf.set_title("Coeficientes y p-values", fontsize=14, fontweight='bold')
        tabla = ax_pdf.table(cellText=df_coefs.round(4).values,
                             colLabels=df_coefs.columns,
                             cellLoc='center', loc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1, 1.5)
        pdf.savefig()
        plt.close()

        # Gráficos históricos
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
    messagebox.showinfo("PDF Generado", f"✅ PDF generado en '{pdf_path}'")


# --- FUNCIONES CSV REAL ---
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
root.title("Dashboard UCI y Fallecidos")

# Variables
vac_var = tk.IntVar()
conf_var = tk.IntVar()

# Checkboxes
tk.Checkbutton(root, text="Hay vacuna", variable=vac_var, command=actualizar_graficos).grid(row=0, column=0, sticky="w",
                                                                                            padx=10)
tk.Checkbutton(root, text="Hay confinamiento", variable=conf_var, command=actualizar_graficos).grid(row=1, column=0,
                                                                                                    sticky="w", padx=10)

# Botones
tk.Button(root, text="Generar PDF", command=generar_pdf).grid(row=2, column=0, pady=5)
tk.Button(root, text="Aplicar CSV Real", command=aplicar_csv_real).grid(row=3, column=0, pady=5)
tk.Button(root, text="Salir", command=root.quit).grid(row=4, column=0, pady=5)

# --- Gráficos embebidos ---
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=1, rowspan=5, padx=10, pady=5)
actualizar_graficos()  # Inicializa los gráficos

root.mainloop()

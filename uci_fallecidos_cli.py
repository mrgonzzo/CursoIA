import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- CONFIGURACIÓN DE RUTAS ---
base_dir = r"C:\Users\tarde\Desktop\ProgramacionCursoIA"
data_path = os.path.join(base_dir, "DATOS", "2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv")
output_path = os.path.join(base_dir, "DATOS")
pdf_path = os.path.join(output_path, "Reporte_UCI_Fallecidos_CLI.pdf")
csv_output_path = os.path.join(output_path, "uci_fallecidos_predicciones_cli.csv")

# --- CARGA DE DATOS ---
print("=== CARGANDO ARCHIVO REAL ===")
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
def predecir_nuevos(hay_vacuna, hay_confinamiento, alpha=0.05):
    X_new = pd.DataFrame([[1, hay_vacuna, hay_confinamiento]], columns=["const"] + features)
    resultados = {}
    for target, model in model_dict.items():
        pred = model.get_prediction(X_new)
        pred_mean = pred.predicted_mean[0]
        ci_lower, ci_upper = pred.conf_int(alpha=alpha)[0]
        resultados[target] = {"Predicción": pred_mean, "CI_lower": ci_lower, "CI_upper": ci_upper}
    return resultados


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


def aplicar_csv_real():
    csv_real = data_path
    if os.path.exists(csv_real):
        df_real = pd.read_csv(csv_real)
        X_real = df_real[features]
        X_real = sm.add_constant(X_real)
        for target, model in model_dict.items():
            pred_real = model.get_prediction(X_real)
            df_real[f"{target}_pred"] = pred_real.predicted_mean
            df_real[f"{target}_CI_lower"] = pred_real.conf_int()[:, 0]
            df_real[f"{target}_CI_upper"] = pred_real.conf_int()[:, 1]
        df_real.to_csv(csv_output_path, index=False)
        print(f"✅ Predicciones guardadas en '{csv_output_path}'")


# --- INTERFAZ CLI ---
def menu_cli():
    print("=== INTERFAZ CLI DE PREDICCIÓN UCI Y FALLECIDOS ===")
    while True:
        print("\nOpciones:")
        print("1 - Hacer una predicción nueva")
        print("2 - Generar PDF completo")
        print("3 - Aplicar modelo al CSV real")
        print("4 - Salir")
        opcion = input("Selecciona una opción (1-4): ")

        if opcion == "1":
            vac = int(input("Hay vacuna? (0=no,1=si): "))
            conf = int(input("Hay confinamiento? (0=no,1=si): "))
            resultados = predecir_nuevos(vac, conf)
            for target, val in resultados.items():
                print(f"{target}: Predicción={val['Predicción']:.2f}, CI=({val['CI_lower']:.2f},{val['CI_upper']:.2f})")
        elif opcion == "2":
            generar_pdf()
            print(f"✅ PDF generado en '{pdf_path}'")
        elif opcion == "3":
            aplicar_csv_real()
        elif opcion == "4":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

3
if __name__ == "__main__":
    menu_cli()

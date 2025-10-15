# ======================================================================
# PREDICTOR DE SALARIOS CON INTERFAZ GRÁFICA (Tkinter)
# Basado en el dataset 'wage1' del paquete wooldridge
# ======================================================================

import wooldridge as woo
import pandas as pd
from statsmodels.formula.api import ols
import tkinter as tk
from tkinter import ttk, messagebox

# ======================================================================
# 1. CARGA DE DATOS Y MODELO
# ======================================================================

# Cargar el dataset 'wage1'
df = woo.dataWoo('wage1').copy()

# Crear variable categórica de educación
bins = [0, 8, 12, 14, 16, df['educ'].max() + 1]
labels = ['A_Baja (<8)', 'B_Secundaria (8-12)', 'C_Colegio/Técnica (13-14)', 'D_Licenciatura (15-16)', 'E_Posgrado (>16)']
df['educ_categorica'] = pd.cut(df['educ'], bins=bins, labels=labels, include_lowest=True, ordered=True)

# Modelo OLS
formula = "wage ~ C(educ_categorica) + female + exper + tenure"
modelo = ols(formula, data=df).fit()

# Extraer coeficientes
params = modelo.params

CATEGORIA_BASE = labels[0]
CATEGORIAS_EDUCACION = {
    'B_Secundaria (8-12)': 'C(educ_categorica)[T.B_Secundaria (8-12)]',
    'C_Colegio/Técnica (13-14)': 'C(educ_categorica)[T.C_Colegio/Técnica (13-14)]',
    'D_Licenciatura (15-16)': 'C(educ_categorica)[T.D_Licenciatura (15-16)]',
    'E_Posgrado (>16)': 'C(educ_categorica)[T.E_Posgrado (>16)]'
}

coef = {
    'Intercept': params['Intercept'],
    'female': params['female'],
    'exper': params['exper'],
    'tenure': params['tenure']
}
for nombre, nombre_modelo in CATEGORIAS_EDUCACION.items():
    coef[nombre] = params[nombre_modelo]

# ======================================================================
# 2. FUNCIÓN DE PREDICCIÓN
# ======================================================================

def calcular_salario_predicho(educ_cat, female, exper, tenure):
    salario = coef['Intercept']
    if educ_cat != CATEGORIA_BASE and educ_cat in CATEGORIAS_EDUCACION:
        salario += coef[educ_cat]
    salario += coef['female'] * female
    salario += coef['exper'] * exper
    salario += coef['tenure'] * tenure
    return salario

# ======================================================================
# 3. INTERFAZ GRÁFICA (Tkinter)
# ======================================================================

ventana = tk.Tk()
ventana.title("📊 Predictor de Salarios (Wooldridge)")
ventana.geometry("450x400")
ventana.resizable(False, False)

tk.Label(ventana, text="Predictor de Salarios", font=("Helvetica", 16, "bold")).pack(pady=10)
tk.Label(ventana, text="Basado en datos de Wooldridge (wage1)").pack()

# Entradas
frame = tk.Frame(ventana)
frame.pack(pady=10)

# Nivel educativo
tk.Label(frame, text="Nivel educativo:").grid(row=0, column=0, sticky="e")
combo_educ = ttk.Combobox(frame, values=labels, state="readonly")
combo_educ.current(0)
combo_educ.grid(row=0, column=1)

# Género
tk.Label(frame, text="Género:").grid(row=1, column=0, sticky="e")
combo_genero = ttk.Combobox(frame, values=["Hombre (0)", "Mujer (1)"], state="readonly")
combo_genero.current(0)
combo_genero.grid(row=1, column=1)

# Experiencia
tk.Label(frame, text="Años de experiencia:").grid(row=2, column=0, sticky="e")
entrada_exper = tk.Entry(frame)
entrada_exper.insert(0, "5")
entrada_exper.grid(row=2, column=1)

# Antigüedad
tk.Label(frame, text="Años de antigüedad:").grid(row=3, column=0, sticky="e")
entrada_tenure = tk.Entry(frame)
entrada_tenure.insert(0, "2")
entrada_tenure.grid(row=3, column=1)

# Resultado
label_resultado = tk.Label(ventana, text="", font=("Helvetica", 14, "bold"))
label_resultado.pack(pady=15)

# Función del botón
def predecir():
    try:
        educ = combo_educ.get()
        genero = 1 if "Mujer" in combo_genero.get() else 0
        exper = float(entrada_exper.get())
        tenure = float(entrada_tenure.get())

        if exper < 0 or tenure < 0:
            messagebox.showerror("Error", "Los años no pueden ser negativos.")
            return

        salario = calcular_salario_predicho(educ, genero, exper, tenure)
        label_resultado.config(text=f"💰 Salario predicho: ${salario:.2f} por hora")
    except ValueError:
        messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos.")

# Botón
tk.Button(ventana, text="Predecir Salario", command=predecir, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=20).pack(pady=10)

ventana.mainloop()
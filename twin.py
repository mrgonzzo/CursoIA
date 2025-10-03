# twins.py
# Análisis econométrico del fenómeno de Déficits Gemelos

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


def main():
    print("=== ANÁLISIS DÉFICITS GEMELOS ===")

    # 1. Cargar y explorar datos
    try:
        df_twin = pd.read_csv('C:/Users/gonzzo/Desktop/CURSO/PycharmProjects/DATOS/TwinDeficits.csv')
        print("✓ Datos cargados exitosamente")
    except FileNotFoundError:
        print("❌ Error: Archivo 'twin_deficit.csv' no encontrado")
        return

    # Exploración inicial de datos
    print("\n1. EXPLORACIÓN DE DATOS:")
    print("Primeras 5 filas:")
    print(df_twin.head())

    # Identificar variables clave
    variables_clave = ['IntDef', 'ExtDef', 'Expenditures', 'Revenues', 'Imports', 'Exports']
    variables_disponibles = [var for var in variables_clave if var in df_twin.columns]

    print(f"\nVariables disponibles para análisis: {variables_disponibles}")

    # Verificar variables con prefijos
    prefijos = ['d_', 'l_', 'sd_']
    variables_con_prefijos = {}

    for prefijo in prefijos:
        vars_prefijo = [col for col in df_twin.columns if col.startswith(prefijo)]
        if vars_prefijo:
            variables_con_prefijos[prefijo] = vars_prefijo

    print("\nVariables con prefijos:")
    for prefijo, variables in variables_con_prefijos.items():
        print(f"  {prefijo}: {variables}")

    # Estadísticas descriptivas
    if variables_disponibles:
        print("\nEstadísticas descriptivas de variables clave:")
        print(df_twin[variables_disponibles].describe())

    # 2. Análisis de estacionariedad (si hay datos temporales)
    print("\n2. ANÁLISIS DE ESTACIONARIEDAD:")

    # Test de raíz unitaria para variables principales
    for var in ['IntDef', 'ExtDef']:
        if var in df_twin.columns:
            result = adfuller(df_twin[var].dropna())
            print(f"Test ADF para {var}:")
            print(f"  Estadístico: {result[0]:.4f}")
            print(f"  P-valor: {result[1]:.4f}")
            if result[1] < 0.05:
                print("  ✓ Serie estacionaria")
            else:
                print("  ✗ Serie no estacionaria - considerar usar diferencias")

    # 3. Modelo de regresión MCO para déficits gemelos
    print("\n3. MODELO DE REGRESIÓN MCO:")

    # Definir variables para el modelo base
    if all(var in df_twin.columns for var in ['ExtDef', 'IntDef']):
        X_vars = ['IntDef']

        # Agregar variables de control si existen
        if 'd_Imports' in df_twin.columns:
            X_vars.append('d_Imports')
        elif 'Imports' in df_twin.columns:
            X_vars.append('Imports')

        # Preparar datos para regresión
        datos_regresion = df_twin[['ExtDef'] + X_vars].dropna()
        y_twin = datos_regresion['ExtDef']
        X_twin = datos_regresion[X_vars]
        X_twin = sm.add_constant(X_twin)

        # Estimar modelo
        modelo_twin = sm.OLS(y_twin, X_twin).fit()

        # Mostrar resultados
        print(modelo_twin.summary())

        # 4. Interpretación de resultados
        print("\n4. INTERPRETACIÓN ECONÓMICA:")

        for var in X_vars:
            coef = modelo_twin.params[var]
            p_valor = modelo_twin.pvalues[var]

            print(f"\nVariable: {var}")
            print(f"Coeficiente: {coef:.4f}")
            print(f"P-valor: {p_valor:.4f}")

            if p_valor < 0.05:
                if var == 'IntDef':
                    if coef > 0:
                        print("✓ Apoya la hipótesis de déficits gemelos")
                    else:
                        print("✗ No apoya la hipótesis de déficits gemelos")

        r_cuadrado = modelo_twin.rsquared
        print(f"\nR-cuadrado del modelo: {r_cuadrado:.4f}")
        print(f"El modelo explica {r_cuadrado * 100:.2f}% de la variación en el déficit externo")

    else:
        print("❌ Variables necesarias (ExtDef, IntDef) no encontradas en el dataset")
        return

    # 5. Visualizaciones
    print("\n5. GENERANDO VISUALIZACIONES...")

    # Configurar estilo de gráficos
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Gráfico 1: Relación entre déficits interno y externo
    if all(var in df_twin.columns for var in ['IntDef', 'ExtDef']):
        axes[0, 0].scatter(df_twin['IntDef'], df_twin['ExtDef'], alpha=0.6)

        # Calcular línea de tendencia
        z = np.polyfit(df_twin['IntDef'].dropna(), df_twin['ExtDef'].dropna(), 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df_twin['IntDef'], p(df_twin['IntDef']), "r--", alpha=0.8)

        axes[0, 0].set_title('Relación Déficits Gemelos', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Déficit Interno (IntDef)')
        axes[0, 0].set_ylabel('Déficit Externo (ExtDef)')
        axes[0, 0].grid(True, alpha=0.3)

    # Gráfico 2: Series de tiempo de déficits
    if all(var in df_twin.columns for var in ['IntDef', 'ExtDef']):
        # Usar índice como tiempo si no hay variable temporal explícita
        tiempo = range(len(df_twin))

        axes[0, 1].plot(tiempo, df_twin['IntDef'], label='Déficit Interno', linewidth=2)
        axes[0, 1].plot(tiempo, df_twin['ExtDef'], label='Déficit Externo', linewidth=2)
        axes[0, 1].set_title('Evolución Temporal de los Déficits', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Período')
        axes[0, 1].set_ylabel('Magnitud del Déficit')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Matriz de correlación
    variables_corr = [var for var in variables_disponibles if var in df_twin.columns]
    if len(variables_corr) > 1:
        matriz_corr = df_twin[variables_corr].corr()
        sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0,
                    ax=axes[1, 0], square=True)
        axes[1, 0].set_title('Matriz de Correlación', fontsize=14, fontweight='bold')

    # Gráfico 4: Residuales del modelo
    if 'modelo_twin' in locals():
        fitted_values = modelo_twin.fittedvalues
        residuals = modelo_twin.resid
        axes[1, 1].scatter(fitted_values, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuales vs Valores Ajustados', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Valores Ajustados')
        axes[1, 1].set_ylabel('Residuales')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analisis_deficits_gemelos.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. Análisis de robustez con diferentes especificaciones
    print("\n6. ANÁLISIS DE ROBUSTEZ:")

    # Probar diferentes especificaciones del modelo
    especificaciones = [
        ['IntDef'],
        ['IntDef', 'd_Imports'] if 'd_Imports' in df_twin.columns else ['IntDef'],
        ['IntDef', 'Expenditures'] if 'Expenditures' in df_twin.columns else ['IntDef']
    ]

    for i, espec in enumerate(especificaciones):
        if all(var in df_twin.columns for var in espec):
            datos_robustez = df_twin[['ExtDef'] + espec].dropna()
            y_rob = datos_robustez['ExtDef']
            X_rob = datos_robustez[espec]
            X_rob = sm.add_constant(X_rob)

            modelo_rob = sm.OLS(y_rob, X_rob).fit()

            print(f"\nEspecificación {i + 1}: ExtDef ~ {', '.join(espec)}")
            print(f"  Coef. IntDef: {modelo_rob.params['IntDef']:.4f} (p-valor: {modelo_rob.pvalues['IntDef']:.4f})")
            print(f"  R-cuadrado: {modelo_rob.rsquared:.4f}")

    print("\n✓ Análisis completado. Gráficos guardados como 'analisis_deficits_gemelos.png'")


if __name__ == "__main__":
    main()
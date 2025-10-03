# ventas_publi.py
# Análisis econométrico de la relación entre ventas y publicidad

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("=== ANÁLISIS VENTAS VS PUBLICIDAD ===")

    # 1. Cargar y explorar datos
    try:
        df_ventas = pd.read_csv('C:/Users/gonzzo/Desktop/CURSO/PycharmProjects/DATOS/ventasypubl.csv')
        print("✓ Datos cargados exitosamente")
    except FileNotFoundError:
        print("❌ Error: Archivo 'ventasypubl.csv' no encontrado")
        return

    # Exploración inicial de datos
    print("\n1. EXPLORACIÓN DE DATOS:")
    print("Primeras 5 filas:")
    print(df_ventas.head())

    print("\nEstadísticas descriptivas:")
    print(df_ventas[['pub', 'vtas']].describe())

    print(f"\nNúmero de observaciones: {len(df_ventas)}")

    # 2. Análisis de correlación
    print("\n2. ANÁLISIS DE CORRELACIÓN:")
    correlacion = df_ventas['pub'].corr(df_ventas['vtas'])
    print(f"Correlación entre pub y vtas: {correlacion:.4f}")

    # 3. Modelo de regresión MCO
    print("\n3. MODELO DE REGRESIÓN MCO:")

    # Preparar variables
    X = df_ventas['pub']
    y = df_ventas['vtas']
    X = sm.add_constant(X)  # Añadir intercepto (β₀)

    # Estimar modelo
    modelo_ventas = sm.OLS(y, X).fit()

    # Mostrar resultados
    print(modelo_ventas.summary())

    # 4. Interpretación de resultados
    print("\n4. INTERPRETACIÓN ECONÓMICA:")
    coef_pub = modelo_ventas.params['pub']
    p_valor_pub = modelo_ventas.pvalues['pub']
    r_cuadrado = modelo_ventas.rsquared

    print(f"Coeficiente de publicidad (β₁): {coef_pub:.4f}")
    print(f"P-valor del coeficiente: {p_valor_pub:.4f}")
    print(f"R-cuadrado: {r_cuadrado:.4f}")

    if p_valor_pub < 0.05:
        significancia = "estadísticamente significativo"
    else:
        significancia = "no estadísticamente significativo"

    print(
        f"\nInterpretación: Cada unidad adicional en publicidad está asociada con {coef_pub:.4f} unidades de cambio en ventas.")
    print(f"La relación es {significancia} (p-valor: {p_valor_pub:.4f}).")
    print(f"El modelo explica {r_cuadrado * 100:.2f}% de la variación en las ventas.")

    # 5. Visualizaciones
    print("\n5. GENERANDO VISUALIZACIONES...")

    # Configurar estilo de gráficos
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Gráfico 1: Diagrama de dispersión con línea de regresión
    sns.regplot(x='pub', y='vtas', data=df_ventas, ci=95, ax=axes[0, 0], line_kws={'color': 'red'})
    axes[0, 0].set_title('Relación entre Publicidad y Ventas', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Gasto en Publicidad')
    axes[0, 0].set_ylabel('Ventas')

    # Gráfico 2: Series de tiempo (si hay variable temporal)
    if 'dateid' in df_ventas.columns or 'dateid01' in df_ventas.columns:
        if 'dateid' in df_ventas.columns:
            tiempo = df_ventas['dateid']
        else:
            tiempo = df_ventas['dateid01']

        axes[0, 1].plot(tiempo, df_ventas['pub'], label='Publicidad', marker='o')
        axes[0, 1].plot(tiempo, df_ventas['vtas'], label='Ventas', marker='s')
        axes[0, 1].set_title('Evolución Temporal de Publicidad y Ventas', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Tiempo')
        axes[0, 1].set_ylabel('Valor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Residuales vs Valores ajustados
    fitted_values = modelo_ventas.fittedvalues
    residuals = modelo_ventas.resid
    axes[1, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_title('Análisis de Residuales', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Valores Ajustados')
    axes[1, 0].set_ylabel('Residuales')
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfico 4: Histograma de residuales
    axes[1, 1].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Distribución de Residuales', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Residuales')
    axes[1, 1].set_ylabel('Frecuencia')

    plt.tight_layout()
    plt.savefig('analisis_ventas_publicidad.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. Diagnósticos adicionales
    print("\n6. DIAGNÓSTICOS DEL MODELO:")
    print(f"Estadístico Durbin-Watson: {sm.stats.durbin_watson(modelo_ventas.resid):.4f}")
    print("(Valores cerca de 2 indican no autocorrelación)")

    # Test de normalidad de residuales
    from scipy import stats
    stat_normal, p_normal = stats.shapiro(residuals)
    print(f"Test de normalidad (Shapiro-Wilk): p-valor = {p_normal:.4f}")

    print("\n✓ Análisis completado. Gráficos guardados como 'analisis_ventas_publicidad.png'")


if __name__ == "__main__":
    main()
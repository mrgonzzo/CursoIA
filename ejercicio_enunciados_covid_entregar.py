"""
===============================================================================
EJERCICIO: ANÁLISIS DE DATOS COVID-19 EN ESPAÑA
Patrones de Búsqueda y Algoritmos de Ordenamiento
===============================================================================

CONTEXTO:
Eres un analista de datos del Ministerio de Sanidad de España. Tu tarea es
desarrollar un sistema para analizar los datos históricos de la pandemia de
COVID-19, específicamente los ingresos en UCI y fallecimientos diarios.

La base de datos contiene información diaria desde enero de 2020 hasta
aproximadamente 2022, con los siguientes campos:
- Año
- Fecha (texto)
- Ingresos_UCI_diarios
- Fallecidos_diarios
- Hay_vacuna (0=No, 1=Sí)
- Hay_Confinamiento (0=No, 1=Sí)
- Fecha_num (formato numérico)
- Fecha_r (formato ISO)

OBJETIVO:
Implementar algoritmos de búsqueda y ordenamiento para extraer información
relevante que ayude en la toma de decisiones de salud pública.
===============================================================================
"""

import csv
from typing import List, Optional, Tuple
from datetime import datetime


class RegistroCovid:
    """Clase que representa un registro diario de COVID-19"""

    def __init__(self, año: int, fecha: str, ingresos_uci: int,
                 fallecidos: int, hay_vacuna: int, hay_confinamiento: int,
                 fecha_num: int, fecha_iso: str):
        self.año = año
        self.fecha = fecha
        self.ingresos_uci = ingresos_uci
        self.fallecidos = fallecidos
        self.hay_vacuna = bool(hay_vacuna)
        self.hay_confinamiento = bool(hay_confinamiento)
        self.fecha_num = fecha_num
        self.fecha_iso = fecha_iso

    def __repr__(self):
        return (f"RegistroCovid({self.fecha}, UCI:{self.ingresos_uci}, "
                f"Fallecidos:{self.fallecidos}, Vacuna:{self.hay_vacuna})")


class AnalizadorCovid:
    """Sistema de análisis de datos COVID-19"""

    def __init__(self, archivo_csv: str):
        self.registros: List[RegistroCovid] = []
        self.cargar_datos(archivo_csv)

    def cargar_datos(self, archivo_csv: str):
        """Carga los datos desde el archivo CSV"""
        with open(archivo_csv, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                registro = RegistroCovid(
                    año=int(row['Año']),
                    fecha=row['Fecha '].strip(),
                    ingresos_uci=int(row['Ingresos_UCI_diarios']),
                    fallecidos=int(row['Fallecidos_diarios']),
                    hay_vacuna=int(row['Hay_vacuna']),
                    hay_confinamiento=int(row['Hay_Confinamiento']),
                    fecha_num=int(row['Fecha_num']),
                    fecha_iso=row['Fecha_r'].strip()
                )
                self.registros.append(registro)

        print(f"✓ Datos cargados: {len(self.registros)} registros")

    # =========================================================================
    # EJERCICIO 1: BÚSQUEDA LINEAL
    # =========================================================================
    def buscar_fecha_exacta(self, fecha_num: int) -> Optional[RegistroCovid]:
        """
        TODO: Implementa una búsqueda lineal para encontrar los datos de una
        fecha específica usando el formato numérico (ej: 20200315 para 15/03/2020)

        Complejidad temporal: O(n)
        Complejidad espacial: O(1)

        Args:
            fecha_num: Fecha en formato numérico (YYYYMMDD)

        Returns:
            El registro de esa fecha o None si no existe

        PISTA: Recorre la lista secuencialmente comparando fecha_num

        RESULTADO ESPERADO para fecha_num=20200325:
        RegistroCovid(25 de marzo de 2020, UCI:345, Fallecidos:324, Vacuna:False)
        """
        # TU CÓDIGO AQUÍ
        pass
        for registro in self.registros:
            if registro.fecha_num == fecha_num:
                return registro
        return None

    # =========================================================================
    # EJERCICIO 2: BÚSQUEDA BINARIA
    # =========================================================================
    def buscar_primer_dia_con_fallecidos_minimos(self,
                                                 minimo_fallecidos: int) -> Optional[RegistroCovid]:
        """
        TODO: Usa búsqueda binaria para encontrar el primer día que tuvo al menos
        X fallecidos. Primero debes ordenar los registros por número de fallecidos.

        Complejidad temporal: O(log n) para la búsqueda + O(n log n) para ordenar
        Complejidad espacial: O(n) para la lista ordenada

        Args:
            minimo_fallecidos: Número mínimo de fallecidos a buscar

        Returns:
            El primer registro que cumple la condición

        PISTA:
        1. Ordena los registros por fallecidos_diarios
        2. Aplica búsqueda binaria para encontrar el valor objetivo

        RESULTADO ESPERADO para minimo_fallecidos=500:
        Debería encontrar un día de 2020 con aproximadamente 500+ fallecidos
        """
        # Paso 1: Ordenar registros por fallecidos
        registros_ordenados = sorted(self.registros, key=lambda r: r.fallecidos)

        # TU CÓDIGO AQUÍ - Implementa búsqueda binaria
        pass
        # Paso 2: búsqueda binaria
        izquierda = 0
        derecha = len(registros_ordenados) - 1
        resultado = None

        while izquierda <= derecha:
            medio = (izquierda + derecha) // 2
            if registros_ordenados[medio].fallecidos >= minimo_fallecidos:
                resultado = registros_ordenados[medio]
                derecha = medio - 1  # buscar si hay uno anterior que también cumpla
            else:
                izquierda = medio + 1

        return resultado.fecha



    # =========================================================================
    # EJERCICIO 3: ORDENAMIENTO BURBUJA (BUBBLE SORT)
    # =========================================================================
    def ordenar_por_burbuja_ingresos_uci(self) -> List[RegistroCovid]:
        """
        TODO: Implementa el algoritmo de ordenamiento burbuja para ordenar
        los registros por número de ingresos en UCI (de menor a mayor).

        Complejidad temporal: O(n²)
        Complejidad espacial: O(n) para la copia

        Returns:
            Lista ordenada de registros

        ALGORITMO:
        1. Hacer una copia de los registros
        2. Repetir n veces:
           - Comparar elementos adyacentes
           - Si están en orden incorrecto, intercambiarlos
        3. Retornar lista ordenada

        RESULTADO ESPERADO (primeros 3):
        - Días con 0 ingresos UCI (varios al inicio de la pandemia)

        RESULTADO ESPERADO (últimos 3):
        - Días con mayor número de ingresos UCI (picos de la pandemia)
        """
        registros_copia = self.registros.copy()
        n = len(registros_copia)

        # TU CÓDIGO AQUÍ
        pass
        for i in range(n):
            for j in range(0, n - i - 1):
                if registros_copia[j].ingresos_uci > registros_copia[j + 1].ingresos_uci:
                    # Intercambiar si están en orden incorrecto
                    registros_copia[j], registros_copia[j + 1] = registros_copia[j + 1], registros_copia[j]

        return registros_copia

    # =========================================================================
    # EJERCICIO 4: ORDENAMIENTO RÁPIDO (QUICKSORT)
    # =========================================================================
    def quicksort_por_fallecidos(self, registros: List[RegistroCovid]) -> List[RegistroCovid]:
        """
        TODO: Implementa QuickSort para ordenar registros por fallecidos (descendente).

        Complejidad temporal: O(n log n) promedio, O(n²) peor caso
        Complejidad espacial: O(log n) por la recursión

        Args:
            registros: Lista de registros a ordenar

        Returns:
            Lista ordenada de mayor a menor número de fallecidos

        ALGORITMO:
        1. Caso base: si la lista tiene 0 o 1 elementos, retornarla
        2. Elegir un pivote (puedes usar el elemento del medio)
        3. Dividir en tres sublistas: menores, iguales y mayores al pivote
        4. Aplicar quicksort recursivamente a menores y mayores
        5. Concatenar: mayores + iguales + menores (para orden descendente)

        RESULTADO ESPERADO (primeros 5):
        Los 5 días con más fallecidos de toda la pandemia
        (probablemente en marzo-abril 2020)
        """
        # TU CÓDIGO AQUÍ
        pass
        if len(registros) <= 1:
            return registros

            # Paso 2: elegir pivote (elemento del medio)
        pivote = registros[len(registros) // 2].fallecidos

        # Paso 3: dividir en tres sublistas
        mayores = [r for r in registros if r.fallecidos > pivote]
        iguales = [r for r in registros if r.fallecidos == pivote]
        menores = [r for r in registros if r.fallecidos < pivote]

        # Paso 4: aplicar quicksort recursivamente
        orden_mayores = self.quicksort_por_fallecidos(mayores)
        orden_menores = self.quicksort_por_fallecidos(menores)

        # Paso 5: concatenar en orden descendente
        return orden_mayores + iguales + orden_menores
    # =========================================================================
    # EJERCICIO 5: BÚSQUEDA CON MÚLTIPLES FILTROS
    # =========================================================================
    def buscar_con_condiciones(self,
                               año: int = None,
                               con_vacuna: bool = None,
                               con_confinamiento: bool = None,
                               min_ingresos_uci: int = None,
                               max_fallecidos: int = None) -> List[RegistroCovid]:
        """
        TODO: Implementa una búsqueda que filtre registros según múltiples condiciones.

        Complejidad temporal: O(n)
        Complejidad espacial: O(k) donde k es el número de resultados

        Args:
            año: Filtrar por año específico
            con_vacuna: True/False/None para filtrar por disponibilidad de vacuna
            con_confinamiento: True/False/None para filtrar por confinamiento
            min_ingresos_uci: Mínimo de ingresos UCI
            max_fallecidos: Máximo de fallecidos

        Returns:
            Lista de registros que cumplen TODAS las condiciones especificadas

        PISTA: Usa condicionales para verificar cada filtro solo si no es None

        RESULTADO ESPERADO para (año=2021, con_vacuna=True, min_ingresos_uci=100):
        Días de 2021 con vacuna disponible y al menos 100 ingresos UCI
        """
        resultado = []

        # TU CÓDIGO AQUÍ
        pass
        for r in self.registros:
            if año is not None and r.año != año:
                continue
            if con_vacuna is not None and r.hay_vacuna != con_vacuna:
                continue
            if con_confinamiento is not None and r.hay_confinamiento != con_confinamiento:
                continue
            if min_ingresos_uci is not None and r.ingresos_uci < min_ingresos_uci:
                continue
            if max_fallecidos is not None and r.fallecidos > max_fallecidos:
                continue

            resultado.append(r)

        return resultado
    # =========================================================================
    # EJERCICIO 6: BÚSQUEDA DE PICOS (MÁXIMOS LOCALES)
    # =========================================================================
    def encontrar_picos_ingresos_uci(self, ventana: int = 7) -> List[RegistroCovid]:
        """
        TODO: Encuentra los días que fueron picos locales de ingresos en UCI.
        Un pico local es un día con más ingresos que los días anteriores y
        posteriores en una ventana de tiempo.

        Complejidad temporal: O(n * ventana)
        Complejidad espacial: O(k) donde k es el número de picos

        Args:
            ventana: Número de días a cada lado para comparar (default=7)

        Returns:
            Lista de registros que son picos locales

        ALGORITMO:
        1. Para cada día (excepto los bordes):
           - Comparar con los 'ventana' días anteriores
           - Comparar con los 'ventana' días posteriores
           - Si es mayor que todos ellos, es un pico

        RESULTADO ESPERADO:
        Identificar las diferentes olas de la pandemia (marzo 2020, enero 2021, etc.)
        """
        picos = []

        # TU CÓDIGO AQUÍ
        pass
        n = len(self.registros)

        # Recorremos desde 'ventana' hasta 'n - ventana - 1' para evitar bordes
        for i in range(ventana, n - ventana):
            actual = self.registros[i].ingresos_uci

            # Extraer ventanas anterior y posterior
            anteriores = [self.registros[j].ingresos_uci for j in range(i - ventana, i)]
            posteriores = [self.registros[j].ingresos_uci for j in range(i + 1, i + ventana + 1)]

            # Verificar si el día actual es mayor que todos los vecinos
            if all(actual > x for x in anteriores + posteriores):
                picos.append(self.registros[i])

        return picos
    # =========================================================================
    # EJERCICIO 7: TOP K DÍAS MÁS CRÍTICOS
    # =========================================================================
    def obtener_top_k_dias_criticos(self, k: int = 10) -> List[RegistroCovid]:
        """
        TODO: Encuentra los K días más críticos basándose en un índice de
        criticidad = (ingresos_uci * 0.5) + (fallecidos * 1.5)

        Complejidad temporal: O(n log k) usando un heap, o O(n log n) ordenando
        Complejidad espacial: O(n) para la lista con índices

        Args:
            k: Número de días críticos a retornar

        Returns:
            Lista de los K días más críticos

        ALGORITMO (opción simple):
        1. Crear una lista de tuplas (índice_criticidad, registro)
        2. Ordenar por índice de criticidad (descendente)
        3. Retornar los primeros K

        RESULTADO ESPERADO para k=10:
        Los 10 días más graves de la pandemia, considerando tanto UCI como fallecidos
        """
        # TU CÓDIGO AQUÍ
        pass
        # Paso 1: calcular índice de criticidad para cada registro
        registros_con_indice = [
            ((r.ingresos_uci * 0.5) + (r.fallecidos * 1.5), r)
            for r in self.registros
        ]

        # Paso 2: ordenar por índice de criticidad (descendente)
        registros_ordenados = sorted(registros_con_indice, key=lambda x: x[0], reverse=True)

        # Paso 3: extraer los primeros K registros
        top_k = [registro for _, registro in registros_ordenados[:k]]

        return top_k
    # =========================================================================
    # EJERCICIO 8: ANÁLISIS DE TENDENCIAS (BÚSQUEDA DE PATRONES)
    # =========================================================================
    def detectar_periodo_crecimiento_sostenido(self, dias_consecutivos: int = 7) -> List[Tuple[str, str]]:
        """
        TODO: Detecta periodos donde los ingresos UCI crecieron durante N días consecutivos.

        Complejidad temporal: O(n)
        Complejidad espacial: O(m) donde m es el número de periodos encontrados

        Args:
            dias_consecutivos: Número mínimo de días de crecimiento continuo

        Returns:
            Lista de tuplas (fecha_inicio, fecha_fin) de cada periodo

        ALGORITMO:
        1. Recorrer los registros secuencialmente
        2. Contar días consecutivos de crecimiento
        3. Cuando se alcance el umbral, guardar el periodo
        4. Reiniciar contador si hay decrecimiento

        RESULTADO ESPERADO:
        Identificar los momentos donde la pandemia se aceleró rápidamente
        (inicio de cada ola)
        """
        periodos = []

        # TU CÓDIGO AQUÍ
        pass
        contador = 0
        inicio = None

        for i in range(1, len(self.registros)):
            anterior = self.registros[i - 1].ingresos_uci
            actual = self.registros[i].ingresos_uci
            if actual > anterior:
                contador += 1
                if contador == 1:
                    inicio = self.registros[i - 1].fecha_iso
                if contador >= dias_consecutivos:
                    print(f'contador = {contador}')
                    fin = self.registros[i].fecha_iso
                    periodos.append((inicio, fin))
                    # Para evitar solapamientos, reiniciamos el contador
                    contador = 0
                    inicio = None
            else:
                contador = 0
                inicio = None

        return periodos
    # =========================================================================
    # EJERCICIO 9: CÁLCULO DE ESTADÍSTICAS POR PERIODO
    # =========================================================================
    def calcular_estadisticas_por_periodo(self, año: int,
                                          con_confinamiento: bool) -> dict:
        """
        TODO: Calcula estadísticas agregadas para un periodo específico.

        Complejidad temporal: O(n)
        Complejidad espacial: O(1)

        Args:
            año: Año a analizar
            con_confinamiento: Analizar periodo con o sin confinamiento

        Returns:
            Diccionario con: total_ingresos_uci, total_fallecidos,
            promedio_ingresos, promedio_fallecidos, max_ingresos, max_fallecidos

        RESULTADO ESPERADO para (año=2020, con_confinamiento=True):
        Estadísticas del periodo de confinamiento de 2020
        """
        # TU CÓDIGO AQUÍ
        pass
        total_ingresos = 0
        total_fallecidos = 0
        max_ingresos = 0
        max_fallecidos = 0
        contador = 0

        for r in self.registros:
            if r.año == año and r.hay_confinamiento == con_confinamiento:
                total_ingresos += r.ingresos_uci
                total_fallecidos += r.fallecidos
                max_ingresos = max(max_ingresos, r.ingresos_uci)
                max_fallecidos = max(max_fallecidos, r.fallecidos)
                contador += 1

        if contador == 0:
            return {
                "total_ingresos_uci": 0,
                "total_fallecidos": 0,
                "promedio_ingresos": 0,
                "promedio_fallecidos": 0,
                "max_ingresos": 0,
                "max_fallecidos": 0
            }

        return {
            "total_ingresos_uci": total_ingresos,
            "total_fallecidos": total_fallecidos,
            "promedio_ingresos": total_ingresos // contador,
            "promedio_fallecidos": total_fallecidos // contador,
            "max_ingresos": max_ingresos,
            "max_fallecidos": max_fallecidos
        }
    # =========================================================================
    # EJERCICIO 10: COMPARACIÓN ANTES/DESPUÉS DE LA VACUNA
    # =========================================================================
    def comparar_pre_post_vacuna(self) -> dict:
        """
        TODO: Compara las estadísticas antes y después de la introducción de la vacuna.

        Complejidad temporal: O(n)
        Complejidad espacial: O(1)

        Returns:
            Diccionario con estadísticas comparativas:
            {
                'antes_vacuna': {promedios y totales},
                'despues_vacuna': {promedios y totales},
                'reduccion_porcentual': {porcentajes de reducción}
            }

        RESULTADO ESPERADO:
        Debería mostrar una reducción significativa en fallecimientos e ingresos
        UCI después de la introducción de las vacunas
        """
        # TU CÓDIGO AQUÍ
        pass
        # Inicializar acumuladores
        totales = {
            'antes': {'ingresos': 0, 'fallecidos': 0, 'dias': 0},
            'despues': {'ingresos': 0, 'fallecidos': 0, 'dias': 0}
        }

        # Recorrer registros
        for r in self.registros:
            grupo = 'despues' if r.hay_vacuna else 'antes'
            totales[grupo]['ingresos'] += r.ingresos_uci
            totales[grupo]['fallecidos'] += r.fallecidos
            totales[grupo]['dias'] += 1

        # Calcular promedios
        def calcular_promedios(data):
            dias = data['dias'] or 1  # evitar división por cero
            return {
                'total_ingresos_uci': data['ingresos'],
                'total_fallecidos': data['fallecidos'],
                'promedio_ingresos': data['ingresos'] // dias,
                'promedio_fallecidos': data['fallecidos'] // dias
            }

        antes = calcular_promedios(totales['antes'])
        despues = calcular_promedios(totales['despues'])

        # Calcular reducción porcentual
        def reduccion(pasado, presente):
            if pasado == 0:
                return 0
            return round(100 * (pasado - presente) / pasado, 2)

        reduccion_porcentual = {
            'ingresos_uci': reduccion(antes['promedio_ingresos'], despues['promedio_ingresos']),
            'fallecidos': reduccion(antes['promedio_fallecidos'], despues['promedio_fallecidos'])
        }

        return {
            'antes_vacuna': antes,
            'despues_vacuna': despues,
            'reduccion_porcentual': reduccion_porcentual
        }

    # =========================================================================
    # MÉTODOS AUXILIARES (YA IMPLEMENTADOS)
    # =========================================================================

    def mostrar_registros(self, registros: List[RegistroCovid], limite: int = 10):
        """Muestra registros de forma legible"""
        print(f"\nMostrando {min(len(registros), limite)} de {len(registros)} registros:")
        print("-" * 80)
        for i, reg in enumerate(registros[:limite], 1):
            vacuna = "✓" if reg.hay_vacuna else "✗"
            conf = "🏠" if reg.hay_confinamiento else "  "
            print(f"{i:2d}. {reg.fecha:25s} | UCI: {reg.ingresos_uci:4d} | "
                  f"Fallecidos: {reg.fallecidos:4d} | Vacuna:{vacuna} | Conf:{conf}")
        print("-" * 80)


# =============================================================================
# PROGRAMA PRINCIPAL - PRUEBAS
# =============================================================================

def ejecutar_pruebas():
    """Ejecuta todas las pruebas del sistema"""

    print("\n" + "=" * 80)
    print("SISTEMA DE ANÁLISIS COVID-19 EN ESPAÑA")
    print("Pruebas de Algoritmos de Búsqueda y Ordenamiento")
    print("=" * 80)

    # Inicializar analizador
    analizador = AnalizadorCovid(r'C:\Users\tarde\Desktop\ProgramacionCursoIA\DATOS\2ingresos_en_uci_y_fallecidos_por_coronavirus_en_españa.csv')

    # -------------------------------------------------------------------------
    # PRUEBA 1: Búsqueda Lineal
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 1: BÚSQUEDA LINEAL - Buscar fecha específica")
    print("=" * 80)
    print("Buscando datos del 25 de marzo de 2020 (20200325)...")
    resultado = analizador.buscar_fecha_exacta(20200325)
    if resultado:
        print(f"✓ Encontrado: {resultado}")
    else:
        print("✗ No implementado o no encontrado")

    # -------------------------------------------------------------------------
    # PRUEBA 2: Búsqueda Binaria
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 2: BÚSQUEDA BINARIA - Primer día con al menos 500 fallecidos")
    print("=" * 80)
    resultado = analizador.buscar_primer_dia_con_fallecidos_minimos(500)
    if resultado:
        print(f"✓ Encontrado: {resultado}")
    else:
        print("✗ No implementado o no encontrado")

    # -------------------------------------------------------------------------
    # PRUEBA 3: Ordenamiento Burbuja
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 3: BUBBLE SORT - Ordenar por ingresos UCI")
    print("=" * 80)
    print("Ordenando todos los registros por ingresos en UCI...")
    registros_ordenados = analizador.ordenar_por_burbuja_ingresos_uci()
    if registros_ordenados:
        print("\n📊 Días con MENOS ingresos UCI:")
        analizador.mostrar_registros(registros_ordenados[:5], 5)
        print("\n📊 Días con MÁS ingresos UCI:")
        analizador.mostrar_registros(registros_ordenados[-5:], 5)
    else:
        print("✗ No implementado")

    # -------------------------------------------------------------------------
    # PRUEBA 4: QuickSort
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 4: QUICKSORT - Ordenar por fallecidos (descendente)")
    print("=" * 80)
    print("Ordenando por número de fallecidos...")
    registros_ordenados = analizador.quicksort_por_fallecidos(analizador.registros)
    if registros_ordenados:
        print("\n☠️  Los 10 días con MÁS fallecidos:")
        analizador.mostrar_registros(registros_ordenados[:10], 10)
    else:
        print("✗ No implementado")

    # -------------------------------------------------------------------------
    # PRUEBA 5: Búsqueda con Filtros
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 5: BÚSQUEDA CON FILTROS MÚLTIPLES")
    print("=" * 80)
    print("Buscando: Año 2021, con vacuna, más de 100 ingresos UCI...")
    resultados = analizador.buscar_con_condiciones(
        año=2021,
        con_vacuna=True,
        min_ingresos_uci=100
    )
    if resultados:
        print(f"✓ Encontrados {len(resultados)} registros que cumplen las condiciones:")
        analizador.mostrar_registros(resultados, 10)
    else:
        print("✗ No implementado o no hay resultados")

    # -------------------------------------------------------------------------
    # PRUEBA 6: Búsqueda de Picos
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 6: DETECCIÓN DE PICOS - Máximos locales de ingresos UCI")
    print("=" * 80)
    print("Buscando picos locales (ventana de 7 días)...")
    picos = analizador.encontrar_picos_ingresos_uci(ventana=7)
    if picos:
        print(f"✓ Encontrados {len(picos)} picos locales:")
        analizador.mostrar_registros(picos, 10)
    else:
        print("✗ No implementado o no hay picos")

    # -------------------------------------------------------------------------
    # PRUEBA 7: Top K Días Críticos
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 7: TOP 10 DÍAS MÁS CRÍTICOS")
    print("=" * 80)
    print("Calculando índice de criticidad: (UCI * 0.5) + (Fallecidos * 1.5)...")
    top_criticos = analizador.obtener_top_k_dias_criticos(k=10)
    if top_criticos:
        print("✓ Los 10 días más críticos de la pandemia:")
        analizador.mostrar_registros(top_criticos, 10)
    else:
        print("✗ No implementado")

    # -------------------------------------------------------------------------
    # PRUEBA 8: Detección de Tendencias
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 8: DETECCIÓN DE PERIODOS DE CRECIMIENTO SOSTENIDO")
    print("=" * 80)
    print("Buscando periodos de 7+ días consecutivos de crecimiento en UCI...")
    periodos = analizador.detectar_periodo_crecimiento_sostenido(dias_consecutivos=7)
    if periodos:
        print(f"✓ Encontrados {len(periodos)} periodos de crecimiento sostenido:")
        for i, (inicio, fin) in enumerate(periodos[:10], 1):
            print(f"  {i}. Del {inicio} al {fin}")
    else:
        print("✗ No implementado o no hay periodos")

    # -------------------------------------------------------------------------
    # PRUEBA 9: Estadísticas por Periodo
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 9: ESTADÍSTICAS DEL PERIODO DE CONFINAMIENTO 2020")
    print("=" * 80)
    stats = analizador.calcular_estadisticas_por_periodo(año=2020, con_confinamiento=True)
    if stats:
        print("✓ Estadísticas calculadas:")
        for clave, valor in stats.items():
            print(f"  {clave}: {valor}")
    else:
        print("✗ No implementado")

    # -------------------------------------------------------------------------
    # PRUEBA 10: Comparación Pre/Post Vacuna
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRUEBA 10: IMPACTO DE LA VACUNA - Comparación antes/después")
    print("=" * 80)
    comparacion = analizador.comparar_pre_post_vacuna()
    if comparacion:
        print("✓ Análisis comparativo:")
        print("\nANTES DE LA VACUNA:")
        for clave, valor in comparacion.get('antes_vacuna', {}).items():
            print(f"  {clave}: {valor}")
        print("\nDESPUÉS DE LA VACUNA:")
        for clave, valor in comparacion.get('despues_vacuna', {}).items():
            print(f"  {clave}: {valor}")
        print("\nREDUCCIÓN PORCENTUAL:")
        for clave, valor in comparacion.get('reduccion_porcentual', {}).items():
            print(f"  {clave}: {valor}%")
    else:
        print("✗ No implementado")

    print("\n" + "=" * 80)
    print("PRUEBAS COMPLETADAS")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    ejecutar_pruebas()
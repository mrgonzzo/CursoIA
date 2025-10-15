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
        # La Búsqueda Lineal recorre la lista secuencialmente:
        for registro in self.datos:

            # Comparamos el campo Fecha_num del registro actual con el valor buscado
            if registro.Fecha_num == fecha_num:
                print('registro')
                # Si hay coincidencia, devolvemos el registro y terminamos inmediatamente.
                return registro

        # Si el bucle termina, significa que no se encontró la fecha.
        return None

    # =============================================================================
    # PROGRAMA PRINCIPAL - PRUEBAS
    # =============================================================================

    def ejecutar_pruebas(self):
        """Ejecuta todas las pruebas del sistema"""

        print("\n" + "=" * 80)
        print("SISTEMA DE ANÁLISIS COVID-19 EN ESPAÑA")
        print("Pruebas de Algoritmos de Búsqueda y Ordenamiento")
        print("=" * 80)

        # Inicializar analizador
        analizador = AnalizadorCovid(r'C:/Users/tarde/Desktop/ProgramacionCursoIA/DATOS/covid_data.csv')

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

        # -------------------------------------------------------
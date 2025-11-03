"""
Calculador de rutas más cortas en el metro de Madrid usando algoritmo de Dijkstra
"""

from collections import defaultdict
import heapq


class RutaCalculator:
    def __init__(self, estaciones, conexiones, combo_conexiones):
        self.estaciones = estaciones
        self.grafo = self._construir_grafo(conexiones, combo_conexiones)
    
    def _construir_grafo(self, conexiones, combo_conexiones):
        """Construye el grafo del metro a partir de las conexiones"""
        grafo = defaultdict(list)
        
        # Agregar conexiones normales (bidireccionales)
        for est1, est2, tiempo in conexiones:
            grafo[est1].append((est2, tiempo))
            grafo[est2].append((est1, tiempo))
        
        # Agregar transbordos (conexiones internas)
        for est1, est2, tiempo in combo_conexiones:
            if est1 == est2:  # Transbordo interno (misma estación)
                # Ya está conectado por las líneas que pasan por ahí
                pass
        
        return grafo
    
    def calcular_ruta_corta(self, origen, destino):
        """
        Calcula la ruta más corta usando el algoritmo de Dijkstra
        Devuelve: (tiempo_total, ruta, estaciones_intermedias)
        """
        if origen not in self.estaciones:
            return None, [], []
        
        if destino not in self.estaciones:
            return None, [], []
        
        if origen == destino:
            return 0, [origen], []
        
        # Dijkstra
        distancias = {est: float('inf') for est in self.estaciones}
        distancias[origen] = 0
        previos = {}
        cola = [(0, origen)]
        visitados = set()
        
        while cola:
            tiempo_actual, estacion_actual = heapq.heappop(cola)
            
            if estacion_actual in visitados:
                continue
            
            visitados.add(estacion_actual)
            
            if estacion_actual == destino:
                break
            
            for vecino, tiempo_viaje in self.grafo.get(estacion_actual, []):
                nuevo_tiempo = tiempo_actual + tiempo_viaje
                
                if nuevo_tiempo < distancias[vecino]:
                    distancias[vecino] = nuevo_tiempo
                    previos[vecino] = estacion_actual
                    heapq.heappush(cola, (nuevo_tiempo, vecino))
        
        # Reconstruir la ruta
        if distancias[destino] == float('inf'):
            return None, [], []
        
        ruta = []
        actual = destino
        while actual is not None:
            ruta.append(actual)
            actual = previos.get(actual)
        
        ruta.reverse()
        
        # Identificar estaciones intermedias (sin origen ni destino)
        estaciones_intermedias = ruta[1:-1] if len(ruta) > 2 else []
        
        return distancias[destino], ruta, estaciones_intermedias
    
    def obtener_lineas_estacion(self, estacion):
        """Devuelve las líneas que pasan por una estación"""
        if estacion in self.estaciones:
            return self.estaciones[estacion].get("lineas", [])
        return []


"""
Visualización del mapa del metro de Madrid con Folium
"""

import folium
from folium import plugins
from typing import List, Tuple, Optional


class VisualizadorMetro:
    def __init__(self, estaciones, centro_lat=40.4168, centro_lon=-3.7038):
        """
        Inicializa el visualizador con las estaciones
        
        Args:
            estaciones: Diccionario con información de estaciones
            centro_lat: Latitud del centro del mapa (Madrid)
            centro_lon: Longitud del centro del mapa (Madrid)
        """
        self.estaciones = estaciones
        self.mapa = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Agregar marcadores de todas las estaciones
        self._agregar_estaciones()
    
    def _agregar_estaciones(self):
        """Agrega marcadores para todas las estaciones del metro"""
        for nombre, info in self.estaciones.items():
            lat, lon = info["lat"], info["lon"]
            lineas = ", ".join([f"L{num}" for num in info["lineas"]])
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                popup=f"<b>{nombre}</b><br>Líneas: {lineas}",
                tooltip=nombre,
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.7,
                weight=1
            ).add_to(self.mapa)
    
    def mostrar_ruta(self, ruta: List[str], titulo: str = "Ruta Calculada"):
        """
        Muestra una ruta en el mapa
        
        Args:
            ruta: Lista de nombres de estaciones en orden
            titulo: Título de la ruta
        """
        if not ruta:
            return
        
        # Obtener coordenadas de la ruta
        coordenadas = []
        for estacion in ruta:
            if estacion in self.estaciones:
                lat, lon = self.estaciones[estacion]["lat"], self.estaciones[estacion]["lon"]
                coordenadas.append([lat, lon])
        
        if not coordenadas:
            return
        
        # Agregar línea de la ruta
        folium.PolyLine(
            coordenadas,
            color='red',
            weight=4,
            opacity=0.8,
            popup=titulo,
            tooltip=titulo
        ).add_to(self.mapa)
        
        # Resaltar estación origen
        if ruta[0] in self.estaciones:
            origen = self.estaciones[ruta[0]]
            folium.Marker(
                location=[origen["lat"], origen["lon"]],
                popup=f"<b>ORIGEN: {ruta[0]}</b>",
                tooltip=f"Origen: {ruta[0]}",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(self.mapa)
        
        # Resaltar estación destino
        if ruta[-1] in self.estaciones:
            destino = self.estaciones[ruta[-1]]
            folium.Marker(
                location=[destino["lat"], destino["lon"]],
                popup=f"<b>DESTINO: {ruta[-1]}</b>",
                tooltip=f"Destino: {ruta[-1]}",
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(self.mapa)
        
        # Resaltar estaciones intermedias
        for estacion in ruta[1:-1]:
            if estacion in self.estaciones:
                inter = self.estaciones[estacion]
                folium.CircleMarker(
                    location=[inter["lat"], inter["lon"]],
                    radius=6,
                    popup=f"<b>{estacion}</b> (Estación Intermedia)",
                    tooltip=estacion,
                    color='orange',
                    fillColor='orange',
                    fillOpacity=0.8,
                    weight=2
                ).add_to(self.mapa)
    
    def agregar_marcador_clima(self, lat: float, lon: float, info_clima: dict):
        """
        Agrega un marcador con información meteorológica
        
        Args:
            lat: Latitud
            lon: Longitud
            info_clima: Diccionario con información del clima
        """
        if not info_clima:
            return
        
        # Crear HTML con información del clima
        html_clima = f"""
        <div style="width: 200px;">
            <h4>🌤️ Clima Actual</h4>
            <p><b>Ciudad:</b> {info_clima.get('ciudad', 'N/A')}</p>
            <p><b>Estado:</b> {info_clima.get('descripcion', 'N/A')}</p>
            <p><b>Temperatura:</b> {info_clima.get('temperatura', 0):.1f}°C</p>
            <p><b>Sensación:</b> {info_clima.get('sensacion_termica', 0):.1f}°C</p>
            <p><b>Humedad:</b> {info_clima.get('humedad', 0)}%</p>
            <p><b>Viento:</b> {info_clima.get('viento_velocidad', 0):.1f} m/s</p>
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(html_clima, max_width=250),
            tooltip="Información Meteorológica",
            icon=folium.Icon(color='lightblue', icon='cloud', prefix='fa')
        ).add_to(self.mapa)
    
    def guardar_mapa(self, ruta_archivo: str):
        """
        Guarda el mapa en un archivo HTML
        
        Args:
            ruta_archivo: Ruta donde guardar el archivo HTML
        """
        self.mapa.save(ruta_archivo)
        print(f"Mapa guardado en: {ruta_archivo}")


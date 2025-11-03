"""
Aplicación web Flask para el calculador de rutas del metro de Madrid
"""

from flask import Flask, render_template, request, jsonify
from metro_data import ESTACIONES, obtener_lista_estaciones, obtener_coordenadas
from metro_data import CONEXIONES, COMBO_CONEXIONES
from ruta_calculator import RutaCalculator
from weather_api import WeatherAPI
from visualizador import VisualizadorMetro
import json
import os

app = Flask(__name__)

# Inicializar componentes
calculator = RutaCalculator(ESTACIONES, CONEXIONES, COMBO_CONEXIONES)
weather_api = WeatherAPI(api_key=None, usar_wttr=True)  # Usar solo wttr.in


@app.route('/')
def index():
    """Página principal"""
    lista_estaciones = obtener_lista_estaciones()
    return render_template('index.html', estaciones=lista_estaciones)


@app.route('/calcular_ruta', methods=['POST'])
def calcular_ruta():
    """Calcula la ruta entre dos estaciones"""
    try:
        origen = request.json.get('origen')
        destino = request.json.get('destino')
        
        if not origen or not destino:
            return jsonify({'error': 'Debes seleccionar origen y destino'}), 400
        
        if origen == destino:
            return jsonify({'error': 'El origen y destino deben ser diferentes'}), 400
        
        # Calcular ruta
        tiempo_total, ruta, estaciones_intermedias = calculator.calcular_ruta_corta(origen, destino)
        
        if tiempo_total is None or not ruta:
            return jsonify({'error': 'No se encontró ruta entre las estaciones seleccionadas'}), 404
        
        # Obtener información de cada estación en la ruta
        ruta_detallada = []
        for i, estacion in enumerate(ruta):
            lineas = calculator.obtener_lineas_estacion(estacion)
            lat, lon = obtener_coordenadas(estacion)
            ruta_detallada.append({
                'nombre': estacion,
                'lineas': lineas,
                'lat': lat,
                'lon': lon,
                'tipo': 'origen' if i == 0 else 'destino' if i == len(ruta) - 1 else 'intermedia'
            })
        
        # Obtener clima en destino
        lat_destino, lon_destino = obtener_coordenadas(destino)
        clima_info = None
        if lat_destino and lon_destino:
            clima_info = weather_api.obtener_clima(lat_destino, lon_destino)
        
        # Generar mapa
        visualizador = VisualizadorMetro(ESTACIONES)
        visualizador.mostrar_ruta(ruta, f"Ruta: {origen} → {destino}")
        if clima_info:
            visualizador.agregar_marcador_clima(lat_destino, lon_destino, clima_info)
        
        # Obtener HTML del mapa
        mapa_html = visualizador.mapa._repr_html_()
        
        return jsonify({
            'exito': True,
            'tiempo_total': tiempo_total,
            'ruta': ruta_detallada,
            'clima': clima_info,
            'mapa_html': mapa_html,
            'num_estaciones': len(ruta)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error al calcular la ruta: {str(e)}'}), 500


@app.route('/api/estaciones', methods=['GET'])
def api_estaciones():
    """API para obtener lista de estaciones"""
    lista_estaciones = obtener_lista_estaciones()
    return jsonify({'estaciones': lista_estaciones})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


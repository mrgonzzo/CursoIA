"""
Aplicación principal para calcular rutas en el metro de Madrid
y obtener información meteorológica en la estación destino
"""

import os
from metro_data import ESTACIONES, obtener_lista_estaciones, obtener_coordenadas
from ruta_calculator import RutaCalculator
from weather_api import WeatherAPI
from visualizador import VisualizadorMetro


def main():
    print("=" * 60)
    print("🚇 METRO DE MADRID - CALCULADOR DE RUTAS CON IA 🌤️")
    print("=" * 60)
    print()
    
    # Verificar API key de OpenWeatherMap
    api_key = os.getenv("OPENWEATHER_API_KEY")
    usar_openweather = False
    
    if api_key:
        print("✅ API key de OpenWeatherMap encontrada")
        usar_openweather = True
    else:
        print("ℹ️  No se encontró API key de OpenWeatherMap")
        print("   Usando wttr.in (gratis, sin necesidad de API key)")
        print()
        opcion = input("¿Deseas usar OpenWeatherMap? (requiere API key) [s/n]: ").lower()
        if opcion == 's':
            api_key = input("Ingresa tu API key de OpenWeatherMap: ").strip()
            if api_key:
                usar_openweather = True
                print("✅ Usando OpenWeatherMap")
            else:
                print("⚠️  No se ingresó API key. Usando wttr.in (gratis)")
        else:
            print("✅ Usando wttr.in (sin API key)")
    
    # Inicializar componentes
    from metro_data import CONEXIONES, COMBO_CONEXIONES
    calculator = RutaCalculator(ESTACIONES, CONEXIONES, COMBO_CONEXIONES)
    
    # Inicializar API meteorológica (siempre habrá una, con o sin API key)
    weather_api = WeatherAPI(api_key=api_key if usar_openweather else None, usar_wttr=not usar_openweather)
    visualizador = VisualizadorMetro(ESTACIONES)
    
    # Obtener lista de estaciones
    lista_estaciones = obtener_lista_estaciones()
    
    print(f"\n📍 Estaciones disponibles: {len(lista_estaciones)} estaciones")
    print("   Ejemplos:", ", ".join(lista_estaciones[:10]) + "...")
    print()
    
    # Solicitar estación origen
    print("-" * 60)
    estacion_origen = input("🚉 Ingresa la estación de ORIGEN: ").strip()
    
    # Verificar si existe (búsqueda flexible)
    origen_encontrado = None
    for est in lista_estaciones:
        if est.lower() == estacion_origen.lower():
            origen_encontrado = est
            break
    
    if not origen_encontrado:
        print(f"❌ Estación '{estacion_origen}' no encontrada.")
        print("   Asegúrate de escribir el nombre exacto de la estación.")
        return
    
    print(f"✅ Origen seleccionado: {origen_encontrado}")
    
    # Solicitar estación destino
    print("-" * 60)
    estacion_destino = input("🎯 Ingresa la estación de DESTINO: ").strip()
    
    destino_encontrado = None
    for est in lista_estaciones:
        if est.lower() == estacion_destino.lower():
            destino_encontrado = est
            break
    
    if not destino_encontrado:
        print(f"❌ Estación '{estacion_destino}' no encontrada.")
        print("   Asegúrate de escribir el nombre exacto de la estación.")
        return
    
    print(f"✅ Destino seleccionado: {destino_encontrado}")
    
    if origen_encontrado == destino_encontrado:
        print("⚠️  La estación origen y destino son la misma.")
        return
    
    print()
    print("=" * 60)
    print("🔍 CALCULANDO RUTA...")
    print("=" * 60)
    
    # Calcular ruta
    tiempo_total, ruta, estaciones_intermedias = calculator.calcular_ruta_corta(
        origen_encontrado, destino_encontrado
    )
    
    if tiempo_total is None or not ruta:
        print("❌ No se pudo encontrar una ruta entre las estaciones seleccionadas.")
        return
    
    # Mostrar resultados
    print()
    print("✅ RUTA ENCONTRADA")
    print("-" * 60)
    print(f"⏱️  Tiempo estimado: {tiempo_total} minutos")
    print(f"🚇 Número de estaciones: {len(ruta)}")
    print()
    print("📍 Ruta completa:")
    for i, estacion in enumerate(ruta, 1):
        lineas = calculator.obtener_lineas_estacion(estacion)
        lineas_str = ", ".join([f"L{num}" for num in lineas]) if lineas else "N/A"
        if i == 1:
            print(f"   {i}. 🟢 {estacion} (Líneas: {lineas_str}) [ORIGEN]")
        elif i == len(ruta):
            print(f"   {i}. 🔴 {estacion} (Líneas: {lineas_str}) [DESTINO]")
        else:
            print(f"   {i}. ⚪ {estacion} (Líneas: {lineas_str})")
    
    # Obtener información meteorológica
    print()
    print("=" * 60)
    print("🌤️  INFORMACIÓN METEOROLÓGICA")
    print("=" * 60)
    
    # Obtener información meteorológica (siempre disponible, con o sin API key)
    lat_destino, lon_destino = obtener_coordenadas(destino_encontrado)
    if lat_destino and lon_destino:
        print(f"📍 Obteniendo clima para {destino_encontrado}...")
        clima_info = weather_api.obtener_clima(lat_destino, lon_destino)
        
        if clima_info:
            print(weather_api.formatear_clima_texto(clima_info))
            # Agregar marcador de clima al mapa
            visualizador.agregar_marcador_clima(lat_destino, lon_destino, clima_info)
        else:
            print("❌ No se pudo obtener información meteorológica.")
    else:
        print("❌ No se encontraron coordenadas para la estación destino.")
    
    # Visualizar ruta en mapa
    print()
    print("=" * 60)
    print("🗺️  GENERANDO MAPA...")
    print("=" * 60)
    
    visualizador.mostrar_ruta(ruta, f"Ruta: {origen_encontrado} → {destino_encontrado}")
    
    # Guardar mapa
    archivo_mapa = "ruta_metro_madrid.html"
    visualizador.guardar_mapa(archivo_mapa)
    
    print()
    print("=" * 60)
    print("✅ PROCESO COMPLETADO")
    print("=" * 60)
    print(f"📄 Mapa guardado en: {archivo_mapa}")
    print("   Abre el archivo HTML en tu navegador para ver la ruta visualizada.")
    print()


if __name__ == "__main__":
    main()


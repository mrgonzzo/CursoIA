# 🚇 Metro de Madrid - Calculador de Rutas con IA

Aplicación de Python que combina inteligencia artificial con datos geolocalizados para calcular la ruta más corta en el metro de Madrid y mostrar información meteorológica en la estación destino.

## 📋 Características

- ✅ Calcula la ruta más corta entre dos estaciones del metro de Madrid
- ✅ Visualización interactiva con Folium
- ✅ Información meteorológica en tiempo real usando OpenWeatherMap API
- ✅ Algoritmo de Dijkstra para encontrar la ruta óptima
- ✅ Interfaz de línea de comandos fácil de usar

## 🛠️ Instalación

### 1. Clonar o descargar el proyecto

El proyecto ya está en: `C:\Users\gonzzo\Desktop\CURSO\PycharmProjects\MadMetroCursor`

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. API Meteorológica (Opcional)

La aplicación soporta **dos opciones** para obtener información meteorológica:

#### Opción 1: wttr.in (Recomendado - Sin API key) ⭐
- ✅ **Completamente gratis**
- ✅ **No requiere registro ni API key**
- ✅ **Se usa automáticamente si no tienes API key de OpenWeatherMap**
- ✅ Sin límites significativos para uso personal

#### Opción 2: OpenWeatherMap (Con API key)
- ✅ Plan gratuito disponible
- ✅ Hasta 60 llamadas por minuto
- ✅ 1,000,000 llamadas por mes
- 📝 Requiere registro en [OpenWeatherMap](https://openweathermap.org/api)

Si deseas usar OpenWeatherMap, configura la variable de entorno:

**Windows:**
```cmd
set OPENWEATHER_API_KEY=tu_api_key_aqui
```

**Linux/Mac:**
```bash
export OPENWEATHER_API_KEY=tu_api_key_aqui
```

**Nota:** Si no configuras la API key, la aplicación usará automáticamente wttr.in (gratis y sin API key).

## 🚀 Uso

### Aplicación Web (Recomendado) 🌐

Ejecutar el servidor web:

```bash
python app.py
```

Luego abre tu navegador en: **http://localhost:5000**

En la interfaz web podrás:
- ✅ Seleccionar estación origen desde un menú desplegable
- ✅ Seleccionar estación destino desde un menú desplegable
- ✅ Ver la ruta calculada con tiempo estimado
- ✅ Ver información meteorológica en la estación destino
- ✅ Ver el mapa interactivo con la ruta visualizada

### Aplicación de Línea de Comandos (Alternativa)

También puedes usar la versión de consola:

```bash
python main.py
```

La aplicación te pedirá:
1. Estación de **origen**
2. Estación de **destino**

Luego calculará y mostrará los resultados.

## 📁 Estructura del Proyecto

```
MadMetroCursor/
├── main.py                 # Aplicación principal
├── metro_data.py           # Datos de estaciones y conexiones
├── ruta_calculator.py      # Algoritmo de cálculo de rutas
├── weather_api.py          # Integración con OpenWeatherMap
├── visualizador.py         # Visualización con Folium
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
└── ruta_metro_madrid.html  # Mapa generado (se crea al ejecutar)
```

## 🗺️ Datos del Metro

El proyecto incluye datos de las principales estaciones del metro de Madrid basados en el [plano oficial](https://www.metromadrid.es/sites/default/files/web/planos/PlanoMetro_Abr2025.pdf), incluyendo:

- Coordenadas geográficas de cada estación
- Líneas que pasan por cada estación
- Conexiones entre estaciones
- Tiempos estimados de viaje

## 🌤️ API Meteorológica

La aplicación utiliza **wttr.in**:
- ✅ **Completamente gratis y sin API key**
- ✅ No requiere registro
- ✅ Sin límites significativos para uso personal
- ✅ Funciona automáticamente sin configuración

## 📊 Ejemplo de Uso

```
🚇 METRO DE MADRID - CALCULADOR DE RUTAS CON IA 🌤️
============================================================

📍 Estaciones disponibles: 150+ estaciones
   Ejemplos: Sol, Gran Vía, Chamartín, Atocha Renfe...

------------------------------------------------------------
🚉 Ingresa la estación de ORIGEN: Sol
✅ Origen seleccionado: Sol

------------------------------------------------------------
🎯 Ingresa la estación de DESTINO: Chamartín
✅ Destino seleccionado: Chamartín

============================================================
🔍 CALCULANDO RUTA...
============================================================

✅ RUTA ENCONTRADA
------------------------------------------------------------
⏱️  Tiempo estimado: 15 minutos
🚇 Número de estaciones: 8

📍 Ruta completa:
   1. 🟢 Sol (Líneas: L1, L2, L3) [ORIGEN]
   2. ⚪ Gran Vía (Líneas: L1, L5)
   ...
   8. 🔴 Chamartín (Líneas: L1, L10) [DESTINO]

============================================================
🌤️  INFORMACIÓN METEOROLÓGICA
============================================================
📍 Obteniendo clima para Chamartín...

🌤️ **Condiciones Meteorológicas en Madrid**

📊 **Estado:** Cielo despejado
🌡️ **Temperatura:** 22.5°C
...
```

## 🔧 Personalización

### Agregar más estaciones

Editar `metro_data.py` para agregar más estaciones con sus coordenadas y conexiones.

### Modificar tiempos de viaje

Ajustar los tiempos en las `CONEXIONES` del archivo `metro_data.py`.

## 📝 Notas

- El mapa se guarda como `ruta_metro_madrid.html` en el directorio del proyecto
- Las coordenadas de las estaciones son aproximadas
- Los tiempos de viaje son estimados

## 📄 Licencia

Este proyecto es de uso educativo y personal.

## 🤝 Contribuciones

Siéntete libre de mejorar el proyecto agregando más estaciones, mejorando el algoritmo o añadiendo nuevas funcionalidades.


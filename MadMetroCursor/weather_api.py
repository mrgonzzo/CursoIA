"""
Integración con APIs meteorológicas para obtener datos del clima
Soporta OpenWeatherMap (requiere API key) y wttr.in (sin API key)
"""

import requests
from typing import Optional, Dict


class WeatherAPI:
    def __init__(self, api_key: Optional[str] = None, usar_wttr: bool = True):
        """
        Inicializa la API meteorológica (usa wttr.in por defecto)
        
        Args:
            api_key: Clave API de OpenWeatherMap (obsoleto, ahora solo usa wttr.in)
            usar_wttr: Siempre True, usa wttr.in sin necesidad de API key
        """
        self.usar_wttr = True  # Siempre usar wttr.in
        self.base_url_wttr = "https://wttr.in"
    
    def obtener_clima(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Obtiene el clima actual para unas coordenadas geográficas usando wttr.in
        
        Args:
            lat: Latitud
            lon: Longitud
        
        Returns:
            Diccionario con información del clima o None si hay error
        """
        return self._obtener_clima_wttr(lat, lon)
    
    def _obtener_clima_wttr(self, lat: float, lon: float) -> Optional[Dict]:
        """Obtiene el clima usando wttr.in (NO requiere API key)"""
        try:
            # wttr.in acepta coordenadas directamente
            url = f"{self.base_url_wttr}/{lat},{lon}?format=j1&lang=es"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get("current_condition", [{}])[0]
            
            # Convertir unidades
            temp_c = float(current.get("temp_C", 0))
            feels_like_c = float(current.get("FeelsLikeC", temp_c))
            viento_kmh = float(current.get("windspeedKmph", 0))
            viento_ms = viento_kmh / 3.6  # Convertir km/h a m/s
            
            clima_info = {
                "temperatura": temp_c,
                "sensacion_termica": feels_like_c,
                "humedad": int(current.get("humidity", 0)),
                "presion": float(current.get("pressure", 0)) / 10,  # Convertir a hPa
                "descripcion": current.get("lang_es", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "N/A")),
                "icono": current.get("weatherCode", ""),
                "viento_velocidad": round(viento_ms, 1),
                "viento_direccion": int(current.get("winddirDegree", 0)),
                "visibilidad": float(current.get("visibility", 0)) / 1000,
                "ciudad": "Madrid",  # wttr.in no devuelve nombre de ciudad por coordenadas
                "pais": "ES"
            }
            
            return clima_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con wttr.in: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error al procesar la respuesta de wttr.in: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado con wttr.in: {e}")
            return None
    
    def formatear_clima_texto(self, clima_info: Dict) -> str:
        """
        Formatea la información del clima como texto legible
        
        Args:
            clima_info: Diccionario con información del clima
        
        Returns:
            String formateado con la información meteorológica
        """
        if not clima_info:
            return "No se pudo obtener información meteorológica."
        
        texto = f"""
🌤️ **Condiciones Meteorológicas en {clima_info['ciudad']}**

📊 **Estado:** {clima_info['descripcion']}
🌡️ **Temperatura:** {clima_info['temperatura']:.1f}°C
❄️ **Sensación Térmica:** {clima_info['sensacion_termica']:.1f}°C
💧 **Humedad:** {clima_info['humedad']}%
📈 **Presión:** {clima_info['presion']} hPa
💨 **Viento:** {clima_info['viento_velocidad']:.1f} m/s
👁️ **Visibilidad:** {clima_info['visibilidad']:.1f} km
        """
        
        return texto.strip()


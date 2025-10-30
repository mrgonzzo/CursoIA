"""
UNIDAD 7 - SISTEMAS BASADOS EN CONOCIMIENTO
Ejemplo 3 (V2): Sistema de Búsqueda de Casos Similares (CFDR) + API Flask

Objetivo:
- Permitir al usuario ingresar síntomas (hechos).
- Encontrar el caso conocido (de cfdr_metrics.json) más similar.
- Devolver el diagnóstico del caso conocido para proveer una solución contextual.

INSTALACIÓN:
pip install flask
"""

from flask import Flask, request, jsonify, render_template_string
import json
import os
import math # Necesario para calcular la "distancia" entre hechos

print("=" * 80)
print("UNIDAD 7 - EJEMPLO 3 (V2): Diagnóstico CFDR basado en Casos Similares")
print("Sistema experto con Búsqueda Contextual")
print("=" * 80)

# ==============================================================================
# 1. FUNCIÓN DE CARGA DE DATOS (Sin cambios)
# ==============================================================================

def cargar_datos_cfdr(archivo="cfdr_metrics.json"):
    """Carga los hechos de fallos simulados desde un archivo JSON, usando UTF-8."""
    if not os.path.exists(archivo):
        print(f"\n[ERROR FATAL] Archivo de datos '{archivo}' no encontrado.")
        return {}
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            datos_lista = json.load(f)
            # Convertimos la lista a un diccionario {caso_id: datos}
            datos_dict = {str(d['caso']): d for d in datos_lista}
            print(f"[INFO] {len(datos_dict)} casos de CFDR cargados exitosamente desde {archivo}.")
            return datos_dict
    except json.JSONDecodeError as e:
        print(f"\n[ERROR ESTRUCTURA JSON] El archivo '{archivo}' tiene un error de formato JSON: {e}")
        return {}
    except Exception as e:
        print(f"\n[ERROR GENÉRICO AL CARGAR] Ocurrió un error inesperado: {e}")
        return {}

# ==============================================================================
# 2. SISTEMA EXPERTO DE DIAGNÓSTICO (CFDR) - MODIFICADO
# ==============================================================================

class SistemaExpertoCFDR:
    """Sistema experto para diagnóstico de fallos de hardware/nodo."""

    def __init__(self, hechos_catalogo):
        # ... (Reglas y atributos iniciales sin cambios)
        self.hechos = {}
        self.diagnosticos = []
        self.reglas = self._definir_reglas()
        self.catalogo_casos = hechos_catalogo

        # Definición de pesos para la búsqueda de similitud (ajustar según importancia)
        self.pesos = {
            'temperatura_cpu': 5.0,
            'perdida_paquetes': 3.0,
            'latencia_red': 2.0,
            'errores_ecc': 4.0,
            'uso_disco_raiz': 1.0
        }

    def _definir_reglas(self):
        # ... (Mantenemos las reglas R1 a R5 sin cambios)
        return [
            {'id': 'R1_SOBRECALENTAMIENTO', 'condiciones': {'temperatura_cpu': '>90', 'estado_ventilador': 'lento'}, 'conclusion': ('Fallo Crítico: CPU (Hardware)', 0.95), 'descripcion': 'Si Temp CPU > 90°C Y Ventilador Lento → Sobrecalentamiento crítico.'},
            {'id': 'R2_FALLO_RED_NODO', 'condiciones': {'perdida_paquetes': '>50', 'latencia_red': '>500'}, 'conclusion': ('Fallo de Nodo: Red/NIC', 0.90), 'descripcion': 'Si Pérdida Paquetes > 50% Y Latencia > 500ms → Problema de conectividad de Nodo.'},
            {'id': 'R3_FALLO_DISCO', 'condiciones': {'smart_status': 'error', 'sectores_reubicados': '>100'}, 'conclusion': ('Fallo de Hardware: Disco (HDD/SSD)', 0.85), 'descripcion': 'Si S.M.A.R.T. es "error" O Sectores Reubicados > 100 → Fallo inminente de Disco.'},
            {'id': 'R4_FALLO_RAM', 'condiciones': {'errores_ecc': '>=5', 'tiempo_actividad': '>24'}, 'conclusion': ('Fallo de Hardware: Memoria RAM', 0.80), 'descripcion': 'Si Errores ECC >= 5 Y Tiempo Actividad > 24h → RAM defectuosa.'},
            {'id': 'R5_DISCO_LLENO', 'condiciones': {'uso_disco_raiz': '>95', 'tipo_sistema': 'produccion'}, 'conclusion': ('Advertencia: Baja Capacidad de Servidor', 0.75), 'descripcion': 'Si Uso Disco > 95% Y Entorno Producción → Advertencia de espacio crítico.'}
        ]

    def set_hechos(self, hechos):
        self.hechos = hechos

    def evaluar_regla(self, regla):
        # ... (Lógica de evaluación de reglas sin cambios)
        # Se mantiene la lógica anterior para evitar errores de indentación.
        if regla['id'] == 'R3_FALLO_DISCO':
            smart_err = self.hechos.get('smart_status') == 'error'
            sectores_err = self.hechos.get('sectores_reubicados', 0) > 100
            return smart_err or sectores_err

        condiciones_cumplidas = 0
        for condicion, valor_esperado in regla['condiciones'].items():
            if isinstance(valor_esperado, str) and (valor_esperado.startswith('>') or valor_esperado.startswith('>=')):
                valor_actual_num = float(self.hechos.get(condicion, 0))

                if valor_esperado.startswith('>='):
                    umbral = float(valor_esperado[2:])
                    if valor_actual_num >= umbral: condiciones_cumplidas += 1
                elif valor_esperado.startswith('>'):
                    umbral = float(valor_esperado[1:])
                    if valor_actual_num > umbral: condiciones_cumplidas += 1
            else:
                if self.hechos.get(condicion) == valor_esperado:
                    condiciones_cumplidas += 1

        return condiciones_cumplidas == len(regla['condiciones'])


    def diagnosticar_caso(self, caso_data):
        """Ejecuta el diagnóstico para un caso específico (hechos pasados)."""
        self.diagnosticos = []
        self.set_hechos(caso_data['datos'])

        for regla in self.reglas:
            if self.evaluar_regla(regla):
                diagnostico, certeza = regla['conclusion']
                self.diagnosticos.append({
                    'diagnostico': diagnostico,
                    'certeza': certeza,
                    'regla_id': regla['id']
                })

        self.diagnosticos.sort(key=lambda x: x['certeza'], reverse=True)
        return self.diagnosticos

    def encontrar_caso_mas_cercano(self, hechos_usuario):
        """
        Encuentra el caso del catálogo más similar al perfil de hechos del usuario.
        Utiliza una "distancia" ponderada basada en métricas numéricas.
        """
        min_distancia = float('inf')
        caso_cercano = None

        for caso_id, caso_data in self.catalogo_casos.items():
            distancia_total = 0.0

            # Calculamos la distancia Euclídea Ponderada (solo en métricas numéricas)
            for key, peso in self.pesos.items():
                val_usuario = hechos_usuario.get(key, 0)
                val_catalogo = caso_data['datos'].get(key, 0)

                try:
                    # Distancia cuadrática ponderada
                    distancia_total += peso * (float(val_usuario) - float(val_catalogo))**2
                except:
                    # Ignorar si no son convertibles a float (ej: 'normal', 'produccion')
                    pass

            # Añadir penalización por diferencias categóricas (ej: ventilador, smart_status)
            if hechos_usuario.get('estado_ventilador') != caso_data['datos'].get('estado_ventilador'):
                 distancia_total += 100 # Gran penalización
            if hechos_usuario.get('smart_status') != caso_data['datos'].get('smart_status'):
                 distancia_total += 50

            distancia_final = math.sqrt(distancia_total)

            if distancia_final < min_distancia:
                min_distancia = distancia_final
                caso_cercano = caso_data

        return caso_cercano

# ==============================================================================
# 3. API FLASK - MODIFICADO
# ==============================================================================

# Cargar el catálogo de fallos al iniciar el script
catalogo_cfdr = cargar_datos_cfdr()
diagnostico_experto = SistemaExpertoCFDR(catalogo_cfdr)
app = Flask(__name__)

# Página principal con formulario (HTML MODIFICADO)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Diagnóstico por Similitud de Casos CFDR</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background-color: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #8B0000; border-bottom: 3px solid #8B0000; padding-bottom: 10px; }
        .form-row { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
        .form-group { flex: 1 1 45%; min-width: 250px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="number"], select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        button { background-color: #8B0000; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; margin-top: 10px; }
        button:hover { background-color: #6a0000; }
        #resultado { margin-top: 30px; border-top: 2px solid #ccc; padding-top: 20px;}
        .fallo { background: #fee; padding: 15px; margin: 10px 0; border-left: 4px solid #8B0000; border-radius: 5px; }
        .alerta { background: #ffc; border-left: 4px solid #FFD700; }
        .similitud { background: #e6f7ff; padding: 15px; border-left: 4px solid #007bff; border-radius: 5px; margin-bottom: 20px;}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Diagnóstico por Similitud de Casos (CFDR)</h1>
        <p>Introduce las métricas de tu servidor para encontrar el fallo más parecido en la BBDD CFDR conocida y su diagnóstico.</p>

        <form id="formulario">
            <div class="form-row">
                <div class="form-group">
                    <label>Temperatura CPU (°C):</label>
                    <input type="number" id="temperatura_cpu" min="30" max="100" value="95" required>
                </div>
                <div class="form-group">
                    <label>Estado Ventilador:</label>
                    <select id="estado_ventilador" required>
                        <option value="normal">Normal</option>
                        <option value="lento">Lento/Fallo</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Pérdida Paquetes (%):</label>
                    <input type="number" id="perdida_paquetes" min="0" max="100" value="5" required>
                </div>
                <div class="form-group">
                    <label>Latencia Red (ms):</label>
                    <input type="number" id="latencia_red" min="1" max="1000" value="50" required>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Errores ECC (Count):</label>
                    <input type="number" id="errores_ecc" min="0" max="10" value="1" required>
                </div>
                <div class="form-group">
                    <label>S.M.A.R.T. Status:</label>
                    <select id="smart_status" required>
                        <option value="ok">OK</option>
                        <option value="error">Error/Warning</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Uso Disco Raíz (%):</label>
                    <input type="number" id="uso_disco_raiz" min="10" max="100" value="90" required>
                </div>
                <div class="form-group">
                    <label>Tipo de Sistema:</label>
                    <select id="tipo_sistema" required>
                        <option value="produccion">Producción</option>
                        <option value="desarrollo">Desarrollo</option>
                    </select>
                </div>
                <input type="hidden" id="tiempo_actividad" value="100"> 
                <input type="hidden" id="sectores_reubicados" value="0">
            </div>

            <button type="submit">Encontrar Caso Similar y Diagnóstico</button>
        </form>

        <div id="resultado"></div>
    </div>

    <script>
        document.getElementById('formulario').addEventListener('submit', async function(e) {
            e.preventDefault();

            const hechos = {
                temperatura_cpu: parseFloat(document.getElementById('temperatura_cpu').value),
                estado_ventilador: document.getElementById('estado_ventilador').value,
                perdida_paquetes: parseFloat(document.getElementById('perdida_paquetes').value),
                latencia_red: parseFloat(document.getElementById('latencia_red').value),
                errores_ecc: parseInt(document.getElementById('errores_ecc').value),
                smart_status: document.getElementById('smart_status').value,
                uso_disco_raiz: parseFloat(document.getElementById('uso_disco_raiz').value),
                tipo_sistema: document.getElementById('tipo_sistema').value,
                // Valores que no preguntamos, pero necesita el SE
                tiempo_actividad: parseInt(document.getElementById('tiempo_actividad').value),
                sectores_reubicados: parseInt(document.getElementById('sectores_reubicados').value)
            };

            const response = await fetch('/api/buscarsimilar', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ hechos_usuario: hechos })
            });

            const data = await response.json();
            mostrarResultado(data);
        });

        function mostrarResultado(data) {
            const resultadoDiv = document.getElementById('resultado');
            let html = '<h2>Resultado de la Búsqueda:</h2>';
            
            if (data.error) {
                resultadoDiv.innerHTML = `<div class="fallo">Error: ${data.error}</div>`;
                return;
            }

            html += `
                <div class="similitud">
                    <p>¡TE PASA COMO A! El sistema experto ha encontrado el caso más similar.</p>
                    <h3>Servidor Similar: ${data.nombre_caso_cercano} (Caso ${data.caso_id_cercano})</h3>
                </div>
            `;
            
            // Mostrar diagnósticos del caso encontrado
            if (data.diagnosticos.length === 0) {
                html += '<div class="info">⚠️ El diagnóstico del caso similar no fue concluyente.</div>';
            } else {
                html += '<h3>Diagnóstico de ese caso conocido:</h3>';
                data.diagnosticos.forEach((diag, i) => {
                    const clase = diag.diagnostico.includes('Crítico') || diag.diagnostico.includes('Fallo de Hardware') ? 'fallo' : 'alerta';
                    html += `
                        <div class="${clase}">
                            <h4>${i + 1}. ${diag.diagnostico} (Certeza: ${diag.certeza*100}%)</h4>
                            <p>Regla Aplicada: ${diag.regla_id}</p>
                        </div>
                    `;
                });
            }

            resultadoDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Página principal con formulario de entrada de hechos"""
    # El catálogo ya no se usa para llenar el selector, pero pasamos un JSON vacío
    return render_template_string(HTML_TEMPLATE, catalogo_json='{}')

@app.route('/api/buscarsimilar', methods=['POST'])
def api_buscarsimilar():
    """Endpoint API para buscar el caso más similar y devolver su diagnóstico."""
    data = request.json
    hechos_usuario = data.get('hechos_usuario')

    if not hechos_usuario:
        return jsonify({'error': 'Faltan los hechos (métricas) del servidor.'}), 400

    # 1. Encontrar el caso más cercano en el catálogo
    caso_cercano = diagnostico_experto.encontrar_caso_mas_cercano(hechos_usuario)

    if not caso_cercano:
        return jsonify({'error': 'No se pudo encontrar un caso similar en la BBDD CFDR.'}), 404

    # 2. Obtener el diagnóstico del caso más cercano
    diagnosticos = diagnostico_experto.diagnosticar_caso(caso_cercano)

    return jsonify({
        'caso_id_cercano': caso_cercano['caso'],
        'nombre_caso_cercano': caso_cercano['nombre'],
        'hechos_caso_cercano': caso_cercano['datos'],
        'diagnosticos': diagnosticos
    })

# Mantenemos el endpoint de catálogo para referencia, aunque no se use directamente en el index
@app.route('/api/catalogo', methods=['GET'])
def api_catalogo():
    """Endpoint para listar todos los casos disponibles en el CFDR simulado"""
    return jsonify({'casos_disponibles': diagnostico_experto.catalogo_casos})

# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================

if __name__ == '__main__':

    # Instrucciones de inicio
    print("\n" + "=" * 80)
    print("SERVIDOR WEB FLASK - INSTRUCCIONES")
    print("=" * 80)
    print("Para iniciar el diagnóstico interactivo por similitud:")
    print("1. Descomenta la línea `app.run(...)` al final del script.")
    print("2. Ejecuta el archivo.")
    print("3. Abre tu navegador en: http://localhost:5000")
    print("\nPara detener el servidor: Ctrl+C")

    # Descomentar la siguiente línea para iniciar Flask:
    # app.run(debug=True, port=5000)
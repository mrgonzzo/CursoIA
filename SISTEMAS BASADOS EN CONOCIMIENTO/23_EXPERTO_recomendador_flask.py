"""
UNIDAD 7 - SISTEMAS BASADOS EN CONOCIMIENTO
Ejemplo 2: Sistema Recomendador con Reglas + API Flask

Este ejemplo muestra:
- Sistema experto para recomendar productos
- Reglas de negocio explícitas
- API REST con Flask (fácil de usar)
- Comparación con sistemas de recomendación basados en ML

INSTALACIÓN:
pip install flask
"""

from flask import Flask, request, jsonify, render_template_string
import json

print("=" * 80)
print("UNIDAD 7 - EJEMPLO 2: Sistema Recomendador con Reglas + Flask API")
print("Sistema experto con interfaz web")
print("=" * 80)

# ==============================================================================
# 1. SISTEMA EXPERTO DE RECOMENDACIÓN
# ==============================================================================

class RecomendadorProductos:
    """Sistema experto para recomendar laptops basado en necesidades del usuario"""

    def __init__(self):
        # Catálogo de productos
        self.productos = [
            {
                'id': 1,
                'nombre': 'Laptop Básica Office',
                'precio': 500,
                'tipo': 'oficina',
                'ram': 8,
                'procesador': 'i3',
                'almacenamiento': 256,
                'peso_kg': 1.8,
                'grafica': 'integrada'
            },
            {
                'id': 2,
                'nombre': 'Laptop Gamer Pro',
                'precio': 1500,
                'tipo': 'gaming',
                'ram': 16,
                'procesador': 'i7',
                'almacenamiento': 512,
                'peso_kg': 2.5,
                'grafica': 'dedicada'
            },
            {
                'id': 3,
                'nombre': 'Laptop Diseño Gráfico',
                'precio': 2000,
                'tipo': 'diseño',
                'ram': 32,
                'procesador': 'i9',
                'almacenamiento': 1024,
                'peso_kg': 2.0,
                'grafica': 'dedicada'
            },
            {
                'id': 4,
                'nombre': 'Ultrabook Portátil',
                'precio': 1200,
                'tipo': 'portatil',
                'ram': 16,
                'procesador': 'i5',
                'almacenamiento': 512,
                'peso_kg': 1.2,
                'grafica': 'integrada'
            },
            {
                'id': 5,
                'nombre': 'Laptop Estudiante',
                'precio': 700,
                'tipo': 'estudiante',
                'ram': 8,
                'procesador': 'i5',
                'almacenamiento': 512,
                'peso_kg': 1.9,
                'grafica': 'integrada'
            }
        ]

        # Reglas de recomendación
        self.reglas = [
            {
                'id': 'R1',
                'descripcion': 'Si el usuario es diseñador → recomendar laptop con mucha RAM y gráfica dedicada',
                'condiciones': lambda perfil: perfil.get('uso') == 'diseño',
                'filtros': lambda p: p['ram'] >= 16 and p['grafica'] == 'dedicada',
                'prioridad': 10
            },
            {
                'id': 'R2',
                'descripcion': 'Si el usuario es gamer → recomendar laptop gaming',
                'condiciones': lambda perfil: perfil.get('uso') == 'gaming',
                'filtros': lambda p: p['tipo'] == 'gaming',
                'prioridad': 10
            },
            {
                'id': 'R3',
                'descripcion': 'Si necesita portabilidad → recomendar laptop ligera',
                'condiciones': lambda perfil: perfil.get('portabilidad') == 'alta',
                'filtros': lambda p: p['peso_kg'] < 1.5,
                'prioridad': 8
            },
            {
                'id': 'R4',
                'descripcion': 'Si presupuesto limitado → filtrar por precio',
                'condiciones': lambda perfil: perfil.get('presupuesto', 99999) < 1000,
                'filtros': lambda p: p['precio'] <= perfil.get('presupuesto', 99999),
                'prioridad': 9
            },
            {
                'id': 'R5',
                'descripcion': 'Si es estudiante → recomendar laptops económicas pero funcionales',
                'condiciones': lambda perfil: perfil.get('uso') == 'estudiante',
                'filtros': lambda p: p['tipo'] in ['estudiante', 'oficina'] and p['precio'] < 800,
                'prioridad': 7
            },
            {
                'id': 'R6',
                'descripcion': 'Si uso es oficina → recomendar laptops básicas',
                'condiciones': lambda perfil: perfil.get('uso') == 'oficina',
                'filtros': lambda p: p['tipo'] == 'oficina',
                'prioridad': 6
            }
        ]

    def recomendar(self, perfil_usuario):
        """
        Recomienda productos basándose en reglas

        perfil_usuario = {
            'uso': 'gaming' | 'diseño' | 'oficina' | 'estudiante',
            'presupuesto': int,
            'portabilidad': 'alta' | 'media' | 'baja'
        }
        """
        recomendaciones = []
        reglas_aplicadas = []

        # Evaluar cada regla
        for regla in self.reglas:
            if regla['condiciones'](perfil_usuario):
                # Regla aplicable
                reglas_aplicadas.append({
                    'id': regla['id'],
                    'descripcion': regla['descripcion'],
                    'prioridad': regla['prioridad']
                })

                # Filtrar productos que cumplen esta regla
                productos_filtrados = [p for p in self.productos if regla['filtros'](p)]

                for producto in productos_filtrados:
                    # Evitar duplicados
                    if not any(r['producto']['id'] == producto['id'] for r in recomendaciones):
                        recomendaciones.append({
                            'producto': producto,
                            'razon': regla['descripcion'],
                            'regla_id': regla['id'],
                            'prioridad': regla['prioridad']
                        })

        # Ordenar por prioridad
        recomendaciones.sort(key=lambda x: x['prioridad'], reverse=True)

        return {
            'recomendaciones': recomendaciones[:3],  # Top 3
            'reglas_aplicadas': reglas_aplicadas,
            'total_productos_evaluados': len(self.productos)
        }

# Crear instancia del recomendador
recomendador = RecomendadorProductos()

# ==============================================================================
# 2. API FLASK
# ==============================================================================

app = Flask(__name__)

# Página principal con formulario
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sistema Recomendador de Laptops</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .form-group {
            margin: 20px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        #resultado {
            margin-top: 30px;
        }
        .producto {
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
            border-radius: 5px;
        }
        .regla {
            background: #e3f2fd;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 14px;
        }
        .precio {
            color: #4CAF50;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖥️ Sistema Recomendador de Laptops</h1>
        <p>Sistema experto basado en reglas (IA Simbólica)</p>

        <form id="formulario">
            <div class="form-group">
                <label>Uso principal:</label>
                <select id="uso" required>
                    <option value="">Seleccione...</option>
                    <option value="oficina">Oficina / Trabajo</option>
                    <option value="estudiante">Estudiante</option>
                    <option value="gaming">Gaming / Juegos</option>
                    <option value="diseño">Diseño Gráfico / Video</option>
                </select>
            </div>

            <div class="form-group">
                <label>Presupuesto máximo (USD):</label>
                <input type="number" id="presupuesto" min="400" max="3000" value="1000" required>
            </div>

            <div class="form-group">
                <label>Importancia de portabilidad:</label>
                <select id="portabilidad" required>
                    <option value="baja">Baja (lo uso en casa)</option>
                    <option value="media">Media (ocasionalmente lo muevo)</option>
                    <option value="alta">Alta (siempre lo llevo conmigo)</option>
                </select>
            </div>

            <button type="submit">Obtener Recomendaciones</button>
        </form>

        <div id="resultado"></div>
    </div>

    <script>
        document.getElementById('formulario').addEventListener('submit', async function(e) {
            e.preventDefault();

            const perfil = {
                uso: document.getElementById('uso').value,
                presupuesto: parseInt(document.getElementById('presupuesto').value),
                portabilidad: document.getElementById('portabilidad').value
            };

            const response = await fetch('/api/recomendar', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(perfil)
            });

            const data = await response.json();
            mostrarResultado(data);
        });

        function mostrarResultado(data) {
            let html = '<h2>Recomendaciones:</h2>';

            if (data.recomendaciones.length === 0) {
                html += '<p>No se encontraron productos que cumplan con tus criterios.</p>';
            } else {
                data.recomendaciones.forEach((rec, i) => {
                    const p = rec.producto;
                    html += `
                        <div class="producto">
                            <h3>${i + 1}. ${p.nombre}</h3>
                            <p class="precio">$${p.precio}</p>
                            <p><b>Especificaciones:</b></p>
                            <ul>
                                <li>Procesador: ${p.procesador}</li>
                                <li>RAM: ${p.ram}GB</li>
                                <li>Almacenamiento: ${p.almacenamiento}GB</li>
                                <li>Peso: ${p.peso_kg}kg</li>
                                <li>Gráfica: ${p.grafica}</li>
                            </ul>
                            <div class="regla">
                                <b>Razón de recomendación:</b> ${rec.razon}
                            </div>
                        </div>
                    `;
                });
            }

            html += '<h3>Reglas aplicadas:</h3>';
            data.reglas_aplicadas.forEach(regla => {
                html += `<div class="regla">✓ ${regla.descripcion}</div>`;
            });

            document.getElementById('resultado').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Página principal con formulario"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recomendar', methods=['POST'])
def api_recomendar():
    """Endpoint API para obtener recomendaciones"""
    perfil = request.json
    resultado = recomendador.recomendar(perfil)
    return jsonify(resultado)

@app.route('/api/productos', methods=['GET'])
def api_productos():
    """Endpoint para listar todos los productos"""
    return jsonify({'productos': recomendador.productos})

# ==============================================================================
# 3. EJEMPLOS DE USO DESDE PYTHON (SIN WEB)
# ==============================================================================

def ejemplos_consola():
    """Ejemplos de uso del sistema sin Flask"""
    print("\n" + "=" * 80)
    print("EJEMPLOS DE RECOMENDACIÓN (Consola)")
    print("=" * 80)

    # Ejemplo 1: Estudiante con presupuesto limitado
    print("\nEjemplo 1: Estudiante con presupuesto limitado")
    print("-" * 60)
    perfil1 = {
        'uso': 'estudiante',
        'presupuesto': 800,
        'portabilidad': 'media'
    }
    resultado1 = recomendador.recomendar(perfil1)

    print(f"Perfil: {perfil1}")
    print(f"\nReglas aplicadas:")
    for regla in resultado1['reglas_aplicadas']:
        print(f"  ✓ {regla['id']}: {regla['descripcion']}")

    print(f"\nRecomendaciones:")
    for i, rec in enumerate(resultado1['recomendaciones'], 1):
        p = rec['producto']
        print(f"\n{i}. {p['nombre']} - ${p['precio']}")
        print(f"   Razón: {rec['razon']}")

    # Ejemplo 2: Diseñador gráfico con buen presupuesto
    print("\n" + "=" * 80)
    print("Ejemplo 2: Diseñador gráfico profesional")
    print("-" * 60)
    perfil2 = {
        'uso': 'diseño',
        'presupuesto': 2500,
        'portabilidad': 'baja'
    }
    resultado2 = recomendador.recomendar(perfil2)

    print(f"Perfil: {perfil2}")
    print(f"\nRecomendaciones:")
    for i, rec in enumerate(resultado2['recomendaciones'], 1):
        p = rec['producto']
        print(f"\n{i}. {p['nombre']} - ${p['precio']}")
        print(f"   Specs: RAM {p['ram']}GB, {p['procesador']}, Gráfica {p['grafica']}")

# ==============================================================================
# 4. COMPARACIÓN CON ML
# ==============================================================================

def comparacion_ml_vs_reglas():
    """Explicación de diferencias"""
    print("\n" + "=" * 80)
    print("COMPARACIÓN: SISTEMA DE REGLAS vs MACHINE LEARNING")
    print("=" * 80)

    print("""
ESTE SISTEMA (Reglas):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Reglas explícitas definidas por expertos
✓ 100% explicable (sabemos POR QUÉ se recomendó cada producto)
✓ No requiere datos históricos de compras
✓ Fácil de modificar reglas de negocio
✓ Predecible y controlable

✗ No aprende de preferencias de usuarios
✗ No captura patrones complejos
✗ Requiere mantener reglas manualmente

SISTEMA CON ML (ej. Filtrado Colaborativo):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Aprende de comportamiento de miles de usuarios
✓ Descubre patrones que humanos no ven
✓ Mejora con más datos
✓ Personalización automática

✗ Necesita miles de interacciones usuario-producto
✗ "Caja negra" - difícil explicar por qué
✗ Puede tener sesgos de los datos
✗ "Cold start problem" - no funciona con usuarios/productos nuevos

CUÁNDO USAR CADA UNO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLAS:
→ Pocos productos (< 1000)
→ Reglas de negocio claras
→ Necesidad de explicabilidad total
→ No hay datos históricos suficientes
→ Regulaciones exigen transparencia

MACHINE LEARNING:
→ Miles de productos y usuarios
→ Patrones complejos de comportamiento
→ Datos históricos abundantes
→ Personalización es crítica
→ Los usuarios aceptan "magia" sin explicación

HÍBRIDO (MEJOR OPCIÓN):
→ ML para recomendación inicial
→ Reglas para filtrar/validar
→ Ejemplo: ML sugiere productos, pero reglas aseguran:
   * No recomendar productos fuera de presupuesto
   * No recomendar productos prohibidos por edad
   * Aplicar promociones/descuentos según políticas
""")

# ==============================================================================
# 5. MAIN - EJECUTAR
# ==============================================================================

if __name__ == '__main__':
    # Mostrar ejemplos en consola
    ejemplos_consola()
    comparacion_ml_vs_reglas()

    # Instrucciones para Flask
    print("\n" + "=" * 80)
    print("SERVIDOR WEB FLASK")
    print("=" * 80)
    print("""
Para iniciar el servidor web:

1. Ejecuta este archivo:
   python 23_EXPERTO_recomendador_flask.py

2. Abre tu navegador en:
   http://localhost:5000

3. Usa el formulario para obtener recomendaciones interactivas

4. También puedes usar la API directamente:

   POST http://localhost:5000/api/recomendar
   Body: {"uso": "gaming", "presupuesto": 1500, "portabilidad": "baja"}

   GET http://localhost:5000/api/productos

Para detener el servidor: Ctrl+C
""")

    # Iniciar Flask (comentado por defecto, descomentar para usar)
    print("\n⚠️  Descomenta la línea siguiente para iniciar el servidor Flask")
    print("app.run(debug=True, port=5000)")

    # Descomentar la siguiente línea para iniciar Flask:
    # app.run(debug=True, port=5000)

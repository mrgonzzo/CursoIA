"""
UNIDAD 7 - SISTEMAS BASADOS EN CONOCIMIENTO
Ejemplo 1: Sistema Experto para Diagnóstico Médico

Este ejemplo muestra:
- Crear reglas if-then con lógica
- Sistema experto simple sin ML
- Diferencia entre IA simbólica (reglas) vs IA estadística (ML)
- Motor de inferencia básico

NO REQUIERE librería externa (solo Python estándar)
"""

print("=" * 80)
print("UNIDAD 7 - EJEMPLO 1: Sistema Experto de Diagnóstico Médico")
print("IA Simbólica: Conocimiento Explícito vs Machine Learning")
print("=" * 80)

# ==============================================================================
# 1. BASE DE CONOCIMIENTO (Reglas del experto)
# ==============================================================================

class SistemaExperto:
    """Sistema experto simple basado en reglas if-then"""

    def __init__(self):
        self.hechos = {}  # Base de hechos (síntomas del paciente)
        self.diagnosticos = []  # Diagnósticos posibles

        # Base de conocimiento: Reglas médicas
        self.reglas = [
            {
                'id': 'R1',
                'condiciones': {'fiebre': True, 'tos': True, 'congestion': True},
                'conclusion': ('Gripe común', 0.8),
                'descripcion': 'Si tiene fiebre Y tos Y congestión → Gripe (80% certeza)'
            },
            {
                'id': 'R2',
                'condiciones': {'fiebre': True, 'dolor_garganta': True, 'ganglios_inflamados': True},
                'conclusion': ('Faringitis estreptocócica', 0.9),
                'descripcion': 'Si tiene fiebre Y dolor garganta Y ganglios → Faringitis (90%)'
            },
            {
                'id': 'R3',
                'condiciones': {'fiebre': False, 'tos': True, 'congestion': True, 'duracion_dias': '>3'},
                'conclusion': ('Resfriado común', 0.7),
                'descripcion': 'Si NO fiebre Y tos Y congestión Y >3 días → Resfriado'
            },
            {
                'id': 'R4',
                'condiciones': {'fiebre': True, 'tos': True, 'dificultad_respiratoria': True},
                'conclusion': ('Neumonía (requiere atención médica)', 0.85),
                'descripcion': 'Si fiebre Y tos Y dificultad respirar → Neumonía ⚠️'
            },
            {
                'id': 'R5',
                'condiciones': {'fiebre': True, 'dolor_cabeza': True, 'nauseas': True, 'rigidez_cuello': True},
                'conclusion': ('Meningitis (EMERGENCIA MÉDICA)', 0.95),
                'descripcion': 'Si fiebre Y dolor cabeza Y náuseas Y rigidez cuello → Meningitis 🚨'
            },
            {
                'id': 'R6',
                'condiciones': {'fiebre': False, 'tos': False, 'estornudos': True, 'picor_ojos': True},
                'conclusion': ('Alergia estacional', 0.75),
                'descripcion': 'Si NO fiebre Y estornudos Y picor ojos → Alergia'
            }
        ]

    def agregar_hecho(self, sintoma, valor):
        """Añadir síntoma del paciente"""
        self.hechos[sintoma] = valor

    def evaluar_regla(self, regla):
        """Evalúa si una regla se cumple con los hechos actuales"""
        for condicion, valor_esperado in regla['condiciones'].items():
            # Manejar condiciones especiales como '>3'
            if isinstance(valor_esperado, str) and valor_esperado.startswith('>'):
                umbral = int(valor_esperado[1:])
                if condicion not in self.hechos or self.hechos.get(condicion, 0) <= umbral:
                    return False
            else:
                if self.hechos.get(condicion) != valor_esperado:
                    return False
        return True

    def inferir(self):
        """Motor de inferencia: evalúa todas las reglas"""
        self.diagnosticos = []

        print("\n" + "=" * 80)
        print("MOTOR DE INFERENCIA - Evaluando reglas...")
        print("=" * 80)

        for regla in self.reglas:
            print(f"\nEvaluando {regla['id']}: {regla['descripcion']}")

            if self.evaluar_regla(regla):
                diagnostico, certeza = regla['conclusion']
                self.diagnosticos.append((diagnostico, certeza, regla['id']))
                print(f"  ✓ REGLA ACTIVADA → {diagnostico} (Certeza: {certeza*100:.0f}%)")
            else:
                print(f"  ✗ Regla no aplicable (condiciones no cumplidas)")

        return self.diagnosticos

    def mostrar_diagnosticos(self):
        """Muestra diagnósticos ordenados por certeza"""
        if not self.diagnosticos:
            print("\n⚠️ No se pudo determinar un diagnóstico con las reglas disponibles.")
            print("   Recomendación: Consultar con médico para evaluación completa.")
            return

        print("\n" + "=" * 80)
        print("DIAGNÓSTICOS POSIBLES (ordenados por certeza)")
        print("=" * 80)

        diagnosticos_ordenados = sorted(self.diagnosticos, key=lambda x: x[1], reverse=True)

        for i, (diagnostico, certeza, regla_id) in enumerate(diagnosticos_ordenados, 1):
            print(f"\n{i}. {diagnostico}")
            print(f"   Certeza: {certeza*100:.0f}%")
            print(f"   Regla aplicada: {regla_id}")

            # Advertencia especial
            if 'EMERGENCIA' in diagnostico.upper():
                print("   🚨 ATENCIÓN: REQUIERE ATENCIÓN MÉDICA INMEDIATA")
            elif 'requiere atención' in diagnostico.lower():
                print("   ⚠️  ADVERTENCIA: Consultar médico pronto")

# ==============================================================================
# 2. CASO PRÁCTICO 1: Paciente con Gripe
# ==============================================================================

print("\n" + "=" * 80)
print("CASO PRÁCTICO 1: Paciente con síntomas de gripe")
print("=" * 80)

sistema1 = SistemaExperto()

# Síntomas del paciente
print("\nSíntomas del paciente:")
sintomas_caso1 = {
    'fiebre': True,
    'tos': True,
    'congestion': True,
    'dolor_garganta': False,
    'dificultad_respiratoria': False
}

for sintoma, valor in sintomas_caso1.items():
    sistema1.agregar_hecho(sintoma, valor)
    print(f"  - {sintoma}: {'Sí' if valor else 'No'}")

# Inferir diagnóstico
diagnosticos = sistema1.inferir()
sistema1.mostrar_diagnosticos()

# ==============================================================================
# 3. CASO PRÁCTICO 2: Paciente con síntomas de emergencia
# ==============================================================================

print("\n\n" + "=" * 80)
print("CASO PRÁCTICO 2: Paciente con síntomas graves")
print("=" * 80)

sistema2 = SistemaExperto()

print("\nSíntomas del paciente:")
sintomas_caso2 = {
    'fiebre': True,
    'dolor_cabeza': True,
    'nauseas': True,
    'rigidez_cuello': True,
    'tos': False
}

for sintoma, valor in sintomas_caso2.items():
    sistema2.agregar_hecho(sintoma, valor)
    print(f"  - {sintoma}: {'Sí' if valor else 'No'}")

diagnosticos = sistema2.inferir()
sistema2.mostrar_diagnosticos()

# ==============================================================================
# 4. COMPARACIÓN: SISTEMA EXPERTO vs MACHINE LEARNING
# ==============================================================================

print("\n\n" + "=" * 80)
print("COMPARACIÓN: SISTEMA EXPERTO vs MACHINE LEARNING")
print("=" * 80)

comparacion = """
┌─────────────────────────────────────────────────────────────────────┐
│                  SISTEMA EXPERTO (Reglas)                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Conocimiento explícito y comprensible                            │
│ ✓ No requiere datos de entrenamiento                               │
│ ✓ Explicable (sabemos QUÉ regla se aplicó)                         │
│ ✓ Fácil de modificar reglas                                        │
│ ✓ Certeza controlada por expertos                                  │
│                                                                     │
│ ✗ Requiere expertos para crear reglas                              │
│ ✗ No aprende de datos nuevos                                       │
│ ✗ Difícil escalar a muchas reglas                                  │
│ ✗ No maneja incertidumbre compleja                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING (ej. Logit, Random Forest)            │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Aprende patrones de miles de casos                               │
│ ✓ No requiere conocimiento de experto                              │
│ ✓ Maneja relaciones complejas                                      │
│ ✓ Mejora con más datos                                             │
│ ✓ Escala bien a muchas variables                                   │
│                                                                     │
│ ✗ Caja negra (difícil explicar decisiones)                         │
│ ✗ Requiere muchos datos etiquetados                                │
│ ✗ Sensible a sesgo en datos                                        │
│ ✗ Necesita reentrenamiento periódico                               │
└─────────────────────────────────────────────────────────────────────┘

¿CUÁNDO USAR CADA UNO?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SISTEMA EXPERTO:
→ Pocas reglas claras (< 100 reglas)
→ Conocimiento experto disponible
→ Necesidad de explicabilidad total
→ Dominios críticos (salud, seguridad)
→ Regulaciones exigen transparencia

MACHINE LEARNING:
→ Muchos datos disponibles (miles de casos)
→ Patrones complejos difíciles de explicitar
→ Relaciones no lineales entre variables
→ Necesidad de adaptación continua
→ Predicción más importante que explicación

HÍBRIDO (LO MEJOR DE AMBOS):
→ ML para predecir + Reglas para validar
→ Ejemplo: ML predice 90% neumonía, pero regla verifica
          que si >65 años + diabetes → ALERTA INMEDIATA
"""

print(comparacion)

# ==============================================================================
# 5. EJEMPLO DE SISTEMA HÍBRIDO (ML + Reglas)
# ==============================================================================

print("\n" + "=" * 80)
print("EJEMPLO DE SISTEMA HÍBRIDO")
print("=" * 80)

print("""
CASO: Sistema de Aprobación de Créditos

1. PREDICCIÓN (ML - Logit):
   modelo.predict(datos_cliente) → Probabilidad de default: 15%

2. VALIDACIÓN (Reglas de Negocio):
   IF probabilidad_default < 20% AND ingreso > $30,000 THEN
       IF tiene_deudas_activas = True THEN
           RECHAZAR (regla de riesgo)
       ELSE
           APROBAR
   ELSE
       RECHAZAR

3. RESULTADO:
   - ML dice: "Bajo riesgo (15%)"
   - Regla dice: "Pero tiene deudas → RECHAZAR"
   - Decisión final: RECHAZAR

VENTAJA:
✓ ML captura patrones complejos
✓ Reglas aseguran cumplimiento de políticas
✓ Explicable al cliente: "Se rechazó por deudas activas"
""")

# ==============================================================================
# 6. EJERCICIO PARA ALUMNOS
# ==============================================================================

print("\n" + "=" * 80)
print("EJERCICIO PROPUESTO PARA ALUMNOS")
print("=" * 80)

print("""
TAREA: Crear sistema experto para otro dominio

Opciones:
1. Diagnóstico de problemas de auto (síntomas → problema)
2. Recomendación de carrera universitaria (intereses → carrera)
3. Diagnóstico de problemas de computadora
4. Recomendación de inversión (perfil riesgo → producto)

Requisitos:
✓ Mínimo 5 reglas if-then
✓ Mínimo 2 casos de prueba
✓ Comparar con cómo lo haría ML
✓ Explicar cuándo usar reglas vs ML en ese dominio

Entregable:
- Código Python del sistema experto
- Documento explicando reglas
- Casos de prueba con resultados
""")

print("\n" + "=" * 80)
print("EJEMPLO COMPLETADO")
print("=" * 80)
print("""
CONCEPTOS APRENDIDOS:
✓ Sistemas basados en reglas if-then
✓ Base de conocimiento vs base de datos
✓ Motor de inferencia simple
✓ Diferencia IA simbólica vs estadística
✓ Cuándo usar reglas vs ML
✓ Sistemas híbridos (lo mejor de ambos)

PRÓXIMO EJEMPLO:
- Unidad 8: Motores de Inferencia (Forward/Backward chaining)
""")

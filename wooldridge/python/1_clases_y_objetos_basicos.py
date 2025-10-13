"""
SCRIPT 1: CLASES Y OBJETOS BÁSICOS EN PYTHON
=============================================

Este script introduce los conceptos fundamentales de clases y objetos en Python.
Aprenderás:
1. Cómo crear una clase
2. Cómo definir atributos
3. Cómo definir métodos
4. Cómo crear objetos (instancias)
5. Cómo acceder a atributos y métodos
"""

# ======================================================
# PARTE 1: DEFINICIÓN DE UNA CLASE BÁSICA
# ======================================================

# Una clase es como un plano o plantilla para crear objetos
# Se define con la palabra clave 'class' seguida del nombre (por convención en CamelCase)
class Persona:
    # El método __init__ es el constructor, se ejecuta automáticamente al crear un objeto
    # 'self' es una referencia a la instancia actual y debe ser el primer parámetro de cada método
    def __init__(self, nombre, edad):
        # Atributos de instancia: variables que pertenecen al objeto
        self.nombre = nombre  # Asignamos el parámetro 'nombre' al atributo 'self.nombre'
        self.edad = edad      # Asignamos el parámetro 'edad' al atributo 'self.edad'
        
    # Métodos de instancia: funciones que puede realizar el objeto
    def saludar(self):
        return f"Hola, me llamo {self.nombre} y tengo {self.edad} años."
    
    def cumplir_anios(self):
        # Modificamos el atributo edad
        self.edad += 1
        return f"¡Feliz cumpleaños! Ahora tengo {self.edad} años."

# ======================================================
# PARTE 2: CREACIÓN Y USO DE OBJETOS (INSTANCIAS)
# ======================================================

# Creamos un objeto (instancia) de la clase Persona
persona1 = Persona("Juan", 25)  # Llamamos al constructor con los argumentos

# Accedemos a los atributos usando la notación de punto
print(f"Nombre: {persona1.nombre}")  # Output: Nombre: Juan
print(f"Edad: {persona1.edad}")      # Output: Edad: 25

# Llamamos a los métodos del objeto
print(persona1.saludar())         # Output: Hola, me llamo Juan y tengo 25 años.
print(persona1.cumplir_anios())   # Output: ¡Feliz cumpleaños! Ahora tengo 26 años.

# Verificamos que el atributo edad ha cambiado
print(f"Nueva edad: {persona1.edad}")  # Output: Nueva edad: 26

# ======================================================
# PARTE 3: MÚLTIPLES INSTANCIAS
# ======================================================

# Podemos crear varias instancias de la misma clase
persona2 = Persona("María", 30)

# Cada instancia tiene sus propios atributos
print(persona1.nombre, persona1.edad)  # Juan 26
print(persona2.nombre, persona2.edad)  # María 30

# Modificar un objeto no afecta a los demás
persona1.nombre = "Juan Pablo"
print(persona1.nombre)  # Juan Pablo
print(persona2.nombre)  # María (no cambia)

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Modifica la clase Persona para añadir un atributo 'profesion'
# 2. Añade un método 'cambiar_profesion' que permita actualizar este atributo
# 3. Crea tres instancias diferentes y experimenta con sus atributos y métodos
# 4. Implementa un método 'presentacion_formal' que devuelva algo como:
#    "Me llamo [nombre], tengo [edad] años y trabajo como [profesion]"
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
class Persona:
    def __init__(self, nombre, edad, profesion="Desempleado"):
        self.nombre = nombre
        self.edad = edad
        self.profesion = profesion
        
    def saludar(self):
        return f"Hola, me llamo {self.nombre} y tengo {self.edad} años."
    
    def cumplir_anios(self):
        self.edad += 1
        return f"¡Feliz cumpleaños! Ahora tengo {self.edad} años."
    
    def cambiar_profesion(self, nueva_profesion):
        self.profesion = nueva_profesion
        return f"Ahora trabajo como {self.profesion}"
    
    def presentacion_formal(self):
        return f"Me llamo {self.nombre}, tengo {self.edad} años y trabajo como {self.profesion}."

# Creando instancias
persona1 = Persona("Juan", 25, "Ingeniero")
persona2 = Persona("María", 30, "Doctora")
persona3 = Persona("Carlos", 22)  # Usará la profesión por defecto

# Probando los métodos
print(persona1.presentacion_formal())
print(persona2.presentacion_formal())
print(persona3.presentacion_formal())

# Cambiando profesión
print(persona3.cambiar_profesion("Estudiante"))
print(persona3.presentacion_formal())
'''

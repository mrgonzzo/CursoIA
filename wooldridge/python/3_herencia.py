"""
SCRIPT 3: HERENCIA EN PYTHON
============================

Este script explora el concepto de herencia en la Programación Orientada a Objetos.
Aprenderás:
1. Qué es la herencia y por qué es útil
2. Cómo crear clases que heredan de otras
3. Cómo sobreescribir métodos
4. El uso de super() para llamar a métodos de la clase padre
5. Herencia múltiple
"""

# ======================================================
# PARTE 1: HERENCIA BÁSICA
# ======================================================

# Definimos una clase base (o clase padre)
class Vehiculo:
    def __init__(self, marca, modelo, año):
        self.marca = marca
        self.modelo = modelo
        self.año = año
        self.encendido = False
        self.velocidad = 0
    
    def encender(self):
        if not self.encendido:
            self.encendido = True
            return f"{self.marca} {self.modelo} encendido."
        return f"{self.marca} {self.modelo} ya está encendido."
    
    def apagar(self):
        if self.encendido:
            self.encendido = False
            self.velocidad = 0
            return f"{self.marca} {self.modelo} apagado."
        return f"{self.marca} {self.modelo} ya está apagado."
    
    def acelerar(self, incremento):
        if self.encendido:
            self.velocidad += incremento
            return f"Acelerando a {self.velocidad} km/h"
        return "No se puede acelerar. El vehículo está apagado."
    
    def __str__(self):
        estado = "encendido" if self.encendido else "apagado"
        return f"{self.marca} {self.modelo} ({self.año}) - {estado} - {self.velocidad} km/h"

# Ahora creamos una clase derivada (o clase hija) que hereda de Vehiculo
class Coche(Vehiculo):  # Entre paréntesis va la clase de la que hereda
    def __init__(self, marca, modelo, año, num_puertas):
        # Llamamos al constructor de la clase padre
        super().__init__(marca, modelo, año)
        # Añadimos atributos específicos de Coche
        self.num_puertas = num_puertas
        self.techo_solar = False
    
    # Añadimos métodos específicos de Coche
    def abrir_techo(self):
        if not self.techo_solar:
            self.techo_solar = True
            return "Techo solar abierto."
        return "El techo solar ya está abierto."
    
    def cerrar_techo(self):
        if self.techo_solar:
            self.techo_solar = False
            return "Techo solar cerrado."
        return "El techo solar ya está cerrado."
    
    # Sobreescribimos el método __str__ para incluir información específica de Coche
    def __str__(self):
        # Llamamos al método __str__ de la clase padre
        info_base = super().__str__()
        techo = "abierto" if self.techo_solar else "cerrado"
        # Añadimos información específica
        return f"{info_base} - {self.num_puertas} puertas - Techo: {techo}"

# Creamos otra clase derivada de Vehiculo
class Motocicleta(Vehiculo):
    def __init__(self, marca, modelo, año, tipo):
        super().__init__(marca, modelo, año)
        self.tipo = tipo  # deportiva, crucero, scooter, etc.
    
    def hacer_caballito(self):
        if self.encendido and self.velocidad > 10:
            return "¡Haciendo un caballito! 🏍️"
        return "No se puede hacer caballito. Velocidad insuficiente o moto apagada."
    
    # Sobreescribimos el método acelerar para que sea más rápido
    def acelerar(self, incremento):
        if self.encendido:
            # Las motocicletas aceleran más rápido (multiplicamos por 1.5)
            self.velocidad += incremento * 1.5
            return f"Acelerando rápidamente a {self.velocidad} km/h"
        return "No se puede acelerar. La motocicleta está apagada."
    
    def __str__(self):
        info_base = super().__str__()
        return f"{info_base} - Tipo: {self.tipo}"

# ======================================================
# PARTE 2: USANDO LA HERENCIA
# ======================================================

# Creamos instancias de nuestras clases
vehiculo = Vehiculo("Genérico", "Básico", 2020)
coche = Coche("Toyota", "Corolla", 2022, 4)
moto = Motocicleta("Honda", "CBR", 2021, "Deportiva")

# Mostramos información básica
print("=== Información de los vehículos ===")
print(vehiculo)
print(coche)
print(moto)
print()

# Usamos los métodos heredados
print("=== Métodos heredados ===")
print(coche.encender())  # Método heredado de Vehiculo
print(coche.acelerar(20))  # Método heredado de Vehiculo
print(moto.encender())  # Método heredado de Vehiculo
print(moto.acelerar(20))  # Método sobreescrito en Motocicleta
print()

# Usamos los métodos específicos
print("=== Métodos específicos ===")
print(coche.abrir_techo())  # Método específico de Coche
print(moto.hacer_caballito())  # Método específico de Motocicleta

print("\n=== Estado final ===")
print(coche)
print(moto)

# ======================================================
# PARTE 3: HERENCIA MÚLTIPLE
# ======================================================

# En Python, a diferencia de otros lenguajes como Java, una clase puede heredar de múltiples clases

class Dispositivo:
    def __init__(self, marca, modelo):
        self.marca = marca
        self.modelo = modelo
        self.conectado = False
    
    def conectar(self):
        self.conectado = True
        return f"{self.marca} {self.modelo} conectado."
    
    def desconectar(self):
        self.conectado = False
        return f"{self.marca} {self.modelo} desconectado."

class Reproductor:
    def __init__(self):
        self.reproduciendo = False
    
    def reproducir(self):
        self.reproduciendo = True
        return "Reproduciendo contenido..."
    
    def detener(self):
        self.reproduciendo = False
        return "Reproducción detenida."

# Herencia múltiple
class RadioCoche(Coche, Dispositivo, Reproductor):
    def __init__(self, marca, modelo, año, num_puertas, potencia):
        # Llamamos a los constructores de las clases padre
        Coche.__init__(self, marca, modelo, año, num_puertas)
        Dispositivo.__init__(self, marca, modelo)
        Reproductor.__init__(self)
        # Atributos propios
        self.potencia = potencia  # potencia en watts
    
    def __str__(self):
        estado_radio = "conectado" if self.conectado else "desconectado"
        estado_repro = "reproduciendo" if self.reproduciendo else "detenido"
        return f"{super().__str__()} - Radio: {estado_radio}, {estado_repro} - Potencia: {self.potencia}W"

# Creamos un RadioCoche
radio_coche = RadioCoche("Ford", "Mustang", 2023, 2, 500)

print("\n=== RadioCoche: Herencia Múltiple ===")
print(radio_coche.encender())  # De Vehiculo a través de Coche
print(radio_coche.conectar())  # De Dispositivo
print(radio_coche.reproducir())  # De Reproductor
print(radio_coche.acelerar(30))  # De Vehiculo a través de Coche
print(radio_coche.abrir_techo())  # De Coche
print(radio_coche)

# ======================================================
# PARTE 4: MRO (METHOD RESOLUTION ORDER)
# ======================================================

# El MRO define el orden en que Python busca métodos en la jerarquía de clases
# Esto es especialmente importante en la herencia múltiple

print("\n=== MRO (Method Resolution Order) ===")
print(RadioCoche.__mro__)
# Muestra el orden de resolución de métodos, que sería aproximadamente:
# RadioCoche -> Coche -> Vehiculo -> Dispositivo -> Reproductor -> object

# ======================================================
# PARTE 5: VERIFICACIÓN DE TIPOS Y HERENCIA
# ======================================================

print("\n=== Verificación de tipos ===")
# isinstance() verifica si un objeto es instancia de una clase
print(f"¿radio_coche es un RadioCoche? {isinstance(radio_coche, RadioCoche)}")
print(f"¿radio_coche es un Coche? {isinstance(radio_coche, Coche)}")
print(f"¿radio_coche es un Vehiculo? {isinstance(radio_coche, Vehiculo)}")
print(f"¿radio_coche es un Dispositivo? {isinstance(radio_coche, Dispositivo)}")
print(f"¿radio_coche es un Reproductor? {isinstance(radio_coche, Reproductor)}")
print(f"¿radio_coche es una Motocicleta? {isinstance(radio_coche, Motocicleta)}")

# issubclass() verifica si una clase es subclase de otra
print(f"\n¿RadioCoche es subclase de Coche? {issubclass(RadioCoche, Coche)}")
print(f"¿Coche es subclase de Vehiculo? {issubclass(Coche, Vehiculo)}")
print(f"¿Motocicleta es subclase de Coche? {issubclass(Motocicleta, Coche)}")

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Crear una clase base Animal con atributos nombre, edad y métodos comer() y dormir()
# 2. Crear dos subclases: Perro y Gato, cada una con métodos específicos (ladrar/maullar)
# 3. Implementar el método hacer_sonido() en cada clase, que sea polimórfico 
#    (Perro: "Guau", Gato: "Miau")
# 4. Crear una clase Mascota con un método jugar()
# 5. Implementar PerroMascota con herencia múltiple (Perro, Mascota)
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
class Animal:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def comer(self):
        return f"{self.nombre} está comiendo."
    
    def dormir(self):
        return f"{self.nombre} está durmiendo."
    
    def hacer_sonido(self):
        # Método base que será sobrescrito
        return "Animal haciendo sonido genérico."
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.nombre}, {self.edad} años"

class Perro(Animal):
    def __init__(self, nombre, edad, raza):
        super().__init__(nombre, edad)
        self.raza = raza
    
    def ladrar(self):
        return f"{self.nombre} dice: ¡Guau, guau!"
    
    def hacer_sonido(self):
        # Sobrescribimos el método de la clase padre
        return self.ladrar()
    
    def __str__(self):
        return f"{super().__str__()} - Raza: {self.raza}"

class Gato(Animal):
    def __init__(self, nombre, edad, color):
        super().__init__(nombre, edad)
        self.color = color
    
    def maullar(self):
        return f"{self.nombre} dice: ¡Miau, miau!"
    
    def hacer_sonido(self):
        # Sobrescribimos el método de la clase padre
        return self.maullar()
    
    def __str__(self):
        return f"{super().__str__()} - Color: {self.color}"

class Mascota:
    def __init__(self):
        self.juguetes = []
    
    def jugar(self):
        if self.juguetes:
            juguete = self.juguetes[0]
            return f"Jugando con {juguete}"
        return "No tengo juguetes para jugar"
    
    def agregar_juguete(self, juguete):
        self.juguetes.append(juguete)
        return f"Añadido juguete: {juguete}"

class PerroMascota(Perro, Mascota):
    def __init__(self, nombre, edad, raza, dueño):
        Perro.__init__(self, nombre, edad, raza)
        Mascota.__init__(self)
        self.dueño = dueño
    
    def saludar_dueño(self):
        return f"{self.nombre} está muy feliz de ver a {self.dueño}"
    
    def __str__(self):
        # Obtener la representación de cadena de Perro
        info_perro = Perro.__str__(self)
        # Añadir información específica de PerroMascota
        return f"{info_perro} - Dueño: {self.dueño} - Juguetes: {len(self.juguetes)}"

# Probando las clases
animal = Animal("Genérico", 5)
perro = Perro("Firulais", 3, "Labrador")
gato = Gato("Michi", 2, "Atigrado")
perro_mascota = PerroMascota("Rex", 4, "Pastor Alemán", "Juan")

print("=== Animales ===")
print(animal)
print(perro)
print(gato)
print(perro_mascota)

print("\n=== Métodos ===")
print(perro.comer())  # Método heredado de Animal
print(gato.dormir())  # Método heredado de Animal
print(perro.hacer_sonido())  # Método sobrescrito
print(gato.hacer_sonido())  # Método sobrescrito

print("\n=== Mascota ===")
print(perro_mascota.agregar_juguete("Pelota"))
print(perro_mascota.agregar_juguete("Hueso"))
print(perro_mascota.jugar())
print(perro_mascota.saludar_dueño())
print(perro_mascota)

print("\n=== MRO ===")
print(PerroMascota.__mro__)
'''

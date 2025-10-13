"""
SCRIPT 4: POLIMORFISMO EN PYTHON
================================

Este script explora el concepto de polimorfismo en la Programación Orientada a Objetos.
Aprenderás:
1. Qué es el polimorfismo y por qué es útil
2. Cómo implementar polimorfismo a través de la herencia
3. Cómo usar el Duck Typing para implementar polimorfismo
4. Interfaces abstractas y polimorfismo
"""

# ======================================================
# PARTE 1: INTRODUCCIÓN AL POLIMORFISMO
# ======================================================

# El polimorfismo es la capacidad de diferentes objetos para responder al mismo 
# mensaje (llamada a método) de manera adecuada según su tipo

# Veamos un ejemplo simple usando herencia:

class Animal:
    def __init__(self, nombre):
        self.nombre = nombre
    
    def hablar(self):
        # Este método será sobrescrito por las clases derivadas
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def presentarse(self):
        # Este método usa hablar(), pero no sabe qué implementación se usará
        return f"Me llamo {self.nombre} y puedo {self.hablar()}"

class Perro(Animal):
    def hablar(self):
        return "ladrar: ¡Guau!"

class Gato(Animal):
    def hablar(self):
        return "maullar: ¡Miau!"

class Vaca(Animal):
    def hablar(self):
        return "mugir: ¡Muuu!"

# Creamos una lista de animales de diferentes tipos
animales = [
    Perro("Firulais"),
    Gato("Michi"),
    Vaca("Lola")
]

# Ahora podemos tratar a todos los objetos de manera uniforme
# Esto es polimorfismo: cada objeto responde de forma diferente al mismo mensaje
print("=== Polimorfismo con hablar() ===")
for animal in animales:
    print(animal.presentarse())

# ======================================================
# PARTE 2: DUCK TYPING Y POLIMORFISMO
# ======================================================

# Python usa "Duck Typing": "Si camina como un pato y habla como un pato, entonces es un pato"
# En lugar de verificar el tipo de un objeto, Python verifica sus comportamientos

# Ejemplo: Clases no relacionadas por herencia, pero con métodos comunes

class Pato:
    def nadar(self):
        return "El pato nada en el estanque"
    
    def volar(self):
        return "El pato vuela bajo"

class Avion:
    def volar(self):
        return "El avión vuela a gran altura"

class Submarino:
    def nadar(self):
        return "El submarino navega bajo el agua"

# Función que acepta cualquier objeto que pueda "nadar"
def hacer_nadar(objeto):
    # No nos importa el tipo, solo que tenga el método 'nadar'
    print(objeto.nadar())

# Función que acepta cualquier objeto que pueda "volar"
def hacer_volar(objeto):
    # No nos importa el tipo, solo que tenga el método 'volar'
    print(objeto.volar())

# Probamos con diferentes objetos
print("\n=== Duck Typing ===")
pato = Pato()
avion = Avion()
submarino = Submarino()

# El pato puede nadar y volar
hacer_nadar(pato)
hacer_volar(pato)

# El avión solo puede volar
hacer_volar(avion)

# El submarino solo puede nadar
hacer_nadar(submarino)

# Si intentamos hacer volar al submarino, fallaría:
try:
    hacer_volar(submarino)
except AttributeError as e:
    print(f"Error: {e}")

# ======================================================
# PARTE 3: POLIMORFISMO CON INTERFACES ABSTRACTAS
# ======================================================

# Python soporta clases abstractas a través del módulo abc (Abstract Base Class)
from abc import ABC, abstractmethod

# Definimos una interfaz (clase abstracta)
class FiguraGeometrica(ABC):
    @abstractmethod
    def calcular_area(self):
        """Cada figura debe implementar el cálculo de su área"""
        pass
    
    @abstractmethod
    def calcular_perimetro(self):
        """Cada figura debe implementar el cálculo de su perímetro"""
        pass
    
    def describir(self):
        """Este método usa los métodos abstractos, que serán implementados por las subclases"""
        return f"Soy una {self.__class__.__name__} con área {self.calcular_area()} y perímetro {self.calcular_perimetro()}"

# Implementamos diferentes figuras
class Cuadrado(FiguraGeometrica):
    def __init__(self, lado):
        self.lado = lado
    
    def calcular_area(self):
        return self.lado ** 2
    
    def calcular_perimetro(self):
        return 4 * self.lado

class Circulo(FiguraGeometrica):
    def __init__(self, radio):
        self.radio = radio
    
    def calcular_area(self):
        import math
        return math.pi * self.radio ** 2
    
    def calcular_perimetro(self):
        import math
        return 2 * math.pi * self.radio

class Rectangulo(FiguraGeometrica):
    def __init__(self, base, altura):
        self.base = base
        self.altura = altura
    
    def calcular_area(self):
        return self.base * self.altura
    
    def calcular_perimetro(self):
        return 2 * (self.base + self.altura)

# No podemos instanciar directamente una clase abstracta
try:
    figura = FiguraGeometrica()
except TypeError as e:
    print(f"\n=== Clases Abstractas ===")
    print(f"Error al instanciar clase abstracta: {e}")

# Creamos una lista de figuras
figuras = [
    Cuadrado(5),
    Circulo(3),
    Rectangulo(4, 6)
]

# Usamos polimorfismo para trabajar con todas las figuras de manera uniforme
print("\n=== Polimorfismo con figuras geométricas ===")
for figura in figuras:
    print(figura.describir())
    print(f"  Área: {figura.calcular_area():.2f}")
    print(f"  Perímetro: {figura.calcular_perimetro():.2f}")

# ======================================================
# PARTE 4: APLICACIÓN PRÁCTICA - SISTEMA DE PAGOS
# ======================================================

# Vamos a modelar diferentes formas de pago usando polimorfismo

class MetodoPago(ABC):
    @abstractmethod
    def procesar_pago(self, monto):
        """Procesa un pago del monto especificado"""
        pass
    
    @abstractmethod
    def obtener_detalles(self):
        """Devuelve los detalles del método de pago"""
        pass

class TarjetaCredito(MetodoPago):
    def __init__(self, nombre, numero, fecha_venc, cvv):
        self.nombre = nombre
        # Ocultamos la mayoría de los dígitos por seguridad
        self.numero = '*' * 12 + numero[-4:]
        self.fecha_venc = fecha_venc
        self.cvv = cvv
    
    def procesar_pago(self, monto):
        # Aquí iría la lógica real de procesamiento con un gateway de pago
        return f"Procesando pago de ${monto:.2f} con tarjeta de crédito {self.numero}"
    
    def obtener_detalles(self):
        return f"Tarjeta de Crédito a nombre de {self.nombre}, terminada en {self.numero[-4:]}"

class PayPal(MetodoPago):
    def __init__(self, email):
        self.email = email
    
    def procesar_pago(self, monto):
        # Simulación del procesamiento con PayPal
        return f"Procesando pago de ${monto:.2f} con PayPal ({self.email})"
    
    def obtener_detalles(self):
        return f"Cuenta PayPal: {self.email}"

class TransferenciaBancaria(MetodoPago):
    def __init__(self, banco, cuenta, titular):
        self.banco = banco
        # Ocultamos parte del número de cuenta
        self.cuenta = '*' * 6 + cuenta[-4:]
        self.titular = titular
    
    def procesar_pago(self, monto):
        # Simulación de transferencia bancaria
        return f"Procesando transferencia de ${monto:.2f} al banco {self.banco}, cuenta {self.cuenta}"
    
    def obtener_detalles(self):
        return f"Transferencia al Banco {self.banco}, cuenta de {self.titular}"

# Sistema de procesamiento de pagos que usa polimorfismo
class ProcesadorPagos:
    def realizar_pago(self, metodo_pago, monto):
        # No nos importa qué tipo de pago sea, solo que implemente la interfaz
        print(f"Iniciando pago de ${monto:.2f}")
        print(f"Método de pago: {metodo_pago.obtener_detalles()}")
        resultado = metodo_pago.procesar_pago(monto)
        print(f"Resultado: {resultado}")
        print("Pago completado.\n")

# Creamos diferentes métodos de pago
tarjeta = TarjetaCredito("Juan Pérez", "1234567890123456", "12/25", "123")
paypal = PayPal("juan@example.com")
transferencia = TransferenciaBancaria("Santander", "987654321", "Juan Pérez")

# Usamos el procesador con diferentes métodos de pago (polimorfismo)
procesador = ProcesadorPagos()
print("\n=== Sistema de Pagos Polimórfico ===")
procesador.realizar_pago(tarjeta, 100.50)
procesador.realizar_pago(paypal, 75.20)
procesador.realizar_pago(transferencia, 500)

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Crea una clase abstracta Empleado con un método abstracto calcular_salario()
# 2. Implementa tres subclases: EmpleadoTiempoCompleto, EmpleadoMedioTiempo y Contratista
# 3. Cada subclase debe implementar calcular_salario() de forma diferente:
#    - EmpleadoTiempoCompleto: salario mensual fijo
#    - EmpleadoMedioTiempo: salario por hora * horas trabajadas
#    - Contratista: tarifa diaria * días trabajados
# 4. Crea una función calcular_nomina que reciba una lista de empleados y muestre el salario 
#    de cada uno, independientemente de su tipo
# 5. Implementa un método mostrar_detalles() en la clase base y sobrescríbelo en las subclases
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
from abc import ABC, abstractmethod

class Empleado(ABC):
    def __init__(self, nombre, id_empleado):
        self.nombre = nombre
        self.id_empleado = id_empleado
    
    @abstractmethod
    def calcular_salario(self):
        """Calcula el salario del empleado"""
        pass
    
    def mostrar_detalles(self):
        """Muestra los detalles del empleado"""
        return f"ID: {self.id_empleado}, Nombre: {self.nombre}"

class EmpleadoTiempoCompleto(Empleado):
    def __init__(self, nombre, id_empleado, salario_mensual):
        super().__init__(nombre, id_empleado)
        self.salario_mensual = salario_mensual
    
    def calcular_salario(self):
        return self.salario_mensual
    
    def mostrar_detalles(self):
        return f"{super().mostrar_detalles()}, Tipo: Tiempo Completo, Salario Mensual: ${self.salario_mensual:.2f}"

class EmpleadoMedioTiempo(Empleado):
    def __init__(self, nombre, id_empleado, salario_hora, horas_trabajadas):
        super().__init__(nombre, id_empleado)
        self.salario_hora = salario_hora
        self.horas_trabajadas = horas_trabajadas
    
    def calcular_salario(self):
        return self.salario_hora * self.horas_trabajadas
    
    def mostrar_detalles(self):
        return f"{super().mostrar_detalles()}, Tipo: Medio Tiempo, Horas: {self.horas_trabajadas}, Tarifa: ${self.salario_hora:.2f}/hora"

class Contratista(Empleado):
    def __init__(self, nombre, id_empleado, tarifa_diaria, dias_trabajados):
        super().__init__(nombre, id_empleado)
        self.tarifa_diaria = tarifa_diaria
        self.dias_trabajados = dias_trabajados
    
    def calcular_salario(self):
        return self.tarifa_diaria * self.dias_trabajados
    
    def mostrar_detalles(self):
        return f"{super().mostrar_detalles()}, Tipo: Contratista, Días: {self.dias_trabajados}, Tarifa: ${self.tarifa_diaria:.2f}/día"

def calcular_nomina(empleados):
    """Calcula y muestra el salario de cada empleado"""
    print("\n=== NÓMINA DE EMPLEADOS ===")
    total_nomina = 0
    
    for empleado in empleados:
        salario = empleado.calcular_salario()
        total_nomina += salario
        print(f"{empleado.mostrar_detalles()} => Salario: ${salario:.2f}")
    
    print(f"\nTotal Nómina: ${total_nomina:.2f}")

# Probamos la solución
if __name__ == "__main__":
    empleados = [
        EmpleadoTiempoCompleto("Ana Gómez", "E001", 5000),
        EmpleadoMedioTiempo("Pedro Sánchez", "E002", 15, 80),
        Contratista("Luis Torres", "C001", 200, 10),
        EmpleadoTiempoCompleto("María Rodríguez", "E003", 4500),
        EmpleadoMedioTiempo("Carlos López", "E004", 12, 100),
    ]
    
    calcular_nomina(empleados)
'''

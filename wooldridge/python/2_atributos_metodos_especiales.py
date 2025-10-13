"""
SCRIPT 2: ATRIBUTOS Y MÉTODOS ESPECIALES
=========================================

Este script profundiza en tipos de atributos y métodos en Python.
Aprenderás:
1. Atributos de instancia vs. atributos de clase
2. Métodos especiales (dunder methods)
3. Properties para controlar el acceso a los atributos
4. Métodos de clase y métodos estáticos
"""

# ======================================================
# PARTE 1: ATRIBUTOS DE INSTANCIA VS. ATRIBUTOS DE CLASE
# ======================================================

class Estudiante:
    # Atributo de clase: compartido por todas las instancias
    escuela = "Instituto Tecnológico"  
    num_estudiantes = 0  # Contador de estudiantes
    
    def __init__(self, nombre, edad, promedio):
        # Atributos de instancia: únicos para cada objeto
        self.nombre = nombre
        self.edad = edad
        self.promedio = promedio
        
        # Incrementamos el contador al crear cada estudiante
        Estudiante.num_estudiantes += 1
        
    def mostrar_info(self):
        return f"Estudiante: {self.nombre}, Edad: {self.edad}, Promedio: {self.promedio}"
    
    @classmethod
    def cambiar_escuela(cls, nueva_escuela):
        # Método de clase: opera sobre la clase, no sobre una instancia
        cls.escuela = nueva_escuela
        return f"La escuela ahora es: {cls.escuela}"

# Creamos estudiantes
estudiante1 = Estudiante("Ana", 20, 9.5)
estudiante2 = Estudiante("Carlos", 22, 8.7)

# Acceso a atributos de instancia (diferentes para cada objeto)
print(estudiante1.nombre)  # Ana
print(estudiante2.nombre)  # Carlos

# Acceso a atributos de clase (compartidos)
print(estudiante1.escuela)  # Instituto Tecnológico
print(estudiante2.escuela)  # Instituto Tecnológico
print(Estudiante.escuela)   # También podemos acceder directamente desde la clase

# Cambiar un atributo de clase afecta a todas las instancias
Estudiante.escuela = "Universidad Nacional"
print(estudiante1.escuela)  # Universidad Nacional
print(estudiante2.escuela)  # Universidad Nacional

# Verificamos el contador de estudiantes
print(f"Total de estudiantes: {Estudiante.num_estudiantes}")  # 2

# Usando el método de clase
print(Estudiante.cambiar_escuela("Academia de Ciencias"))
print(estudiante1.escuela)  # Academia de Ciencias

# ======================================================
# PARTE 2: MÉTODOS ESPECIALES (DUNDER METHODS)
# ======================================================

class Rectangulo:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
    
    # __str__: Representación legible para humanos (str(objeto) o print(objeto))
    def __str__(self):
        return f"Rectángulo de {self.ancho}x{self.alto}"
    
    # __repr__: Representación para desarrolladores, debe ser inequívoca
    def __repr__(self):
        return f"Rectangulo({self.ancho}, {self.alto})"
    
    # __eq__: Define el comportamiento del operador == (igualdad)
    def __eq__(self, otro):
        if not isinstance(otro, Rectangulo):
            return False
        return self.ancho == otro.ancho and self.alto == otro.alto
    
    # __lt__: Define el comportamiento del operador < (menor que)
    def __lt__(self, otro):
        if not isinstance(otro, Rectangulo):
            return NotImplemented
        return self.area() < otro.area()
    
    # __add__: Define el comportamiento del operador + (suma)
    def __add__(self, otro):
        if isinstance(otro, Rectangulo):
            return Rectangulo(self.ancho + otro.ancho, self.alto + otro.alto)
        return NotImplemented
    
    # Método regular para calcular el área
    def area(self):
        return self.ancho * self.alto

# Creamos rectángulos
rect1 = Rectangulo(5, 3)
rect2 = Rectangulo(2, 4)
rect3 = Rectangulo(5, 3)  # Mismas dimensiones que rect1

# Usamos los métodos especiales
print(str(rect1))  # Llama a __str__: "Rectángulo de 5x3"
print(repr(rect2))  # Llama a __repr__: "Rectangulo(2, 4)"

# Operador de igualdad (==)
print(rect1 == rect3)  # True (mismas dimensiones)
print(rect1 == rect2)  # False (diferentes dimensiones)

# Operador menor que (<)
print(rect2 < rect1)  # True (área de rect2 es menor)

# Operador suma (+)
rect4 = rect1 + rect2
print(rect4)  # "Rectángulo de 7x7"

# ======================================================
# PARTE 3: PROPERTIES - GETTERS Y SETTERS
# ======================================================

class Empleado:
    def __init__(self, nombre, salario):
        self.nombre = nombre
        # El salario es un atributo privado (indicado por el _)
        self._salario = salario
    
    # Getter - Permite acceder al salario de forma controlada
    @property
    def salario(self):
        return self._salario
    
    # Setter - Permite modificar el salario con validación
    @salario.setter
    def salario(self, valor):
        if valor < 0:
            raise ValueError("El salario no puede ser negativo")
        self._salario = valor
    
    # Property calculada - No tiene un atributo directo
    @property
    def salario_anual(self):
        return self._salario * 12

# Creamos un empleado
empleado = Empleado("María", 5000)

# Accedemos a la propiedad (parece un atributo pero llama al método getter)
print(empleado.salario)  # 5000

# Modificamos usando el setter
empleado.salario = 6000
print(empleado.salario)  # 6000

# Intentamos asignar un valor inválido
try:
    empleado.salario = -1000  # Lanzará un ValueError
except ValueError as e:
    print(f"Error: {e}")

# Accedemos a la propiedad calculada
print(f"Salario anual: {empleado.salario_anual}")  # 72000

# ======================================================
# PARTE 4: MÉTODOS DE CLASE Y MÉTODOS ESTÁTICOS
# ======================================================

class Fecha:
    def __init__(self, dia, mes, año):
        self.dia = dia
        self.mes = mes
        self.año = año
    
    def __str__(self):
        return f"{self.dia:02d}/{self.mes:02d}/{self.año}"
    
    # Método de clase: puede acceder/modificar atributos de la clase
    @classmethod
    def desde_cadena(cls, cadena_fecha):
        """Crea una instancia de Fecha a partir de una cadena 'DD-MM-AAAA'"""
        dia, mes, año = map(int, cadena_fecha.split('-'))
        return cls(dia, mes, año)
    
    # Método estático: no accede a la clase ni a la instancia
    @staticmethod
    def es_fecha_valida(dia, mes, año):
        """Verifica si una fecha es válida"""
        if año < 0 or mes < 1 or mes > 12 or dia < 1:
            return False
        
        # Días en cada mes (ignorando años bisiestos para simplificar)
        dias_por_mes = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # Ajuste para febrero en años bisiestos
        if mes == 2 and (año % 4 == 0 and (año % 100 != 0 or año % 400 == 0)):
            return dia <= 29
        
        return dia <= dias_por_mes[mes]

# Usando el método de clase para crear una instancia
fecha1 = Fecha.desde_cadena("15-06-2023")
print(fecha1)  # 15/06/2023

# Usando el método estático (no necesita instancia)
print(Fecha.es_fecha_valida(29, 2, 2020))  # True (2020 es bisiesto)
print(Fecha.es_fecha_valida(29, 2, 2023))  # False (2023 no es bisiesto)

# También podemos llamar al método estático desde una instancia
print(fecha1.es_fecha_valida(31, 4, 2023))  # False (abril tiene 30 días)

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Crea una clase Producto con atributos nombre, precio y stock
# 2. Implementa métodos especiales __str__, __repr__ y __eq__
# 3. Añade una property 'valor_total' que calcule precio * stock
# 4. Crea un método de clase 'desde_diccionario' que cree un Producto a partir de un diccionario
# 5. Implementa un método estático 'aplicar_descuento' que reciba un precio y un porcentaje y devuelva el precio con descuento
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
class Producto:
    # Atributo de clase
    iva = 0.21  # 21% IVA
    
    def __init__(self, nombre, precio, stock):
        self.nombre = nombre
        self._precio = precio  # Atributo con property
        self.stock = stock
    
    def __str__(self):
        return f"{self.nombre} - Precio: ${self._precio:.2f} - Stock: {self.stock} unidades"
    
    def __repr__(self):
        return f"Producto('{self.nombre}', {self._precio}, {self.stock})"
    
    def __eq__(self, otro):
        if not isinstance(otro, Producto):
            return False
        return (self.nombre == otro.nombre and 
                self._precio == otro._precio)
    
    @property
    def precio(self):
        return self._precio
    
    @precio.setter
    def precio(self, valor):
        if valor < 0:
            raise ValueError("El precio no puede ser negativo")
        self._precio = valor
    
    @property
    def valor_total(self):
        return self._precio * self.stock
    
    @property
    def precio_con_iva(self):
        return self._precio * (1 + Producto.iva)
    
    @classmethod
    def desde_diccionario(cls, diccionario):
        """Crea un producto desde un diccionario con claves: nombre, precio, stock"""
        return cls(
            diccionario.get('nombre', 'Sin nombre'),
            diccionario.get('precio', 0),
            diccionario.get('stock', 0)
        )
    
    @staticmethod
    def aplicar_descuento(precio, porcentaje):
        """Aplica un descuento a un precio"""
        if porcentaje < 0 or porcentaje > 100:
            raise ValueError("El porcentaje debe estar entre 0 y 100")
        factor = 1 - (porcentaje / 100)
        return precio * factor

# Probando la clase
producto1 = Producto("Laptop", 1200, 5)
producto2 = Producto("Monitor", 300, 10)

# Probando properties
print(producto1.precio)  # 1200
print(producto1.valor_total)  # 6000
print(producto1.precio_con_iva)  # 1452.0

# Probando método de clase
datos = {
    'nombre': 'Teclado',
    'precio': 80,
    'stock': 15
}
producto3 = Producto.desde_diccionario(datos)
print(producto3)  # Teclado - Precio: $80.00 - Stock: 15 unidades

# Probando método estático
precio_original = 100
descuento = 20
precio_final = Producto.aplicar_descuento(precio_original, descuento)
print(f"Precio original: ${precio_original}, con {descuento}% de descuento: ${precio_final}")
# Output: Precio original: $100, con 20% de descuento: $80.0
'''

"""
SCRIPT 5: ENCAPSULAMIENTO EN PYTHON
===================================

Este script explora el concepto de encapsulamiento en la Programación Orientada a Objetos.
Aprenderás:
1. Qué es el encapsulamiento y por qué es importante
2. Convenciones para atributos públicos, protegidos y privados en Python
3. Uso de getters y setters
4. Implementación de properties (propiedades)
5. Buenas prácticas de encapsulamiento
"""

# ======================================================
# PARTE 1: INTRODUCCIÓN AL ENCAPSULAMIENTO
# ======================================================

# El encapsulamiento es uno de los principios fundamentales de la POO
# Consiste en ocultar los detalles internos de implementación de una clase
# y mostrar solo lo necesario para el uso externo

# En Python, el encapsulamiento se basa principalmente en convenciones de nombres:
# - Atributos públicos: sin guion bajo al inicio (name)
# - Atributos protegidos: un guion bajo al inicio (_name) - convención, no restricción
# - Atributos privados: dos guiones bajos al inicio (__name) - name-mangling

class CuentaBancaria:
    def __init__(self, titular, saldo_inicial=0):
        # Atributo público: puede ser accedido directamente
        self.titular = titular
        
        # Atributo protegido: por convención, no debería accederse directamente
        self._saldo = saldo_inicial
        
        # Atributo privado: Python lo renombra para dificultar el acceso directo
        self.__historial = []
        
        # Registramos la creación de la cuenta en el historial
        self.__registrar_transaccion("Apertura de cuenta", saldo_inicial)
    
    # Método público
    def depositar(self, monto):
        if monto <= 0:
            raise ValueError("El monto a depositar debe ser positivo")
        
        self._saldo += monto
        self.__registrar_transaccion("Depósito", monto)
        return f"Depósito de ${monto} realizado. Nuevo saldo: ${self._saldo}"
    
    # Método público
    def retirar(self, monto):
        if monto <= 0:
            raise ValueError("El monto a retirar debe ser positivo")
        
        if monto > self._saldo:
            raise ValueError("Saldo insuficiente")
        
        self._saldo -= monto
        self.__registrar_transaccion("Retiro", -monto)
        return f"Retiro de ${monto} realizado. Nuevo saldo: ${self._saldo}"
    
    # Método público que permite consultar el saldo
    def consultar_saldo(self):
        return f"Saldo actual: ${self._saldo}"
    
    # Método privado (solo para uso interno)
    def __registrar_transaccion(self, tipo, monto):
        import datetime
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__historial.append({
            "fecha": fecha,
            "tipo": tipo,
            "monto": monto,
            "saldo_resultante": self._saldo
        })
    
    # Método público para acceder al historial (de manera controlada)
    def obtener_historial(self):
        # Devolvemos una copia del historial, no la referencia original
        return self.__historial.copy()

# ======================================================
# PARTE 2: DEMOSTRANDO EL ENCAPSULAMIENTO
# ======================================================

# Creamos una cuenta
cuenta = CuentaBancaria("Juan Pérez", 1000)

print("=== ENCAPSULAMIENTO BÁSICO ===")

# Acceso a atributo público
print(f"Titular: {cuenta.titular}")

# Intentar acceder al atributo protegido (Python lo permite, pero por convención no deberíamos)
print(f"Saldo (accediendo a _saldo): {cuenta._saldo}")  # No recomendado, pero funciona

# Intentar acceder al atributo privado directamente fallará
try:
    print(f"Historial directo: {cuenta.__historial}")
except AttributeError as e:
    print(f"Error al acceder a __historial: {e}")

# Pero Python en realidad renombra los atributos privados, no los hace verdaderamente privados
# Esto se llama "name mangling" (deformación de nombres)
print(f"Historial usando name mangling: {cuenta._CuentaBancaria__historial}")  # No recomendado

# Forma correcta de acceder: mediante métodos públicos
print(cuenta.consultar_saldo())
print(cuenta.depositar(500))
print(cuenta.retirar(200))

# Acceso al historial a través del método público
historial = cuenta.obtener_historial()
print("\n=== Historial de transacciones ===")
for transaccion in historial:
    print(f"{transaccion['fecha']} - {transaccion['tipo']}: ${transaccion['monto']} - Saldo: ${transaccion['saldo_resultante']}")

# ======================================================
# PARTE 3: GETTERS Y SETTERS
# ======================================================

# Los getters y setters son métodos que permiten controlar el acceso a los atributos
# Esto permite validar o procesar los datos antes de modificarlos

class Producto:
    def __init__(self, nombre, precio, stock):
        self.nombre = nombre  # Atributo público
        self._precio = precio  # Atributo protegido
        self._stock = stock    # Atributo protegido
    
    # Getter para precio
    def get_precio(self):
        return self._precio
    
    # Setter para precio (con validación)
    def set_precio(self, nuevo_precio):
        if nuevo_precio < 0:
            raise ValueError("El precio no puede ser negativo")
        self._precio = nuevo_precio
    
    # Getter para stock
    def get_stock(self):
        return self._stock
    
    # Setter para stock (con validación)
    def set_stock(self, nuevo_stock):
        if nuevo_stock < 0:
            raise ValueError("El stock no puede ser negativo")
        self._stock = nuevo_stock
    
    # Método para calcular el valor total (precio * stock)
    def valor_total(self):
        return self._precio * self._stock

print("\n=== GETTERS Y SETTERS ===")
producto = Producto("Laptop", 1200, 5)

# Usando getters
print(f"Producto: {producto.nombre}")
print(f"Precio: ${producto.get_precio()}")
print(f"Stock: {producto.get_stock()} unidades")
print(f"Valor total: ${producto.valor_total()}")

# Usando setters con validación
try:
    producto.set_precio(-100)  # Esto debería fallar
except ValueError as e:
    print(f"Error: {e}")

producto.set_precio(1500)  # Esto debería funcionar
print(f"Nuevo precio: ${producto.get_precio()}")

# ======================================================
# PARTE 4: PROPERTIES - MANERA PYTHÓNICA DE ENCAPSULAR
# ======================================================

# Las properties son una forma más elegante y pythónica de implementar
# getters y setters, haciendo que parezca que se accede directamente a los atributos

class Empleado:
    def __init__(self, nombre, apellido, salario_base):
        self.nombre = nombre          # Atributo público
        self.apellido = apellido      # Atributo público
        self._salario_base = salario_base  # Atributo protegido
        self._bonus = 0               # Atributo protegido
    
    # Property para salario_base
    @property
    def salario_base(self):
        """Obtiene el salario base del empleado."""
        return self._salario_base
    
    @salario_base.setter
    def salario_base(self, valor):
        """Establece el salario base con validación."""
        if valor < 0:
            raise ValueError("El salario base no puede ser negativo")
        self._salario_base = valor
    
    # Property para bonus
    @property
    def bonus(self):
        """Obtiene el bonus del empleado."""
        return self._bonus
    
    @bonus.setter
    def bonus(self, valor):
        """Establece el bonus con validación."""
        if valor < 0:
            raise ValueError("El bonus no puede ser negativo")
        self._bonus = valor
    
    # Property calculada - solo getter (read-only)
    @property
    def salario_total(self):
        """Calcula el salario total (base + bonus)."""
        return self._salario_base + self._bonus
    
    # Property para nombre completo
    @property
    def nombre_completo(self):
        """Devuelve el nombre completo del empleado."""
        return f"{self.nombre} {self.apellido}"

print("\n=== PROPERTIES ===")
empleado = Empleado("Ana", "Gómez", 3000)

# Usando properties (parece acceso directo pero usa getters)
print(f"Empleado: {empleado.nombre_completo}")
print(f"Salario base: ${empleado.salario_base}")
print(f"Bonus actual: ${empleado.bonus}")
print(f"Salario total: ${empleado.salario_total}")

# Modificando valores mediante properties (usa setters)
empleado.salario_base = 3500
empleado.bonus = 500

print(f"\nDespués de los cambios:")
print(f"Salario base: ${empleado.salario_base}")
print(f"Bonus actual: ${empleado.bonus}")
print(f"Salario total: ${empleado.salario_total}")

# Intentando modificar una property de solo lectura
try:
    empleado.salario_total = 5000  # Esto fallará porque no tiene setter
except AttributeError as e:
    print(f"Error: {e}")

# Intentando establecer un valor inválido
try:
    empleado.bonus = -200  # Esto fallará por la validación en el setter
except ValueError as e:
    print(f"Error: {e}")

# ======================================================
# PARTE 5: BUENAS PRÁCTICAS DE ENCAPSULAMIENTO
# ======================================================

class GestorArchivos:
    def __init__(self, nombre_archivo):
        self.nombre_archivo = nombre_archivo
        self._archivo = None  # Atributo protegido
        self.__contenido_cache = None  # Atributo privado (caché interno)
        self.__esta_modificado = False  # Flag interno para seguimiento
    
    # Método público
    def abrir(self):
        """Método público para abrir el archivo."""
        print(f"Abriendo archivo: {self.nombre_archivo}")
        self._archivo = f"Conexión al archivo {self.nombre_archivo}"  # Simulamos la apertura
        self.__cargar_contenido()
    
    # Método privado
    def __cargar_contenido(self):
        """Método privado para cargar el contenido del archivo."""
        print("Cargando contenido...")
        # Aquí cargaríamos el contenido real del archivo
        self.__contenido_cache = f"Contenido del archivo {self.nombre_archivo}"
    
    # Método público
    def leer(self):
        """Método público para leer el contenido."""
        if not self._archivo:
            raise RuntimeError("El archivo no está abierto")
        return self.__contenido_cache
    
    # Método público
    def escribir(self, nuevo_contenido):
        """Método público para escribir contenido."""
        if not self._archivo:
            raise RuntimeError("El archivo no está abierto")
        print(f"Escribiendo: {nuevo_contenido}")
        self.__contenido_cache = nuevo_contenido
        self.__esta_modificado = True
        self.__registrar_cambio("Escritura")
    
    # Método privado
    def __registrar_cambio(self, tipo_cambio):
        """Método privado para registrar cambios."""
        import datetime
        ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ahora}] {tipo_cambio} en {self.nombre_archivo}")
    
    # Método público
    def guardar(self):
        """Método público para guardar los cambios."""
        if not self._archivo:
            raise RuntimeError("El archivo no está abierto")
        
        if self.__esta_modificado:
            print(f"Guardando cambios en {self.nombre_archivo}")
            # Aquí guardaríamos los cambios realmente
            self.__esta_modificado = False
            self.__registrar_cambio("Guardado")
        else:
            print("No hay cambios que guardar")
    
    # Método público
    def cerrar(self):
        """Método público para cerrar el archivo."""
        if not self._archivo:
            return
        
        if self.__esta_modificado:
            print("ADVERTENCIA: Hay cambios sin guardar")
        
        print(f"Cerrando archivo: {self.nombre_archivo}")
        self._archivo = None
        self.__contenido_cache = None
        self.__esta_modificado = False

print("\n=== BUENAS PRÁCTICAS DE ENCAPSULAMIENTO ===")
gestor = GestorArchivos("documento.txt")

# Usamos la interfaz pública
gestor.abrir()
print(f"Contenido: {gestor.leer()}")
gestor.escribir("Nuevo contenido para el documento")
gestor.guardar()
gestor.cerrar()

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Crea una clase Estudiante con los siguientes atributos privados:
#    - nombre
#    - edad
#    - calificaciones (lista de calificaciones)
# 2. Implementa getters y setters para cada atributo, con validaciones:
#    - La edad debe estar entre 18 y 99
#    - Las calificaciones deben estar entre 0 y 10
# 3. Añade las siguientes propiedades:
#    - promedio: calcula el promedio de calificaciones
#    - aprobado: devuelve True si el promedio es >= 6, False en caso contrario
# 4. Implementa un método para añadir una calificación a la lista, con validación
# 5. Implementa un método privado para validar si una calificación es válida
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
class Estudiante:
    def __init__(self, nombre, edad):
        self.__nombre = nombre
        self.__edad = None  # Lo establecemos a través del setter para validar
        self.__calificaciones = []
        
        # Usamos el setter para validar la edad
        self.set_edad(edad)
    
    # Getters y Setters para nombre
    def get_nombre(self):
        return self.__nombre
    
    def set_nombre(self, nombre):
        if not nombre or not isinstance(nombre, str):
            raise ValueError("El nombre debe ser un texto válido")
        self.__nombre = nombre
    
    # Getters y Setters para edad
    def get_edad(self):
        return self.__edad
    
    def set_edad(self, edad):
        if not isinstance(edad, int) or edad < 18 or edad > 99:
            raise ValueError("La edad debe ser un entero entre 18 y 99")
        self.__edad = edad
    
    # Getters para calificaciones (devuelve una copia para mantener encapsulamiento)
    def get_calificaciones(self):
        return self.__calificaciones.copy()
    
    # Método privado para validar calificación
    def __es_calificacion_valida(self, calificacion):
        return isinstance(calificacion, (int, float)) and 0 <= calificacion <= 10
    
    # Método para añadir calificación
    def agregar_calificacion(self, calificacion):
        if not self.__es_calificacion_valida(calificacion):
            raise ValueError("La calificación debe ser un número entre 0 y 10")
        self.__calificaciones.append(calificacion)
        return f"Calificación {calificacion} agregada"
    
    # Property para el promedio
    @property
    def promedio(self):
        if not self.__calificaciones:
            return 0
        return sum(self.__calificaciones) / len(self.__calificaciones)
    
    # Property para determinar si está aprobado
    @property
    def aprobado(self):
        return self.promedio >= 6
    
    # Property para nombre (versión con decoradores)
    @property
    def nombre(self):
        return self.__nombre
    
    @nombre.setter
    def nombre(self, valor):
        self.set_nombre(valor)
    
    # Property para edad
    @property
    def edad(self):
        return self.__edad
    
    @edad.setter
    def edad(self, valor):
        self.set_edad(valor)
    
    # Property para calificaciones
    @property
    def calificaciones(self):
        return self.get_calificaciones()
    
    def __str__(self):
        estado = "APROBADO" if self.aprobado else "REPROBADO"
        return f"Estudiante: {self.__nombre}, Edad: {self.__edad}, Promedio: {self.promedio:.2f} - {estado}"

# Probamos la clase
estudiante = Estudiante("Carlos López", 20)

# Añadimos calificaciones
estudiante.agregar_calificacion(7)
estudiante.agregar_calificacion(8.5)
estudiante.agregar_calificacion(6)
estudiante.agregar_calificacion(9)

# Mostramos información
print(f"Nombre: {estudiante.nombre}")
print(f"Edad: {estudiante.edad}")
print(f"Calificaciones: {estudiante.calificaciones}")
print(f"Promedio: {estudiante.promedio:.2f}")
print(f"¿Aprobado? {'Sí' if estudiante.aprobado else 'No'}")
print(estudiante)

# Probamos validaciones
try:
    estudiante.edad = 15  # Debería fallar
except ValueError as e:
    print(f"Error edad: {e}")

try:
    estudiante.agregar_calificacion(11)  # Debería fallar
except ValueError as e:
    print(f"Error calificación: {e}")

# Modificamos algunos valores
estudiante.nombre = "Carlos Alberto López"
estudiante.edad = 21
print(f"\nDespués de modificar:")
print(estudiante)
'''

"""
SCRIPT 7: PATRONES DE DISEÑO EN PYTHON
======================================

Este script explora algunos patrones de diseño comunes implementados en Python.
Aprenderás:
1. Patrón Singleton (Creacional)
2. Patrón Factory Method (Creacional)
3. Patrón Observer (Comportamiento)
4. Patrón Strategy (Comportamiento)
5. Patrón Adapter (Estructural)
"""

# ======================================================
# PARTE 1: PATRÓN SINGLETON
# ======================================================
# "Garantiza que una clase solo tenga una instancia y proporciona un punto de acceso global a ella"
print("=== PATRÓN SINGLETON ===")

class ConfiguracionSingleton:
    # Variable de clase para almacenar la única instancia
    _instancia = None
    
    def __new__(cls):
        # Si no existe una instancia, la creamos
        if cls._instancia is None:
            print("Creando la instancia de Configuración")
            cls._instancia = super(ConfiguracionSingleton, cls).__new__(cls)
            # Inicializamos los atributos aquí
            cls._instancia.host = "localhost"
            cls._instancia.puerto = 3306
            cls._instancia.usuario = "admin"
            cls._instancia.password = "admin123"
            cls._instancia.debug = False
        return cls._instancia
    
    def mostrar_configuracion(self):
        return f"Host: {self.host}, Puerto: {self.puerto}, Usuario: {self.usuario}, Debug: {self.debug}"
    
    def activar_debug(self):
        self.debug = True
        return "Modo debug activado"

# Probamos el Singleton
print("Creando primera instancia...")
config1 = ConfiguracionSingleton()
print(config1.mostrar_configuracion())

print("\nCreando segunda instancia...")
config2 = ConfiguracionSingleton()  # No se crea una nueva instancia
print(config2.mostrar_configuracion())

# Modificamos la configuración a través de una instancia
print("\nModificando configuración...")
config1.activar_debug()
config1.host = "192.168.1.1"

# Verificamos que los cambios se reflejan en todas las "instancias"
print("\nVerificando que ambas referencias apuntan a la misma instancia:")
print(f"Config1: {config1.mostrar_configuracion()}")
print(f"Config2: {config2.mostrar_configuracion()}")

# Comprobamos que son la misma instancia
print(f"\n¿Son la misma instancia? {config1 is config2}")  # True

# ======================================================
# PARTE 2: PATRÓN FACTORY METHOD
# ======================================================
# "Define una interfaz para crear un objeto, pero permite a las subclases decidir qué clase instanciar"
print("\n=== PATRÓN FACTORY METHOD ===")

from abc import ABC, abstractmethod

# Interfaz del producto
class Animal(ABC):
    @abstractmethod
    def hablar(self):
        pass
    
    @abstractmethod
    def tipo(self):
        pass

# Productos concretos
class Perro(Animal):
    def hablar(self):
        return "¡Guau! ¡Guau!"
    
    def tipo(self):
        return "Perro"

class Gato(Animal):
    def hablar(self):
        return "¡Miau! ¡Miau!"
    
    def tipo(self):
        return "Gato"

class Vaca(Animal):
    def hablar(self):
        return "¡Muuu! ¡Muuu!"
    
    def tipo(self):
        return "Vaca"

# Factory Method (Creator)
class FabricaAnimales(ABC):
    @abstractmethod
    def crear_animal(self):
        pass
    
    def hacer_hablar_animal(self):
        # Método que utiliza el objeto creado por el Factory Method
        animal = self.crear_animal()
        return f"El {animal.tipo()} dice: {animal.hablar()}"

# Concrete Creators
class FabricaPerros(FabricaAnimales):
    def crear_animal(self):
        return Perro()

class FabricaGatos(FabricaAnimales):
    def crear_animal(self):
        return Gato()

class FabricaVacas(FabricaAnimales):
    def crear_animal(self):
        return Vaca()

# Uso del Factory Method
fabrica_perros = FabricaPerros()
fabrica_gatos = FabricaGatos()
fabrica_vacas = FabricaVacas()

print(fabrica_perros.hacer_hablar_animal())
print(fabrica_gatos.hacer_hablar_animal())
print(fabrica_vacas.hacer_hablar_animal())

# Factory simplificado (función en lugar de jerarquía de clases)
def crear_animal(tipo_animal):
    animales = {
        'perro': Perro(),
        'gato': Gato(),
        'vaca': Vaca()
    }
    return animales.get(tipo_animal.lower())

print("\nFactory simplificado:")
animal1 = crear_animal('perro')
animal2 = crear_animal('gato')

if animal1:
    print(f"El {animal1.tipo()} dice: {animal1.hablar()}")
if animal2:
    print(f"El {animal2.tipo()} dice: {animal2.hablar()}")

# ======================================================
# PARTE 3: PATRÓN OBSERVER
# ======================================================
# "Define una dependencia uno a muchos entre objetos, de forma que cuando un objeto cambia de estado,
# todos sus dependientes son notificados y actualizados automáticamente."
print("\n=== PATRÓN OBSERVER ===")

class Observador(ABC):
    @abstractmethod
    def actualizar(self, temperatura, humedad, presion):
        pass

class Sujeto(ABC):
    @abstractmethod
    def registrar_observador(self, observador):
        pass
    
    @abstractmethod
    def eliminar_observador(self, observador):
        pass
    
    @abstractmethod
    def notificar_observadores(self):
        pass

class EstacionMeteorologica(Sujeto):
    def __init__(self):
        self._observadores = []
        self._temperatura = 0
        self._humedad = 0
        self._presion = 0
    
    def registrar_observador(self, observador):
        if observador not in self._observadores:
            self._observadores.append(observador)
    
    def eliminar_observador(self, observador):
        self._observadores.remove(observador)
    
    def notificar_observadores(self):
        for observador in self._observadores:
            observador.actualizar(self._temperatura, self._humedad, self._presion)
    
    def establecer_mediciones(self, temperatura, humedad, presion):
        self._temperatura = temperatura
        self._humedad = humedad
        self._presion = presion
        self.mediciones_cambiadas()
    
    def mediciones_cambiadas(self):
        self.notificar_observadores()

class PantallaCondicionesActuales(Observador):
    def __init__(self, estacion_meteorologica):
        self.estacion_meteorologica = estacion_meteorologica
        estacion_meteorologica.registrar_observador(self)
    
    def actualizar(self, temperatura, humedad, presion):
        print(f"Condiciones Actuales: {temperatura}°C, {humedad}% de humedad")

class PantallaEstadisticas(Observador):
    def __init__(self, estacion_meteorologica):
        self.estacion_meteorologica = estacion_meteorologica
        estacion_meteorologica.registrar_observador(self)
        self.min_temp = float('inf')
        self.max_temp = float('-inf')
        self.suma_temp = 0
        self.num_lecturas = 0
    
    def actualizar(self, temperatura, humedad, presion):
        self.suma_temp += temperatura
        self.num_lecturas += 1
        
        if temperatura < self.min_temp:
            self.min_temp = temperatura
        
        if temperatura > self.max_temp:
            self.max_temp = temperatura
        
        print(f"Estadísticas: Temp. Media: {self.suma_temp/self.num_lecturas:.1f}°C, " +
              f"Min: {self.min_temp}°C, Max: {self.max_temp}°C")

class PantallaPronostico(Observador):
    def __init__(self, estacion_meteorologica):
        self.estacion_meteorologica = estacion_meteorologica
        estacion_meteorologica.registrar_observador(self)
        self.ultima_presion = 0
    
    def actualizar(self, temperatura, humedad, presion):
        pronostico = "Estable"
        if presion > self.ultima_presion:
            pronostico = "Mejorando, tiempo despejado"
        elif presion < self.ultima_presion:
            pronostico = "Enfriando, lluvia probable"
        
        self.ultima_presion = presion
        print(f"Pronóstico: {pronostico}")

# Uso del patrón Observer
print("Creando estación meteorológica y observadores...")
estacion = EstacionMeteorologica()

pantalla_actual = PantallaCondicionesActuales(estacion)
pantalla_estadisticas = PantallaEstadisticas(estacion)
pantalla_pronostico = PantallaPronostico(estacion)

print("\nPrimera medición:")
estacion.establecer_mediciones(27, 65, 1013)

print("\nSegunda medición:")
estacion.establecer_mediciones(28, 70, 1012)

print("\nTercera medición:")
estacion.establecer_mediciones(26, 75, 1015)

# Eliminamos un observador
print("\nEliminando pantalla de pronóstico...")
estacion.eliminar_observador(pantalla_pronostico)

print("\nCuarta medición (sin pronóstico):")
estacion.establecer_mediciones(29, 60, 1014)

# ======================================================
# PARTE 4: PATRÓN STRATEGY
# ======================================================
# "Define una familia de algoritmos, encapsula cada uno, y los hace intercambiables.
# Permite que el algoritmo varíe independientemente de los clientes que lo usan."
print("\n=== PATRÓN STRATEGY ===")

# Interfaz Strategy
class EstrategiaPago(ABC):
    @abstractmethod
    def pagar(self, monto):
        pass

# Estrategias concretas
class PagoTarjetaCredito(EstrategiaPago):
    def __init__(self, nombre, numero_tarjeta, cvv, fecha_exp):
        self.nombre = nombre
        self.numero_tarjeta = numero_tarjeta
        self.cvv = cvv
        self.fecha_exp = fecha_exp
    
    def pagar(self, monto):
        # En una implementación real, aquí iría la lógica para procesar el pago
        print(f"Pago de ${monto} procesado con tarjeta de crédito.")
        print(f"  Titular: {self.nombre}")
        print(f"  Tarjeta: {self.numero_tarjeta[:4]}...{self.numero_tarjeta[-4:]}")
        return True

class PagoPaypal(EstrategiaPago):
    def __init__(self, email, password):
        self.email = email
        self.password = password
    
    def pagar(self, monto):
        # En una implementación real, aquí iría la lógica para procesar el pago
        print(f"Pago de ${monto} procesado con PayPal.")
        print(f"  Cuenta: {self.email}")
        return True

class PagoCriptomoneda(EstrategiaPago):
    def __init__(self, direccion_wallet):
        self.direccion_wallet = direccion_wallet
    
    def pagar(self, monto):
        # En una implementación real, aquí iría la lógica para procesar el pago
        print(f"Pago de ${monto} procesado con criptomoneda.")
        print(f"  Wallet: {self.direccion_wallet[:6]}...{self.direccion_wallet[-6:]}")
        return True

# Contexto que usa la estrategia
class CarritoCompra:
    def __init__(self):
        self.productos = []
        self.estrategia_pago = None
    
    def agregar_producto(self, nombre, precio):
        self.productos.append({"nombre": nombre, "precio": precio})
    
    def calcular_total(self):
        return sum(item["precio"] for item in self.productos)
    
    def establecer_estrategia_pago(self, estrategia):
        self.estrategia_pago = estrategia
    
    def procesar_pago(self):
        if not self.estrategia_pago:
            raise ValueError("Debe seleccionar un método de pago")
        
        monto = self.calcular_total()
        return self.estrategia_pago.pagar(monto)

# Uso del patrón Strategy
carrito = CarritoCompra()
carrito.agregar_producto("Laptop", 1200)
carrito.agregar_producto("Mouse", 25)
carrito.agregar_producto("Teclado", 50)

print(f"Total de la compra: ${carrito.calcular_total()}")

# Elegimos la estrategia de pago
print("\nPagando con tarjeta de crédito:")
pago_tarjeta = PagoTarjetaCredito("Juan Pérez", "1234567890123456", "123", "12/25")
carrito.establecer_estrategia_pago(pago_tarjeta)
carrito.procesar_pago()

print("\nPagando con PayPal:")
pago_paypal = PagoPaypal("juan@example.com", "clave123")
carrito.establecer_estrategia_pago(pago_paypal)
carrito.procesar_pago()

print("\nPagando con criptomoneda:")
pago_cripto = PagoCriptomoneda("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
carrito.establecer_estrategia_pago(pago_cripto)
carrito.procesar_pago()

# ======================================================
# PARTE 5: PATRÓN ADAPTER
# ======================================================
# "Convierte la interfaz de una clase en otra interfaz que los clientes esperan.
# Permite que clases trabajen juntas cuando no podrían hacerlo de otra manera."
print("\n=== PATRÓN ADAPTER ===")

# Clase existente con una interfaz incompatible
class MotorGasolina:
    def encender(self):
        return "Motor de gasolina: Encendido"
    
    def acelerar(self):
        return "Motor de gasolina: Acelerando"
    
    def recargar_combustible(self):
        return "Motor de gasolina: Recargando combustible"

# Otra clase con una interfaz diferente
class MotorElectrico:
    def conectar(self):
        return "Motor eléctrico: Conectado"
    
    def activar(self):
        return "Motor eléctrico: Activado"
    
    def mover(self):
        return "Motor eléctrico: Moviendo vehículo"
    
    def cargar(self):
        return "Motor eléctrico: Cargando baterías"

# Interfaz objetivo que espera nuestro cliente
class Motor(ABC):
    @abstractmethod
    def arrancar(self):
        pass
    
    @abstractmethod
    def acelerar(self):
        pass
    
    @abstractmethod
    def recargar(self):
        pass

# Adaptador para el motor de gasolina
class MotorGasolinaAdapter(Motor):
    def __init__(self, motor_gasolina):
        self.motor = motor_gasolina
    
    def arrancar(self):
        return self.motor.encender()
    
    def acelerar(self):
        return self.motor.acelerar()
    
    def recargar(self):
        return self.motor.recargar_combustible()

# Adaptador para el motor eléctrico
class MotorElectricoAdapter(Motor):
    def __init__(self, motor_electrico):
        self.motor = motor_electrico
    
    def arrancar(self):
        return f"{self.motor.conectar()} - {self.motor.activar()}"
    
    def acelerar(self):
        return self.motor.mover()
    
    def recargar(self):
        return self.motor.cargar()

# Cliente que usa la interfaz estandarizada
class Conductor:
    def operar_vehiculo(self, motor):
        acciones = []
        acciones.append(motor.arrancar())
        acciones.append(motor.acelerar())
        return acciones

# Uso del patrón Adapter
print("Vehículo con motor de gasolina:")
motor_gasolina = MotorGasolina()
motor_gasolina_adapter = MotorGasolinaAdapter(motor_gasolina)
conductor = Conductor()

acciones = conductor.operar_vehiculo(motor_gasolina_adapter)
for accion in acciones:
    print(f"  - {accion}")
print(f"  - {motor_gasolina_adapter.recargar()}")

print("\nVehículo con motor eléctrico:")
motor_electrico = MotorElectrico()
motor_electrico_adapter = MotorElectricoAdapter(motor_electrico)

acciones = conductor.operar_vehiculo(motor_electrico_adapter)
for accion in acciones:
    print(f"  - {accion}")
print(f"  - {motor_electrico_adapter.recargar()}")

print("\nVentaja del Adapter: el conductor puede operar cualquier tipo de vehículo sin conocer los detalles de implementación.")

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Implementa el patrón Decorator para añadir funcionalidades adicionales a un objeto de forma dinámica.
#    Crea una clase base Cafe y decoradores como Leche, Azucar y Canela que añadan ingredientes.
# 
# 2. Implementa el patrón Composite para representar una estructura jerárquica de archivos y directorios.
#    Crea una clase base Componente y dos clases derivadas: Archivo y Directorio.
#
# 3. Implementa el patrón Command para encapsular una solicitud como un objeto.
#    Crea comandos para un editor de texto simple (Copiar, Pegar, Borrar) y un historial para deshacer.
#
# 4. Implementa el patrón Proxy para controlar el acceso a un objeto.
#    Crea un proxy que controle el acceso a un servicio de base de datos, verificando permisos antes de cada operación.
#
# 5. Implementa el patrón Template Method para definir el esqueleto de un algoritmo.
#    Crea una clase base para procesar documentos y subclases para diferentes tipos (CSV, XML, JSON).
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
# Ejercicio 1: Patrón Decorator
from abc import ABC, abstractmethod

class CafeComponente(ABC):
    @abstractmethod
    def obtener_descripcion(self):
        pass
    
    @abstractmethod
    def calcular_costo(self):
        pass

class CafeBase(CafeComponente):
    def obtener_descripcion(self):
        return "Café simple"
    
    def calcular_costo(self):
        return 1.0

class DecoradorCafe(CafeComponente):
    def __init__(self, cafe_componente):
        self.cafe_componente = cafe_componente
    
    def obtener_descripcion(self):
        return self.cafe_componente.obtener_descripcion()
    
    def calcular_costo(self):
        return self.cafe_componente.calcular_costo()

class DecoradorLeche(DecoradorCafe):
    def obtener_descripcion(self):
        return f"{self.cafe_componente.obtener_descripcion()}, Leche"
    
    def calcular_costo(self):
        return self.cafe_componente.calcular_costo() + 0.5

class DecoradorAzucar(DecoradorCafe):
    def obtener_descripcion(self):
        return f"{self.cafe_componente.obtener_descripcion()}, Azúcar"
    
    def calcular_costo(self):
        return self.cafe_componente.calcular_costo() + 0.2

class DecoradorCanela(DecoradorCafe):
    def obtener_descripcion(self):
        return f"{self.cafe_componente.obtener_descripcion()}, Canela"
    
    def calcular_costo(self):
        return self.cafe_componente.calcular_costo() + 0.3

# Prueba del patrón Decorator
cafe = CafeBase()
print(f"{cafe.obtener_descripcion()}: ${cafe.calcular_costo():.2f}")

cafe_con_leche = DecoradorLeche(cafe)
print(f"{cafe_con_leche.obtener_descripcion()}: ${cafe_con_leche.calcular_costo():.2f}")

cafe_con_leche_y_azucar = DecoradorAzucar(cafe_con_leche)
print(f"{cafe_con_leche_y_azucar.obtener_descripcion()}: ${cafe_con_leche_y_azucar.calcular_costo():.2f}")

cafe_completo = DecoradorCanela(cafe_con_leche_y_azucar)
print(f"{cafe_completo.obtener_descripcion()}: ${cafe_completo.calcular_costo():.2f}")

# Ejercicio 2: Patrón Composite
from abc import ABC, abstractmethod

class Componente(ABC):
    def __init__(self, nombre):
        self.nombre = nombre
    
    @abstractmethod
    def mostrar(self, nivel=0):
        pass
    
    @abstractmethod
    def obtener_tamaño(self):
        pass

class Archivo(Componente):
    def __init__(self, nombre, tamaño):
        super().__init__(nombre)
        self.tamaño = tamaño
    
    def mostrar(self, nivel=0):
        return "  " * nivel + f"- {self.nombre} ({self.tamaño} KB)"
    
    def obtener_tamaño(self):
        return self.tamaño

class Directorio(Componente):
    def __init__(self, nombre):
        super().__init__(nombre)
        self.hijos = []
    
    def añadir(self, componente):
        self.hijos.append(componente)
    
    def eliminar(self, componente):
        self.hijos.remove(componente)
    
    def mostrar(self, nivel=0):
        resultado = "  " * nivel + f"+ {self.nombre} ({self.obtener_tamaño()} KB)\n"
        
        for hijo in self.hijos:
            resultado += hijo.mostrar(nivel + 1) + "\n"
        
        return resultado.rstrip()
    
    def obtener_tamaño(self):
        total = 0
        for hijo in self.hijos:
            total += hijo.obtener_tamaño()
        return total

# Prueba del patrón Composite
raiz = Directorio("raiz")

documentos = Directorio("Documentos")
documentos.añadir(Archivo("documento1.txt", 10))
documentos.añadir(Archivo("documento2.txt", 15))
documentos.añadir(Archivo("documento3.txt", 8))

imagenes = Directorio("Imágenes")
imagenes.añadir(Archivo("imagen1.jpg", 200))
imagenes.añadir(Archivo("imagen2.jpg", 300))

descargas = Directorio("Descargas")
descargas.añadir(Archivo("archivo1.zip", 2000))
descargas.añadir(Archivo("archivo2.exe", 4000))

documentos_imagenes = Directorio("Documentos con imágenes")
documentos_imagenes.añadir(Archivo("doc_con_imagen1.docx", 500))
documentos_imagenes.añadir(Archivo("doc_con_imagen2.docx", 700))
imagenes.añadir(documentos_imagenes)

raiz.añadir(documentos)
raiz.añadir(imagenes)
raiz.añadir(descargas)

print(raiz.mostrar())
print(f"Tamaño total: {raiz.obtener_tamaño()} KB")

# Ejercicio 3: Patrón Command
from abc import ABC, abstractmethod

# Interfaz Command
class Comando(ABC):
    @abstractmethod
    def ejecutar(self):
        pass
    
    @abstractmethod
    def deshacer(self):
        pass

# Receiver
class EditorTexto:
    def __init__(self):
        self.texto = ""
        self.portapapeles = ""
        self.posicion_cursor = 0
    
    def obtener_texto(self):
        return self.texto
    
    def insertar_texto(self, texto):
        self.texto = self.texto[:self.posicion_cursor] + texto + self.texto[self.posicion_cursor:]
        self.posicion_cursor += len(texto)
    
    def borrar_texto(self, inicio, fin):
        texto_borrado = self.texto[inicio:fin]
        self.texto = self.texto[:inicio] + self.texto[fin:]
        self.posicion_cursor = inicio
        return texto_borrado
    
    def seleccionar(self, inicio, fin):
        self.inicio_seleccion = inicio
        self.fin_seleccion = fin
        return self.texto[inicio:fin]
    
    def copiar(self, texto):
        self.portapapeles = texto
    
    def pegar(self, posicion):
        self.insertar_texto(self.portapapeles)
    
    def obtener_portapapeles(self):
        return self.portapapeles

# Comandos concretos
class ComandoInsertar(Comando):
    def __init__(self, editor, texto):
        self.editor = editor
        self.texto = texto
        self.posicion_cursor_anterior = editor.posicion_cursor
    
    def ejecutar(self):
        self.editor.insertar_texto(self.texto)
    
    def deshacer(self):
        self.editor.borrar_texto(self.posicion_cursor_anterior, self.posicion_cursor_anterior + len(self.texto))

class ComandoBorrar(Comando):
    def __init__(self, editor, inicio, fin):
        self.editor = editor
        self.inicio = inicio
        self.fin = fin
        self.texto_borrado = None
    
    def ejecutar(self):
        self.texto_borrado = self.editor.borrar_texto(self.inicio, self.fin)
    
    def deshacer(self):
        self.editor.insertar_texto(self.texto_borrado)

class ComandoCopiar(Comando):
    def __init__(self, editor, inicio, fin):
        self.editor = editor
        self.inicio = inicio
        self.fin = fin
        self.portapapeles_anterior = editor.obtener_portapapeles()
    
    def ejecutar(self):
        texto_seleccionado = self.editor.seleccionar(self.inicio, self.fin)
        self.editor.copiar(texto_seleccionado)
    
    def deshacer(self):
        self.editor.copiar(self.portapapeles_anterior)

class ComandoPegar(Comando):
    def __init__(self, editor):
        self.editor = editor
        self.posicion_cursor_anterior = editor.posicion_cursor
        self.texto_pegado = editor.obtener_portapapeles()
    
    def ejecutar(self):
        self.editor.pegar(self.posicion_cursor_anterior)
    
    def deshacer(self):
        self.editor.borrar_texto(self.posicion_cursor_anterior, 
                                 self.posicion_cursor_anterior + len(self.texto_pegado))

# Invocador
class HistorialComandos:
    def __init__(self):
        self.historial = []
        self.posicion = -1
    
    def ejecutar_comando(self, comando):
        # Si hemos retrocedido y ejecutamos un nuevo comando, descartamos los comandos futuros
        if self.posicion < len(self.historial) - 1:
            self.historial = self.historial[:self.posicion + 1]
        
        comando.ejecutar()
        self.historial.append(comando)
        self.posicion += 1
    
    def deshacer(self):
        if self.posicion >= 0:
            self.historial[self.posicion].deshacer()
            self.posicion -= 1
            return True
        return False
    
    def rehacer(self):
        if self.posicion < len(self.historial) - 1:
            self.posicion += 1
            self.historial[self.posicion].ejecutar()
            return True
        return False

# Cliente
editor = EditorTexto()
historial = HistorialComandos()

# Ejecutamos algunos comandos
historial.ejecutar_comando(ComandoInsertar(editor, "Hola "))
historial.ejecutar_comando(ComandoInsertar(editor, "mundo!"))
print(f"Texto actual: '{editor.obtener_texto()}'")

# Copiamos "mundo"
historial.ejecutar_comando(ComandoCopiar(editor, 5, 10))
print(f"Portapapeles: '{editor.obtener_portapapeles()}'")

# Borramos "mundo"
historial.ejecutar_comando(ComandoBorrar(editor, 5, 10))
print(f"Texto después de borrar: '{editor.obtener_texto()}'")

# Pegamos lo que habíamos copiado
historial.ejecutar_comando(ComandoPegar(editor))
print(f"Texto después de pegar: '{editor.obtener_texto()}'")

# Deshacemos (deshacer pegar)
historial.deshacer()
print(f"Texto después de deshacer: '{editor.obtener_texto()}'")

# Deshacemos de nuevo (deshacer borrar)
historial.deshacer()
print(f"Texto después de deshacer de nuevo: '{editor.obtener_texto()}'")

# Rehacemos (rehacer borrar)
historial.rehacer()
print(f"Texto después de rehacer: '{editor.obtener_texto()}'")

# Ejercicio 4: Patrón Proxy
from abc import ABC, abstractmethod

# Interfaz Sujeto
class ServicioDB(ABC):
    @abstractmethod
    def consultar(self, consulta):
        pass
    
    @abstractmethod
    def actualizar(self, consulta):
        pass
    
    @abstractmethod
    def eliminar(self, consulta):
        pass

# Sujeto Real
class ServicioDBReal(ServicioDB):
    def __init__(self):
        print("Iniciando conexión con la base de datos...")
        # Aquí iría el código para conectar con la base de datos
    
    def consultar(self, consulta):
        print(f"Ejecutando consulta: {consulta}")
        return ["Resultado 1", "Resultado 2", "Resultado 3"]
    
    def actualizar(self, consulta):
        print(f"Actualizando con: {consulta}")
        return True
    
    def eliminar(self, consulta):
        print(f"Eliminando con: {consulta}")
        return True
    
    def __del__(self):
        print("Cerrando conexión con la base de datos...")

# Proxy
class ProxyServicioDB(ServicioDB):
    def __init__(self, usuario, contraseña):
        self.usuario = usuario
        self.contraseña = contraseña
        self.servicio_real = None
    
    def autenticar(self):
        # Aquí iría la lógica real de autenticación
        return self.usuario == "admin" and self.contraseña == "admin123"
    
    def tiene_permiso(self, operacion):
        # Aquí iría la lógica real de verificación de permisos
        if self.usuario == "admin":
            return True
        elif operacion == "consultar":
            return True
        else:
            return False
    
    def obtener_servicio_real(self):
        if not self.servicio_real:
            self.servicio_real = ServicioDBReal()
        return self.servicio_real
    
    def consultar(self, consulta):
        if not self.autenticar():
            return "Error: Autenticación fallida"
        
        if not self.tiene_permiso("consultar"):
            return "Error: No tiene permisos para consultar"
        
        return self.obtener_servicio_real().consultar(consulta)
    
    def actualizar(self, consulta):
        if not self.autenticar():
            return "Error: Autenticación fallida"
        
        if not self.tiene_permiso("actualizar"):
            return "Error: No tiene permisos para actualizar"
        
        return self.obtener_servicio_real().actualizar(consulta)
    
    def eliminar(self, consulta):
        if not self.autenticar():
            return "Error: Autenticación fallida"
        
        if not self.tiene_permiso("eliminar"):
            return "Error: No tiene permisos para eliminar"
        
        return self.obtener_servicio_real().eliminar(consulta)

# Cliente
print("\n--- Usuario Admin ---")
proxy_admin = ProxyServicioDB("admin", "admin123")
print(proxy_admin.consultar("SELECT * FROM usuarios"))
print(proxy_admin.actualizar("UPDATE usuarios SET nombre='Nuevo' WHERE id=1"))
print(proxy_admin.eliminar("DELETE FROM usuarios WHERE id=2"))

print("\n--- Usuario Lector ---")
proxy_lector = ProxyServicioDB("lector", "password123")
print(proxy_lector.consultar("SELECT * FROM productos"))
print(proxy_lector.actualizar("UPDATE productos SET precio=100 WHERE id=1"))
print(proxy_lector.eliminar("DELETE FROM productos WHERE id=2"))

print("\n--- Usuario con credenciales incorrectas ---")
proxy_incorrecto = ProxyServicioDB("usuario", "clave_incorrecta")
print(proxy_incorrecto.consultar("SELECT * FROM clientes"))

# Ejercicio 5: Patrón Template Method
from abc import ABC, abstractmethod

class ProcesadorDocumento(ABC):
    # Método plantilla
    def procesar(self, ruta_archivo):
        datos = self.leer_archivo(ruta_archivo)
        datos_procesados = self.parsear_datos(datos)
        resultados = self.procesar_datos(datos_procesados)
        self.generar_reporte(resultados)
        return resultados
    
    def leer_archivo(self, ruta_archivo):
        print(f"Leyendo archivo {ruta_archivo}...")
        # Aquí iría el código real para leer el archivo
        return f"Contenido del archivo {ruta_archivo}"
    
    @abstractmethod
    def parsear_datos(self, datos):
        pass
    
    def procesar_datos(self, datos):
        print("Procesando datos genéricos...")
        return f"Datos procesados: {datos}"
    
    def generar_reporte(self, resultados):
        print(f"Generando reporte con resultados: {resultados}")

class ProcesadorCSV(ProcesadorDocumento):
    def parsear_datos(self, datos):
        print("Parseando datos CSV...")
        # Aquí iría el código real para parsear CSV
        return f"Datos CSV parseados de: {datos}"
    
    # Sobreescribimos para personalizar
    def generar_reporte(self, resultados):
        print(f"Generando reporte CSV con resultados: {resultados}")
        print("El reporte incluye gráficos de barras")

class ProcesadorXML(ProcesadorDocumento):
    def parsear_datos(self, datos):
        print("Parseando datos XML...")
        # Aquí iría el código real para parsear XML
        return f"Datos XML parseados de: {datos}"
    
    # Sobreescribimos para personalizar
    def procesar_datos(self, datos):
        print("Procesando datos XML con validación especial...")
        return f"Datos XML procesados y validados: {datos}"

class ProcesadorJSON(ProcesadorDocumento):
    def parsear_datos(self, datos):
        print("Parseando datos JSON...")
        # Aquí iría el código real para parsear JSON
        return f"Datos JSON parseados de: {datos}"
    
    # No sobreescribimos los otros métodos, usando la implementación por defecto

# Cliente
print("\n--- Procesamiento de CSV ---")
procesador_csv = ProcesadorCSV()
procesador_csv.procesar("datos.csv")

print("\n--- Procesamiento de XML ---")
procesador_xml = ProcesadorXML()
procesador_xml.procesar("datos.xml")

print("\n--- Procesamiento de JSON ---")
procesador_json = ProcesadorJSON()
procesador_json.procesar("datos.json")
'''

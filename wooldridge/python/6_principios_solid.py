"""
SCRIPT 6: PRINCIPIOS SOLID
==========================

Este script explora los principios SOLID de diseño orientado a objetos.
Aprenderás:
1. S - Principio de Responsabilidad Única (SRP)
2. O - Principio de Abierto/Cerrado (OCP)
3. L - Principio de Sustitución de Liskov (LSP)
4. I - Principio de Segregación de Interfaces (ISP)
5. D - Principio de Inversión de Dependencias (DIP)
"""

# ======================================================
# PARTE 1: S - PRINCIPIO DE RESPONSABILIDAD ÚNICA (SRP)
# ======================================================
# "Una clase debe tener una y solo una razón para cambiar"
print("=== PRINCIPIO DE RESPONSABILIDAD ÚNICA (SRP) ===")

# Mal enfoque (violando SRP):
class LibroMalEnfoque:
    def __init__(self, titulo, autor, isbn):
        self.titulo = titulo
        self.autor = autor
        self.isbn = isbn
        self.contenido = ""
    
    # Responsabilidad 1: Gestionar propiedades del libro
    def obtener_info(self):
        return f"{self.titulo} por {self.autor} (ISBN: {self.isbn})"
    
    # Responsabilidad 2: Manejo del contenido
    def escribir_contenido(self, texto):
        self.contenido += texto
    
    def eliminar_contenido(self):
        self.contenido = ""
    
    # Responsabilidad 3: Gestión de archivos (!!!)
    def guardar_a_archivo(self, ruta):
        with open(ruta, 'w') as archivo:
            archivo.write(f"Título: {self.titulo}\n")
            archivo.write(f"Autor: {self.autor}\n")
            archivo.write(f"ISBN: {self.isbn}\n")
            archivo.write(f"Contenido: {self.contenido}")
    
    def cargar_desde_archivo(self, ruta):
        with open(ruta, 'r') as archivo:
            lineas = archivo.readlines()
            self.titulo = lineas[0].split(': ')[1].strip()
            self.autor = lineas[1].split(': ')[1].strip()
            self.isbn = lineas[2].split(': ')[1].strip()
            self.contenido = ': '.join(lineas[3].split(': ')[1:]).strip()
    
    # Responsabilidad 4: Imprimir (!!!)
    def imprimir_libro(self, impresora):
        print(f"Enviando a imprimir a {impresora}: {self.titulo}")
        # Lógica de impresión aquí...

print("Mal enfoque (violando SRP):")
libro_malo = LibroMalEnfoque("El Principito", "Antoine de Saint-Exupéry", "9788498381498")
print(libro_malo.obtener_info())
libro_malo.escribir_contenido("Era una vez...")
print(f"Contenido: {libro_malo.contenido}")
print("El problema: esta clase tiene demasiadas responsabilidades diferentes.")
print()

# Buen enfoque (siguiendo SRP):
# Cada clase tiene una única responsabilidad

class Libro:
    def __init__(self, titulo, autor, isbn):
        self.titulo = titulo
        self.autor = autor
        self.isbn = isbn
        self.contenido = ""
    
    def obtener_info(self):
        return f"{self.titulo} por {self.autor} (ISBN: {self.isbn})"
    
    def escribir_contenido(self, texto):
        self.contenido += texto
    
    def eliminar_contenido(self):
        self.contenido = ""

class PersistenciaLibro:
    @staticmethod
    def guardar_libro(libro, ruta):
        with open(ruta, 'w') as archivo:
            archivo.write(f"Título: {libro.titulo}\n")
            archivo.write(f"Autor: {libro.autor}\n")
            archivo.write(f"ISBN: {libro.isbn}\n")
            archivo.write(f"Contenido: {libro.contenido}")
    
    @staticmethod
    def cargar_libro(ruta):
        with open(ruta, 'r') as archivo:
            lineas = archivo.readlines()
            titulo = lineas[0].split(': ')[1].strip()
            autor = lineas[1].split(': ')[1].strip()
            isbn = lineas[2].split(': ')[1].strip()
            contenido = ': '.join(lineas[3].split(': ')[1:]).strip()
            
            libro = Libro(titulo, autor, isbn)
            libro.escribir_contenido(contenido)
            return libro

class ImpresoraLibro:
    @staticmethod
    def imprimir(libro, impresora):
        print(f"Enviando a imprimir a {impresora}: {libro.titulo}")
        # Lógica de impresión aquí...

print("Buen enfoque (siguiendo SRP):")
libro_bueno = Libro("El Principito", "Antoine de Saint-Exupéry", "9788498381498")
print(libro_bueno.obtener_info())
libro_bueno.escribir_contenido("Era una vez...")
print(f"Contenido: {libro_bueno.contenido}")

# Si necesitamos guardar, usamos la clase específica para eso
# PersistenciaLibro.guardar_libro(libro_bueno, "libro.txt")

# Si necesitamos imprimir, usamos la clase específica para eso
ImpresoraLibro.imprimir(libro_bueno, "Impresora HP")

print("Ventaja: cada clase tiene una única responsabilidad, lo que facilita los cambios y el mantenimiento.")

# ======================================================
# PARTE 2: O - PRINCIPIO DE ABIERTO/CERRADO (OCP)
# ======================================================
# "Las entidades de software deben estar abiertas para la extensión pero cerradas para la modificación."
print("\n=== PRINCIPIO DE ABIERTO/CERRADO (OCP) ===")

# Mal enfoque (violando OCP):
class CalculadoraAreaMala:
    def calcular_area(self, figuras):
        area_total = 0
        for figura in figuras:
            if figura.tipo == "rectangulo":
                area_total += figura.ancho * figura.alto
            elif figura.tipo == "circulo":
                import math
                area_total += math.pi * (figura.radio ** 2)
            # Si queremos añadir un triángulo, tendríamos que MODIFICAR esta clase
            # elif figura.tipo == "triangulo":
            #     area_total += 0.5 * figura.base * figura.altura
        return area_total

print("Mal enfoque (violando OCP):")
print("Para añadir un nuevo tipo de figura, tendríamos que modificar la clase CalculadoraAreaMala.")
print()

# Buen enfoque (siguiendo OCP):
from abc import ABC, abstractmethod

class Figura(ABC):
    @abstractmethod
    def calcular_area(self):
        pass

class Rectangulo(Figura):
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
    
    def calcular_area(self):
        return self.ancho * self.alto

class Circulo(Figura):
    def __init__(self, radio):
        self.radio = radio
    
    def calcular_area(self):
        import math
        return math.pi * (self.radio ** 2)

class CalculadoraArea:
    def calcular_area_total(self, figuras):
        return sum(figura.calcular_area() for figura in figuras)

# Si queremos añadir un triángulo, NO MODIFICAMOS las clases existentes, creamos una nueva:
class Triangulo(Figura):
    def __init__(self, base, altura):
        self.base = base
        self.altura = altura
    
    def calcular_area(self):
        return 0.5 * self.base * self.altura

print("Buen enfoque (siguiendo OCP):")
rectangulo = Rectangulo(5, 4)
circulo = Circulo(3)
triangulo = Triangulo(6, 8)

calculadora = CalculadoraArea()
figuras = [rectangulo, circulo, triangulo]
area_total = calculadora.calcular_area_total(figuras)

print(f"Área del rectángulo: {rectangulo.calcular_area()}")
print(f"Área del círculo: {circulo.calcular_area():.2f}")
print(f"Área del triángulo: {triangulo.calcular_area()}")
print(f"Área total: {area_total:.2f}")
print("Ventaja: podemos añadir nuevas figuras sin modificar el código existente.")

# ======================================================
# PARTE 3: L - PRINCIPIO DE SUSTITUCIÓN DE LISKOV (LSP)
# ======================================================
# "Si S es un subtipo de T, entonces los objetos de tipo T pueden ser reemplazados por objetos de tipo S 
# sin alterar las propiedades deseables del programa."
print("\n=== PRINCIPIO DE SUSTITUCIÓN DE LISKOV (LSP) ===")

# Mal enfoque (violando LSP):
class Ave:
    def volar(self):
        return "Volando alto..."

class Pinguino(Ave):  # Los pingüinos son aves pero no vuelan
    def volar(self):
        raise NotImplementedError("Los pingüinos no pueden volar")  # ¡ROMPE LSP!

print("Mal enfoque (violando LSP):")
try:
    ave = Ave()
    pinguino = Pinguino()
    
    aves = [ave, pinguino]
    for a in aves:
        print(a.volar())  # Esto fallará en el pingüino
except NotImplementedError as e:
    print(f"Error: {e}")
print("El problema: no podemos sustituir un Ave por un Pingüino sin que falle el programa.")
print()

# Buen enfoque (siguiendo LSP):
class AveLSP:
    def comer(self):
        return "Comiendo..."

class AveVoladora(AveLSP):
    def volar(self):
        return "Volando alto..."

class AveNoVoladora(AveLSP):
    def caminar(self):
        return "Caminando..."

class Paloma(AveVoladora):
    def volar(self):
        return "La paloma está volando"

class PinguinoLSP(AveNoVoladora):
    def caminar(self):
        return "El pingüino está caminando"
    
    def nadar(self):
        return "El pingüino está nadando"

print("Buen enfoque (siguiendo LSP):")
ave = AveLSP()
paloma = Paloma()
pinguino = PinguinoLSP()

# Todas las aves pueden comer
aves = [ave, paloma, pinguino]
for a in aves:
    print(a.comer())  # Todas las aves pueden sustituir a la clase base para comer

# Solo las aves voladoras pueden volar
aves_voladoras = [paloma]
for a in aves_voladoras:
    print(a.volar())

# Solo las aves no voladoras pueden caminar
aves_no_voladoras = [pinguino]
for a in aves_no_voladoras:
    print(a.caminar())
    print(a.nadar())

print("Ventaja: cada subclase puede sustituir a su clase base sin causar errores.")

# ======================================================
# PARTE 4: I - PRINCIPIO DE SEGREGACIÓN DE INTERFACES (ISP)
# ======================================================
# "Los clientes no deben ser forzados a depender de interfaces que no usan."
print("\n=== PRINCIPIO DE SEGREGACIÓN DE INTERFACES (ISP) ===")

# Mal enfoque (violando ISP):
class DispositivoMultifuncionMalo(ABC):
    @abstractmethod
    def imprimir(self, documento):
        pass
    
    @abstractmethod
    def escanear(self, documento):
        pass
    
    @abstractmethod
    def enviar_fax(self, documento):
        pass
    
    @abstractmethod
    def fotocopiar(self, documento):
        pass

class ImpresoraSimpleMala(DispositivoMultifuncionMalo):
    def imprimir(self, documento):
        return f"Imprimiendo {documento}"
    
    def escanear(self, documento):
        raise NotImplementedError("Esta impresora no puede escanear")  # Método no soportado
    
    def enviar_fax(self, documento):
        raise NotImplementedError("Esta impresora no puede enviar fax")  # Método no soportado
    
    def fotocopiar(self, documento):
        raise NotImplementedError("Esta impresora no puede fotocopiar")  # Método no soportado

print("Mal enfoque (violando ISP):")
impresora_mala = ImpresoraSimpleMala()
print(impresora_mala.imprimir("documento.pdf"))

try:
    impresora_mala.escanear("imagen.jpg")
except NotImplementedError as e:
    print(f"Error: {e}")
print("El problema: la impresora simple está forzada a implementar métodos que no necesita.")
print()

# Buen enfoque (siguiendo ISP):
class Impresora(ABC):
    @abstractmethod
    def imprimir(self, documento):
        pass

class Escaner(ABC):
    @abstractmethod
    def escanear(self, documento):
        pass

class Fax(ABC):
    @abstractmethod
    def enviar_fax(self, documento):
        pass

class Fotocopiadora(ABC):
    @abstractmethod
    def fotocopiar(self, documento):
        pass

# Ahora podemos implementar solo las interfaces que necesitamos
class ImpresoraSimple(Impresora):
    def imprimir(self, documento):
        return f"Imprimiendo {documento}"

class DispositivoMultifuncion(Impresora, Escaner, Fax, Fotocopiadora):
    def imprimir(self, documento):
        return f"Multifunción: Imprimiendo {documento}"
    
    def escanear(self, documento):
        return f"Multifunción: Escaneando {documento}"
    
    def enviar_fax(self, documento):
        return f"Multifunción: Enviando fax de {documento}"
    
    def fotocopiar(self, documento):
        return f"Multifunción: Fotocopiando {documento}"

print("Buen enfoque (siguiendo ISP):")
impresora_simple = ImpresoraSimple()
multifuncion = DispositivoMultifuncion()

print(impresora_simple.imprimir("documento.pdf"))
print(multifuncion.imprimir("documento.pdf"))
print(multifuncion.escanear("imagen.jpg"))
print(multifuncion.enviar_fax("contrato.pdf"))
print(multifuncion.fotocopiar("carta.docx"))

print("Ventaja: cada clase implementa solo los métodos que necesita.")

# ======================================================
# PARTE 5: D - PRINCIPIO DE INVERSIÓN DE DEPENDENCIAS (DIP)
# ======================================================
# "Los módulos de alto nivel no deben depender de módulos de bajo nivel. 
# Ambos deben depender de abstracciones."
print("\n=== PRINCIPIO DE INVERSIÓN DE DEPENDENCIAS (DIP) ===")

# Mal enfoque (violando DIP):
class BaseDatosMysql:
    def guardar(self, datos):
        return f"Guardando {datos} en MySQL"

class GestorUsuariosMalo:
    def __init__(self):
        # Dependencia directa a una implementación concreta
        self.bd = BaseDatosMysql()
    
    def guardar_usuario(self, usuario):
        # Llamamos directamente a la implementación
        return self.bd.guardar(usuario)

print("Mal enfoque (violando DIP):")
gestor_malo = GestorUsuariosMalo()
print(gestor_malo.guardar_usuario("usuario1"))
print("El problema: GestorUsuarios depende directamente de una implementación concreta de base de datos.")
print()

# Buen enfoque (siguiendo DIP):
class BaseDatos(ABC):
    @abstractmethod
    def guardar(self, datos):
        pass

class BaseDatosMysqlDIP(BaseDatos):
    def guardar(self, datos):
        return f"Guardando {datos} en MySQL"

class BaseDatosPostgres(BaseDatos):
    def guardar(self, datos):
        return f"Guardando {datos} en PostgreSQL"

class BaseDatosMemoria(BaseDatos):
    def __init__(self):
        self.datos = []
    
    def guardar(self, datos):
        self.datos.append(datos)
        return f"Guardando {datos} en memoria"

class GestorUsuarios:
    def __init__(self, base_datos: BaseDatos):
        # Dependencia a una abstracción, no a una implementación
        self.bd = base_datos
    
    def guardar_usuario(self, usuario):
        return self.bd.guardar(usuario)

print("Buen enfoque (siguiendo DIP):")
# Podemos usar cualquier implementación concreta de BaseDatos
bd_mysql = BaseDatosMysqlDIP()
bd_postgres = BaseDatosPostgres()
bd_memoria = BaseDatosMemoria()

gestor_mysql = GestorUsuarios(bd_mysql)
gestor_postgres = GestorUsuarios(bd_postgres)
gestor_memoria = GestorUsuarios(bd_memoria)

print(gestor_mysql.guardar_usuario("usuario1"))
print(gestor_postgres.guardar_usuario("usuario2"))
print(gestor_memoria.guardar_usuario("usuario3"))

print("Ventaja: podemos cambiar la implementación de la base de datos sin afectar al gestor de usuarios.")

# ======================================================
# EJERCICIOS PROPUESTOS:
# ======================================================
# 1. Refactoriza la siguiente clase para que cumpla con el principio SRP:
"""
class Empleado:
    def __init__(self, nombre, salario):
        self.nombre = nombre
        self.salario = salario
    
    def calcular_salario(self):
        return self.salario
    
    def guardar_en_base_de_datos(self):
        # Código para guardar en base de datos
        print(f"Guardando empleado {self.nombre} en la base de datos")
    
    def generar_reporte_pdf(self):
        # Código para generar un PDF
        print(f"Generando PDF para {self.nombre}")
    
    def enviar_email(self, destinatario):
        # Código para enviar un email
        print(f"Enviando email sobre {self.nombre} a {destinatario}")
"""

# 2. Refactoriza el siguiente código para cumplir con OCP:
"""
class Notificador:
    def enviar_notificacion(self, tipo, mensaje, destinatario):
        if tipo == "email":
            print(f"Enviando email a {destinatario}: {mensaje}")
        elif tipo == "sms":
            print(f"Enviando SMS a {destinatario}: {mensaje}")
        elif tipo == "push":
            print(f"Enviando notificación push a {destinatario}: {mensaje}")
"""

# 3. Corrige la siguiente violación del principio LSP:
"""
class Rectangulo:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
    
    def set_ancho(self, ancho):
        self.ancho = ancho
    
    def set_alto(self, alto):
        self.alto = alto
    
    def area(self):
        return self.ancho * self.alto

class Cuadrado(Rectangulo):
    def __init__(self, lado):
        super().__init__(lado, lado)
    
    def set_ancho(self, ancho):
        self.ancho = ancho
        self.alto = ancho  # Un cuadrado debe tener lados iguales
    
    def set_alto(self, alto):
        self.alto = alto
        self.ancho = alto  # Un cuadrado debe tener lados iguales

# Esta función asume que el comportamiento de un Rectangulo no cambiará
def procesar_rectangulo(rectangulo):
    rectangulo.set_ancho(5)
    rectangulo.set_alto(4)
    area = rectangulo.area()
    assert area == 20, f"El área debería ser 20, pero es {area}"
"""

# 4. Refactoriza el siguiente código para cumplir con ISP:
"""
class Trabajador(ABC):
    @abstractmethod
    def trabajar(self):
        pass
    
    @abstractmethod
    def comer(self):
        pass
    
    @abstractmethod
    def dormir(self):
        pass

class Robot(Trabajador):
    def trabajar(self):
        return "Robot trabajando..."
    
    def comer(self):
        raise NotImplementedError("Los robots no comen")
    
    def dormir(self):
        raise NotImplementedError("Los robots no duermen")
"""

# 5. Refactoriza el siguiente código para cumplir con DIP:
"""
class EmailSender:
    def enviar(self, mensaje, destinatario):
        print(f"Enviando email a {destinatario}: {mensaje}")

class ServicioNotificacion:
    def __init__(self):
        self.email_sender = EmailSender()
    
    def notificar(self, mensaje, destinatario):
        self.email_sender.enviar(mensaje, destinatario)
"""
"""

# ======================================================
# SOLUCIÓN EJERCICIOS (comentada para que implementen su propia solución)
# ======================================================
'''
# Ejercicio 1: SRP
class Empleado:
    def __init__(self, nombre, salario):
        self.nombre = nombre
        self.salario = salario
    
    def calcular_salario(self):
        return self.salario

class PersistenciaEmpleado:
    @staticmethod
    def guardar_en_base_de_datos(empleado):
        print(f"Guardando empleado {empleado.nombre} en la base de datos")

class GeneradorReportes:
    @staticmethod
    def generar_reporte_pdf(empleado):
        print(f"Generando PDF para {empleado.nombre}")

class NotificadorEmpleado:
    @staticmethod
    def enviar_email(empleado, destinatario):
        print(f"Enviando email sobre {empleado.nombre} a {destinatario}")

# Ejercicio 2: OCP
from abc import ABC, abstractmethod

class Notificador(ABC):
    @abstractmethod
    def enviar(self, mensaje, destinatario):
        pass

class NotificadorEmail(Notificador):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando email a {destinatario}: {mensaje}")

class NotificadorSMS(Notificador):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando SMS a {destinatario}: {mensaje}")

class NotificadorPush(Notificador):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando notificación push a {destinatario}: {mensaje}")

# Si queremos añadir un nuevo tipo de notificación, simplemente creamos una nueva clase:
class NotificadorWhatsApp(Notificador):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando WhatsApp a {destinatario}: {mensaje}")

# Ejercicio 3: LSP
# La jerarquía actual viola LSP porque un Cuadrado no se comporta como un Rectángulo
# Solución: usar una jerarquía diferente

from abc import ABC, abstractmethod

class Forma(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangulo(Forma):
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
    
    def set_ancho(self, ancho):
        self.ancho = ancho
    
    def set_alto(self, alto):
        self.alto = alto
    
    def area(self):
        return self.ancho * self.alto

class Cuadrado(Forma):
    def __init__(self, lado):
        self.lado = lado
    
    def set_lado(self, lado):
        self.lado = lado
    
    def area(self):
        return self.lado * self.lado

# Ahora la función trabaja solo con rectángulos, no asume que un cuadrado es un rectángulo
def procesar_rectangulo(rectangulo):
    if not isinstance(rectangulo, Rectangulo):
        raise TypeError("Se esperaba un Rectangulo")
    
    rectangulo.set_ancho(5)
    rectangulo.set_alto(4)
    area = rectangulo.area()
    assert area == 20, f"El área debería ser 20, pero es {area}"

# Ejercicio 4: ISP
from abc import ABC, abstractmethod

class Trabajador(ABC):
    @abstractmethod
    def trabajar(self):
        pass

class SerVivo(ABC):
    @abstractmethod
    def comer(self):
        pass
    
    @abstractmethod
    def dormir(self):
        pass

class Humano(Trabajador, SerVivo):
    def trabajar(self):
        return "Humano trabajando..."
    
    def comer(self):
        return "Humano comiendo..."
    
    def dormir(self):
        return "Humano durmiendo..."

class Robot(Trabajador):
    def trabajar(self):
        return "Robot trabajando..."

# Ejercicio 5: DIP
from abc import ABC, abstractmethod

class EnviadorMensajes(ABC):
    @abstractmethod
    def enviar(self, mensaje, destinatario):
        pass

class EmailSender(EnviadorMensajes):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando email a {destinatario}: {mensaje}")

class SMSSender(EnviadorMensajes):
    def enviar(self, mensaje, destinatario):
        print(f"Enviando SMS a {destinatario}: {mensaje}")

class ServicioNotificacion:
    def __init__(self, enviador: EnviadorMensajes):
        self.enviador = enviador
    
    def notificar(self, mensaje, destinatario):
        self.enviador.enviar(mensaje, destinatario)

# Uso:
email_sender = EmailSender()
sms_sender = SMSSender()

servicio_email = ServicioNotificacion(email_sender)
servicio_email.notificar("Hola", "usuario@ejemplo.com")

servicio_sms = ServicioNotificacion(sms_sender)
servicio_sms.notificar("Hola", "+1234567890")
'''

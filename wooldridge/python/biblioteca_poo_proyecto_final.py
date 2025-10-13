"""
PROYECTO FINAL: SISTEMA DE BIBLIOTECA
=====================================

Este script implementa un sistema de gestión de biblioteca completo utilizando todos los conceptos
de Programación Orientada a Objetos vistos durante el curso.

Módulos del Sistema:
1. Gestión de Libros
2. Gestión de Usuarios
3. Gestión de Préstamos
4. Persistencia de datos
5. Interfaz de usuario básica
"""

import datetime
import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any

# ======================================================
# INTERFACES Y CLASES ABSTRACTAS
# ======================================================

class Item(ABC):
    """Clase base abstracta para items de la biblioteca"""
    
    @abstractmethod
    def obtener_info(self) -> str:
        """Devuelve información del item en formato legible"""
        pass
    
    @abstractmethod
    def esta_disponible(self) -> bool:
        """Verifica si el item está disponible para préstamo"""
        pass
    
    @abstractmethod
    def a_dict(self) -> Dict:
        """Convierte el objeto a un diccionario para serialización"""
        pass

class Persona(ABC):
    """Clase base abstracta para personas en el sistema"""
    
    @abstractmethod
    def obtener_info(self) -> str:
        """Devuelve información de la persona en formato legible"""
        pass
    
    @abstractmethod
    def a_dict(self) -> Dict:
        """Convierte el objeto a un diccionario para serialización"""
        pass

class Persistencia(ABC):
    """Interfaz para la persistencia de datos"""
    
    @abstractmethod
    def guardar_datos(self, datos: Dict[str, List], ruta: str) -> bool:
        """Guarda los datos en la ubicación especificada"""
        pass
    
    @abstractmethod
    def cargar_datos(self, ruta: str) -> Dict[str, List]:
        """Carga los datos desde la ubicación especificada"""
        pass

# ======================================================
# IMPLEMENTACIÓN DE LIBROS Y OTROS ITEMS
# ======================================================

class Libro(Item):
    """Clase que representa un libro en la biblioteca"""
    
    def __init__(self, id_libro: str, titulo: str, autor: str, anio: int, 
                 genero: str = "", isbn: str = ""):
        self.id = id_libro
        self.titulo = titulo
        self.autor = autor
        self.anio = anio
        self.genero = genero
        self.isbn = isbn
        self._disponible = True
        self._prestamos = []  # Historial de préstamos
    
    def obtener_info(self) -> str:
        """Devuelve información del libro en formato legible"""
        estado = "Disponible" if self._disponible else "Prestado"
        return f"ID: {self.id}, '{self.titulo}' por {self.autor} ({self.anio}) - {estado}"
    
    def esta_disponible(self) -> bool:
        """Verifica si el libro está disponible para préstamo"""
        return self._disponible
    
    def prestar(self, id_usuario: str, fecha: datetime.datetime) -> bool:
        """Marca el libro como prestado"""
        if not self._disponible:
            return False
        
        self._disponible = False
        self._prestamos.append({
            "id_usuario": id_usuario,
            "fecha_prestamo": fecha.strftime("%Y-%m-%d %H:%M:%S"),
            "fecha_devolucion": None
        })
        return True
    
    def devolver(self, fecha: datetime.datetime) -> bool:
        """Marca el libro como devuelto"""
        if self._disponible or not self._prestamos:
            return False
        
        self._disponible = True
        self._prestamos[-1]["fecha_devolucion"] = fecha.strftime("%Y-%m-%d %H:%M:%S")
        return True
    
    def obtener_prestamo_actual(self) -> Optional[Dict]:
        """Devuelve información del préstamo actual si existe"""
        if not self._disponible and self._prestamos and self._prestamos[-1]["fecha_devolucion"] is None:
            return self._prestamos[-1]
        return None
    
    def obtener_historial_prestamos(self) -> List[Dict]:
        """Devuelve el historial completo de préstamos"""
        return self._prestamos
    
    def a_dict(self) -> Dict:
        """Convierte el libro a un diccionario para serialización"""
        return {
            "id": self.id,
            "titulo": self.titulo,
            "autor": self.autor,
            "anio": self.anio,
            "genero": self.genero,
            "isbn": self.isbn,
            "disponible": self._disponible,
            "prestamos": self._prestamos
        }
    
    @classmethod
    def desde_dict(cls, datos: Dict) -> 'Libro':
        """Crea un libro a partir de un diccionario"""
        libro = cls(
            datos["id"], 
            datos["titulo"], 
            datos["autor"], 
            datos["anio"],
            datos.get("genero", ""),
            datos.get("isbn", "")
        )
        libro._disponible = datos.get("disponible", True)
        libro._prestamos = datos.get("prestamos", [])
        return libro

class Revista(Item):
    """Clase que representa una revista en la biblioteca"""
    
    def __init__(self, id_revista: str, titulo: str, editorial: str, numero: int, 
                 fecha_publicacion: str):
        self.id = id_revista
        self.titulo = titulo
        self.editorial = editorial
        self.numero = numero
        self.fecha_publicacion = fecha_publicacion
        self._disponible = True
        self._prestamos = []  # Historial de préstamos
    
    def obtener_info(self) -> str:
        """Devuelve información de la revista en formato legible"""
        estado = "Disponible" if self._disponible else "Prestada"
        return f"ID: {self.id}, '{self.titulo}' #{self.numero} ({self.fecha_publicacion}) - {estado}"
    
    def esta_disponible(self) -> bool:
        """Verifica si la revista está disponible para préstamo"""
        return self._disponible
    
    def prestar(self, id_usuario: str, fecha: datetime.datetime) -> bool:
        """Marca la revista como prestada"""
        if not self._disponible:
            return False
        
        self._disponible = False
        self._prestamos.append({
            "id_usuario": id_usuario,
            "fecha_prestamo": fecha.strftime("%Y-%m-%d %H:%M:%S"),
            "fecha_devolucion": None
        })
        return True
    
    def devolver(self, fecha: datetime.datetime) -> bool:
        """Marca la revista como devuelta"""
        if self._disponible or not self._prestamos:
            return False
        
        self._disponible = True
        self._prestamos[-1]["fecha_devolucion"] = fecha.strftime("%Y-%m-%d %H:%M:%S")
        return True
    
    def obtener_prestamo_actual(self) -> Optional[Dict]:
        """Devuelve información del préstamo actual si existe"""
        if not self._disponible and self._prestamos and self._prestamos[-1]["fecha_devolucion"] is None:
            return self._prestamos[-1]
        return None
    
    def obtener_historial_prestamos(self) -> List[Dict]:
        """Devuelve el historial completo de préstamos"""
        return self._prestamos
    
    def a_dict(self) -> Dict:
        """Convierte la revista a un diccionario para serialización"""
        return {
            "id": self.id,
            "titulo": self.titulo,
            "editorial": self.editorial,
            "numero": self.numero,
            "fecha_publicacion": self.fecha_publicacion,
            "disponible": self._disponible,
            "prestamos": self._prestamos
        }
    
    @classmethod
    def desde_dict(cls, datos: Dict) -> 'Revista':
        """Crea una revista a partir de un diccionario"""
        revista = cls(
            datos["id"], 
            datos["titulo"], 
            datos["editorial"], 
            datos["numero"],
            datos["fecha_publicacion"]
        )
        revista._disponible = datos.get("disponible", True)
        revista._prestamos = datos.get("prestamos", [])
        return revista

# ======================================================
# IMPLEMENTACIÓN DE USUARIOS
# ======================================================

class Usuario(Persona):
    """Clase que representa un usuario de la biblioteca"""
    
    def __init__(self, id_usuario: str, nombre: str, email: str, telefono: str = ""):
        self.id = id_usuario
        self.nombre = nombre
        self.email = email
        self.telefono = telefono
        self.fecha_registro = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._prestamos_actuales = []  # IDs de los items actualmente prestados
        self._historial_prestamos = []  # Historial completo de préstamos
    
    def obtener_info(self) -> str:
        """Devuelve información del usuario en formato legible"""
        return f"ID: {self.id}, Nombre: {self.nombre}, Email: {self.email}, Préstamos activos: {len(self._prestamos_actuales)}"
    
    def puede_pedir_prestado(self, max_prestamos: int = 3) -> bool:
        """Verifica si el usuario puede pedir más items prestados"""
        return len(self._prestamos_actuales) < max_prestamos
    
    def registrar_prestamo(self, id_item: str, fecha: datetime.datetime) -> bool:
        """Registra un nuevo préstamo para el usuario"""
        if id_item in self._prestamos_actuales:
            return False
        
        self._prestamos_actuales.append(id_item)
        self._historial_prestamos.append({
            "id_item": id_item,
            "fecha_prestamo": fecha.strftime("%Y-%m-%d %H:%M:%S"),
            "fecha_devolucion": None
        })
        return True
    
    def registrar_devolucion(self, id_item: str, fecha: datetime.datetime) -> bool:
        """Registra la devolución de un item"""
        if id_item not in self._prestamos_actuales:
            return False
        
        self._prestamos_actuales.remove(id_item)
        
        # Actualizar el historial
        for prestamo in reversed(self._historial_prestamos):
            if prestamo["id_item"] == id_item and prestamo["fecha_devolucion"] is None:
                prestamo["fecha_devolucion"] = fecha.strftime("%Y-%m-%d %H:%M:%S")
                break
        
        return True
    
    def obtener_prestamos_actuales(self) -> List[str]:
        """Devuelve la lista de IDs de los items actualmente prestados"""
        return self._prestamos_actuales
    
    def obtener_historial_prestamos(self) -> List[Dict]:
        """Devuelve el historial completo de préstamos"""
        return self._historial_prestamos
    
    def a_dict(self) -> Dict:
        """Convierte el usuario a un diccionario para serialización"""
        return {
            "id": self.id,
            "nombre": self.nombre,
            "email": self.email,
            "telefono": self.telefono,
            "fecha_registro": self.fecha_registro,
            "prestamos_actuales": self._prestamos_actuales,
            "historial_prestamos": self._historial_prestamos
        }
    
    @classmethod
    def desde_dict(cls, datos: Dict) -> 'Usuario':
        """Crea un usuario a partir de un diccionario"""
        usuario = cls(
            datos["id"], 
            datos["nombre"], 
            datos["email"],
            datos.get("telefono", "")
        )
        usuario.fecha_registro = datos.get("fecha_registro", 
                                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        usuario._prestamos_actuales = datos.get("prestamos_actuales", [])
        usuario._historial_prestamos = datos.get("historial_prestamos", [])
        return usuario

class Bibliotecario(Persona):
    """Clase que representa un bibliotecario"""
    
    def __init__(self, id_bibliotecario: str, nombre: str, email: str, password: str):
        self.id = id_bibliotecario
        self.nombre = nombre
        self.email = email
        self._password = password  # Atributo protegido
        self._acciones = []  # Registro de acciones
    
    def obtener_info(self) -> str:
        """Devuelve información del bibliotecario en formato legible"""
        return f"ID: {self.id}, Nombre: {self.nombre}, Email: {self.email}"
    
    def verificar_password(self, password: str) -> bool:
        """Verifica si la contraseña es correcta"""
        return self._password == password
    
    def cambiar_password(self, password_actual: str, nueva_password: str) -> bool:
        """Cambia la contraseña del bibliotecario"""
        if not self.verificar_password(password_actual):
            return False
        
        self._password = nueva_password
        self._registrar_accion("Cambio de contraseña")
        return True
    
    def _registrar_accion(self, accion: str) -> None:
        """Registra una acción realizada por el bibliotecario"""
        self._acciones.append({
            "accion": accion,
            "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def a_dict(self) -> Dict:
        """Convierte el bibliotecario a un diccionario para serialización"""
        return {
            "id": self.id,
            "nombre": self.nombre,
            "email": self.email,
            "password": self._password,
            "acciones": self._acciones
        }
    
    @classmethod
    def desde_dict(cls, datos: Dict) -> 'Bibliotecario':
        """Crea un bibliotecario a partir de un diccionario"""
        bibliotecario = cls(
            datos["id"], 
            datos["nombre"], 
            datos["email"],
            datos["password"]
        )
        bibliotecario._acciones = datos.get("acciones", [])
        return bibliotecario

# ======================================================
# IMPLEMENTACIÓN DE PERSISTENCIA
# ======================================================

class PersistenciaJSON(Persistencia):
    """Implementación de persistencia usando archivos JSON"""
    
    def guardar_datos(self, datos: Dict[str, List], ruta: str) -> bool:
        """Guarda los datos en un archivo JSON"""
        try:
            with open(ruta, 'w') as archivo:
                json.dump(datos, archivo, indent=2)
            return True
        except Exception as e:
            print(f"Error al guardar datos: {e}")
            return False
    
    def cargar_datos(self, ruta: str) -> Dict[str, List]:
        """Carga los datos desde un archivo JSON"""
        if not os.path.exists(ruta):
            return {"libros": [], "revistas": [], "usuarios": [], "bibliotecarios": []}
        
        try:
            with open(ruta, 'r') as archivo:
                return json.load(archivo)
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return {"libros": [], "revistas": [], "usuarios": [], "bibliotecarios": []}

# ======================================================
# IMPLEMENTACIÓN DE GESTORES
# ======================================================

class GestorItems:
    """Gestor de items de la biblioteca (libros y revistas)"""
    
    def __init__(self):
        self._items = {}  # Diccionario de items por ID
    
    def agregar_item(self, item: Item) -> bool:
        """Agrega un item a la colección"""
        if item.id in self._items:
            return False
        
        self._items[item.id] = item
        return True
    
    def obtener_item(self, id_item: str) -> Optional[Item]:
        """Obtiene un item por su ID"""
        return self._items.get(id_item)
    
    def eliminar_item(self, id_item: str) -> bool:
        """Elimina un item de la colección"""
        if id_item not in self._items:
            return False
        
        del self._items[id_item]
        return True
    
    def listar_items(self) -> List[Item]:
        """Devuelve la lista de todos los items"""
        return list(self._items.values())
    
    def buscar_items(self, termino: str) -> List[Item]:
        """Busca items que coincidan con el término de búsqueda"""
        termino = termino.lower()
        resultados = []
        
        for item in self._items.values():
            if isinstance(item, Libro):
                if (termino in item.titulo.lower() or 
                    termino in item.autor.lower() or 
                    termino in item.genero.lower()):
                    resultados.append(item)
            elif isinstance(item, Revista):
                if (termino in item.titulo.lower() or 
                    termino in item.editorial.lower()):
                    resultados.append(item)
        
        return resultados
    
    def items_disponibles(self) -> List[Item]:
        """Devuelve la lista de items disponibles para préstamo"""
        return [item for item in self._items.values() if item.esta_disponible()]
    
    def items_prestados(self) -> List[Item]:
        """Devuelve la lista de items prestados"""
        return [item for item in self._items.values() if not item.esta_disponible()]
    
    def cargar_desde_dict(self, datos: Dict) -> None:
        """Carga los items desde un diccionario"""
        self._items = {}
        
        for datos_libro in datos.get("libros", []):
            libro = Libro.desde_dict(datos_libro)
            self._items[libro.id] = libro
        
        for datos_revista in datos.get("revistas", []):
            revista = Revista.desde_dict(datos_revista)
            self._items[revista.id] = revista
    
    def a_dict(self) -> Dict[str, List]:
        """Convierte los items a un diccionario para serialización"""
        libros = []
        revistas = []
        
        for item in self._items.values():
            if isinstance(item, Libro):
                libros.append(item.a_dict())
            elif isinstance(item, Revista):
                revistas.append(item.a_dict())
        
        return {
            "libros": libros,
            "revistas": revistas
        }

class GestorPersonas:
    """Gestor de personas (usuarios y bibliotecarios)"""
    
    def __init__(self):
        self._usuarios = {}  # Diccionario de usuarios por ID
        self._bibliotecarios = {}  # Diccionario de bibliotecarios por ID
    
    def agregar_usuario(self, usuario: Usuario) -> bool:
        """Agrega un usuario a la colección"""
        if usuario.id in self._usuarios:
            return False
        
        self._usuarios[usuario.id] = usuario
        return True
    
    def obtener_usuario(self, id_usuario: str) -> Optional[Usuario]:
        """Obtiene un usuario por su ID"""
        return self._usuarios.get(id_usuario)
    
    def eliminar_usuario(self, id_usuario: str) -> bool:
        """Elimina un usuario de la colección"""
        if id_usuario not in self._usuarios:
            return False
        
        del self._usuarios[id_usuario]
        return True
    
    def listar_usuarios(self) -> List[Usuario]:
        """Devuelve la lista de todos los usuarios"""
        return list(self._usuarios.values())
    
    def buscar_usuarios(self, termino: str) -> List[Usuario]:
        """Busca usuarios que coincidan con el término de búsqueda"""
        termino = termino.lower()
        return [u for u in self._usuarios.values() 
                if termino in u.nombre.lower() or termino in u.email.lower()]
    
    def agregar_bibliotecario(self, bibliotecario: Bibliotecario) -> bool:
        """Agrega un bibliotecario a la colección"""
        if bibliotecario.id in self._bibliotecarios:
            return False
        
        self._bibliotecarios[bibliotecario.id] = bibliotecario
        return True
    
    def obtener_bibliotecario(self, id_bibliotecario: str) -> Optional[Bibliotecario]:
        """Obtiene un bibliotecario por su ID"""
        return self._bibliotecarios.get(id_bibliotecario)
    
    def autenticar_bibliotecario(self, email: str, password: str) -> Optional[Bibliotecario]:
        """Autentica un bibliotecario por email y contraseña"""
        for bibliotecario in self._bibliotecarios.values():
            if bibliotecario.email == email and bibliotecario.verificar_password(password):
                return bibliotecario
        return None
    
    def cargar_desde_dict(self, datos: Dict) -> None:
        """Carga los usuarios y bibliotecarios desde un diccionario"""
        self._usuarios = {}
        self._bibliotecarios = {}
        
        for datos_usuario in datos.get("usuarios", []):
            usuario = Usuario.desde_dict(datos_usuario)
            self._usuarios[usuario.id] = usuario
        
        for datos_bibliotecario in datos.get("bibliotecarios", []):
            bibliotecario = Bibliotecario.desde_dict(datos_bibliotecario)
            self._bibliotecarios[bibliotecario.id] = bibliotecario
    
    def a_dict(self) -> Dict[str, List]:
        """Convierte los usuarios y bibliotecarios a un diccionario para serialización"""
        return {
            "usuarios": [u.a_dict() for u in self._usuarios.values()],
            "bibliotecarios": [b.a_dict() for b in self._bibliotecarios.values()]
        }

class GestorPrestamos:
    """Gestor de préstamos de items"""
    
    def __init__(self, gestor_items: GestorItems, gestor_personas: GestorPersonas):
        self.gestor_items = gestor_items
        self.gestor_personas = gestor_personas
        self._dias_prestamo = 14  # Días por defecto para préstamos
    
    def prestar_item(self, id_item: str, id_usuario: str) -> bool:
        """Registra el préstamo de un item a un usuario"""
        # Obtener el item y el usuario
        item = self.gestor_items.obtener_item(id_item)
        usuario = self.gestor_personas.obtener_usuario(id_usuario)
        
        # Verificar que existan y que se puedan prestar
        if not item or not usuario:
            return False
        
        if not item.esta_disponible():
            return False
        
        if not usuario.puede_pedir_prestado():
            return False
        
        # Registrar el préstamo
        fecha_actual = datetime.datetime.now()
        
        if item.prestar(id_usuario, fecha_actual) and usuario.registrar_prestamo(id_item, fecha_actual):
            return True
        
        # Si algo falló, revertir cambios
        item.devolver(fecha_actual)
        usuario.registrar_devolucion(id_item, fecha_actual)
        return False
    
    def devolver_item(self, id_item: str) -> bool:
        """Registra la devolución de un item"""
        # Obtener el item
        item = self.gestor_items.obtener_item(id_item)
        
        if not item or item.esta_disponible():
            return False
        
        # Obtener información del préstamo actual
        prestamo_actual = item.obtener_prestamo_actual()
        if not prestamo_actual:
            return False
        
        # Obtener el usuario
        id_usuario = prestamo_actual["id_usuario"]
        usuario = self.gestor_personas.obtener_usuario(id_usuario)
        
        if not usuario:
            return False
        
        # Registrar la devolución
        fecha_actual = datetime.datetime.now()
        
        if item.devolver(fecha_actual) and usuario.registrar_devolucion(id_item, fecha_actual):
            return True
        
        # Si algo falló, revertir cambios
        item.prestar(id_usuario, fecha_actual)
        usuario.registrar_prestamo(id_item, fecha_actual)
        return False
    
    def listar_prestamos_actuales(self) -> List[Dict]:
        """Lista todos los préstamos actuales"""
        prestamos = []
        for item in self.gestor_items.items_prestados():
            prestamo = item.obtener_prestamo_actual()
            if prestamo:
                usuario = self.gestor_personas.obtener_usuario(prestamo["id_usuario"])
                prestamos.append({
                    "id_item": item.id,
                    "titulo_item": getattr(item, "titulo", "Sin título"),
                    "id_usuario": prestamo["id_usuario"],
                    "nombre_usuario": usuario.nombre if usuario else "Usuario desconocido",
                    "fecha_prestamo": prestamo["fecha_prestamo"]
                })
        return prestamos
    
    def listar_prestamos_vencidos(self) -> List[Dict]:
        """Lista los préstamos vencidos"""
        prestamos_vencidos = []
        fecha_actual = datetime.datetime.now()
        
        for item in self.gestor_items.items_prestados():
            prestamo = item.obtener_prestamo_actual()
            if prestamo:
                fecha_prestamo = datetime.datetime.strptime(
                    prestamo["fecha_prestamo"], "%Y-%m-%d %H:%M:%S")
                dias_prestado = (fecha_actual - fecha_prestamo).days
                
                if dias_prestado > self._dias_prestamo:
                    usuario = self.gestor_personas.obtener_usuario(prestamo["id_usuario"])
                    prestamos_vencidos.append({
                        "id_item": item.id,
                        "titulo_item": getattr(item, "titulo", "Sin título"),
                        "id_usuario": prestamo["id_usuario"],
                        "nombre_usuario": usuario.nombre if usuario else "Usuario desconocido",
                        "fecha_prestamo": prestamo["fecha_prestamo"],
                        "dias_vencido": dias_prestado - self._dias_prestamo
                    })
        
        return prestamos_vencidos
    
    def listar_prestamos_usuario(self, id_usuario: str) -> List[Dict]:
        """Lista los préstamos actuales de un usuario específico"""
        usuario = self.gestor_personas.obtener_usuario(id_usuario)
        if not usuario:
            return []
        
        prestamos = []
        for id_item in usuario.obtener_prestamos_actuales():
            item = self.gestor_items.obtener_item(id_item)
            if item:
                prestamo = item.obtener_prestamo_actual()
                if prestamo:
                    prestamos.append({
                        "id_item": item.id,
                        "titulo_item": getattr(item, "titulo", "Sin título"),
                        "fecha_prestamo": prestamo["fecha_prestamo"]
                    })
        
        return prestamos
    
    def establecer_dias_prestamo(self, dias: int) -> None:
        """Establece la cantidad de días para préstamos"""
        if dias > 0:
            self._dias_prestamo = dias

# ======================================================
# SISTEMA DE BIBLIOTECA
# ======================================================

class SistemaBiblioteca:
    """Clase principal que coordina todas las operaciones del sistema"""
    
    def __init__(self, ruta_datos: str = "biblioteca_datos.json"):
        # Inicializar componentes
        self.gestor_items = GestorItems()
        self.gestor_personas = GestorPersonas()
        self.gestor_prestamos = GestorPrestamos(self.gestor_items, self.gestor_personas)
        self.persistencia = PersistenciaJSON()
        self.ruta_datos = ruta_datos
        
        # Crear bibliotecario por defecto si no hay ninguno
        self._bibliotecario_actual = None
        self._usuario_actual = None
        
        # Cargar datos iniciales
        self.cargar_datos()
        
        # Si no hay bibliotecarios, crear uno por defecto
        if not self.gestor_personas.listar_usuarios() and not [b for b in self.gestor_personas._bibliotecarios.values()]:
            self._crear_datos_iniciales()
    
    def _crear_datos_iniciales(self):
        """Crea datos iniciales para el sistema"""
        # Crear un bibliotecario por defecto
        bibliotecario = Bibliotecario("BIB001", "Admin", "admin@biblioteca.com", "admin123")
        self.gestor_personas.agregar_bibliotecario(bibliotecario)
        
        # Crear algunos usuarios
        usuario1 = Usuario("USR001", "Juan Pérez", "juan@example.com", "555-1234")
        usuario2 = Usuario("USR002", "María García", "maria@example.com", "555-5678")
        self.gestor_personas.agregar_usuario(usuario1)
        self.gestor_personas.agregar_usuario(usuario2)
        
        # Crear algunos libros
        libro1 = Libro("LIB001", "Don Quijote de la Mancha", "Miguel de Cervantes", 1605, "Clásico", "9788466333849")
        libro2 = Libro("LIB002", "1984", "George Orwell", 1949, "Ciencia Ficción", "9788499890944")
        libro3 = Libro("LIB003", "Cien años de soledad", "Gabriel García Márquez", 1967, "Realismo mágico", "9788497592208")
        self.gestor_items.agregar_item(libro1)
        self.gestor_items.agregar_item(libro2)
        self.gestor_items.agregar_item(libro3)
        
        # Crear algunas revistas
        revista1 = Revista("REV001", "National Geographic", "National Geographic Society", 301, "Enero 2023")
        revista2 = Revista("REV002", "Scientific American", "Springer Nature", 152, "Febrero 2023")
        self.gestor_items.agregar_item(revista1)
        self.gestor_items.agregar_item(revista2)
        
        # Guardar los datos
        self.guardar_datos()
    
    def cargar_datos(self) -> bool:
        """Carga los datos del sistema desde el archivo de persistencia"""
        datos = self.persistencia.cargar_datos(self.ruta_datos)
        self.gestor_items.cargar_desde_dict(datos)
        self.gestor_personas.cargar_desde_dict(datos)
        return True
    
    def guardar_datos(self) -> bool:
        """Guarda los datos del sistema en el archivo de persistencia"""
        datos_items = self.gestor_items.a_dict()
        datos_personas = self.gestor_personas.a_dict()
        
        datos_completos = {**datos_items, **datos_personas}
        return self.persistencia.guardar_datos(datos_completos, self.ruta_datos)
    
    def login_bibliotecario(self, email: str, password: str) -> Optional[Bibliotecario]:
        """Inicia sesión como bibliotecario"""
        bibliotecario = self.gestor_personas.autenticar_bibliotecario(email, password)
        if bibliotecario:
            self._bibliotecario_actual = bibliotecario
            self._usuario_actual = None
        return bibliotecario
    
    def login_usuario(self, id_usuario: str) -> Optional[Usuario]:
        """Inicia sesión como usuario"""
        usuario = self.gestor_personas.obtener_usuario(id_usuario)
        if usuario:
            self._usuario_actual = usuario
            self._bibliotecario_actual = None
        return usuario
    
    def logout(self) -> None:
        """Cierra la sesión actual"""
        self._bibliotecario_actual = None
        self._usuario_actual = None
    
    def obtener_usuario_actual(self) -> Optional[Union[Bibliotecario, Usuario]]:
        """Devuelve el usuario o bibliotecario que ha iniciado sesión"""
        return self._bibliotecario_actual or self._usuario_actual
    
    def es_bibliotecario(self) -> bool:
        """Verifica si hay un bibliotecario con sesión iniciada"""
        return self._bibliotecario_actual is not None
    
    # --------- Métodos para gestión de ítems ---------
    
    def agregar_libro(self, id_libro: str, titulo: str, autor: str, anio: int, 
                     genero: str = "", isbn: str = "") -> bool:
        """Agrega un nuevo libro al sistema"""
        if not self.es_bibliotecario():
            return False
        
        libro = Libro(id_libro, titulo, autor, anio, genero, isbn)
        resultado = self.gestor_items.agregar_item(libro)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def agregar_revista(self, id_revista: str, titulo: str, editorial: str, numero: int, 
                       fecha_publicacion: str) -> bool:
        """Agrega una nueva revista al sistema"""
        if not self.es_bibliotecario():
            return False
        
        revista = Revista(id_revista, titulo, editorial, numero, fecha_publicacion)
        resultado = self.gestor_items.agregar_item(revista)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def eliminar_item(self, id_item: str) -> bool:
        """Elimina un item del sistema"""
        if not self.es_bibliotecario():
            return False
        
        resultado = self.gestor_items.eliminar_item(id_item)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def buscar_items(self, termino: str) -> List[Item]:
        """Busca items que coincidan con el término de búsqueda"""
        return self.gestor_items.buscar_items(termino)
    
    def listar_items(self) -> List[Item]:
        """Devuelve la lista de todos los items"""
        return self.gestor_items.listar_items()
    
    def listar_items_disponibles(self) -> List[Item]:
        """Devuelve la lista de items disponibles para préstamo"""
        return self.gestor_items.items_disponibles()
    
    def listar_items_prestados(self) -> List[Item]:
        """Devuelve la lista de items prestados"""
        return self.gestor_items.items_prestados()
    
    # --------- Métodos para gestión de usuarios ---------
    
    def agregar_usuario(self, id_usuario: str, nombre: str, email: str, telefono: str = "") -> bool:
        """Agrega un nuevo usuario al sistema"""
        if not self.es_bibliotecario():
            return False
        
        usuario = Usuario(id_usuario, nombre, email, telefono)
        resultado = self.gestor_personas.agregar_usuario(usuario)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def agregar_bibliotecario(self, id_bibliotecario: str, nombre: str, email: str, 
                             password: str) -> bool:
        """Agrega un nuevo bibliotecario al sistema"""
        if not self.es_bibliotecario():
            return False
        
        bibliotecario = Bibliotecario(id_bibliotecario, nombre, email, password)
        resultado = self.gestor_personas.agregar_bibliotecario(bibliotecario)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def eliminar_usuario(self, id_usuario: str) -> bool:
        """Elimina un usuario del sistema"""
        if not self.es_bibliotecario():
            return False
        
        resultado = self.gestor_personas.eliminar_usuario(id_usuario)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def buscar_usuarios(self, termino: str) -> List[Usuario]:
        """Busca usuarios que coincidan con el término de búsqueda"""
        if not self.es_bibliotecario():
            return []
        
        return self.gestor_personas.buscar_usuarios(termino)
    
    def listar_usuarios(self) -> List[Usuario]:
        """Devuelve la lista de todos los usuarios"""
        if not self.es_bibliotecario():
            return []
        
        return self.gestor_personas.listar_usuarios()
    
    # --------- Métodos para gestión de préstamos ---------
    
    def prestar_item(self, id_item: str, id_usuario: str) -> bool:
        """Registra el préstamo de un item a un usuario"""
        if not self.es_bibliotecario():
            # Si es un usuario normal, solo puede pedir prestado para sí mismo
            if self._usuario_actual and self._usuario_actual.id != id_usuario:
                return False
        
        resultado = self.gestor_prestamos.prestar_item(id_item, id_usuario)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def devolver_item(self, id_item: str) -> bool:
        """Registra la devolución de un item"""
        if not self.es_bibliotecario():
            # Si es un usuario normal, verificar que el item esté prestado a él
            if self._usuario_actual:
                item = self.gestor_items.obtener_item(id_item)
                if not item:
                    return False
                
                prestamo = item.obtener_prestamo_actual()
                if not prestamo or prestamo["id_usuario"] != self._usuario_actual.id:
                    return False
        
        resultado = self.gestor_prestamos.devolver_item(id_item)
        if resultado:
            self.guardar_datos()
        return resultado
    
    def listar_prestamos_actuales(self) -> List[Dict]:
        """Lista todos los préstamos actuales"""
        if not self.es_bibliotecario() and self._usuario_actual:
            # Si es un usuario normal, solo ve sus préstamos
            return self.listar_prestamos_usuario(self._usuario_actual.id)
        
        return self.gestor_prestamos.listar_prestamos_actuales()
    
    def listar_prestamos_vencidos(self) -> List[Dict]:
        """Lista los préstamos vencidos"""
        if not self.es_bibliotecario() and self._usuario_actual:
            # Filtrar solo los préstamos vencidos del usuario actual
            todos_vencidos = self.gestor_prestamos.listar_prestamos_vencidos()
            return [p for p in todos_vencidos if p["id_usuario"] == self._usuario_actual.id]
        
        return self.gestor_prestamos.listar_prestamos_vencidos()
    
    def listar_prestamos_usuario(self, id_usuario: str) -> List[Dict]:
        """Lista los préstamos actuales de un usuario específico"""
        if not self.es_bibliotecario() and (not self._usuario_actual or self._usuario_actual.id != id_usuario):
            return []
        
        return self.gestor_prestamos.listar_prestamos_usuario(id_usuario)

# ======================================================
# INTERFAZ DE USUARIO (BÁSICA, MODO CONSOLA)
# ======================================================

class InterfazConsola:
    """Interfaz de usuario básica en modo consola"""
    
    def __init__(self, sistema: SistemaBiblioteca):
        self.sistema = sistema
    
    def mostrar_menu_principal(self):
        print("\n=== SISTEMA DE BIBLIOTECA ===")
        print("1. Iniciar sesión como bibliotecario")
        print("2. Iniciar sesión como usuario")
        print("3. Salir")
    
    def mostrar_menu_bibliotecario(self):
        print("\n=== MENÚ BIBLIOTECARIO ===")
        print("1. Gestión de libros")
        print("2. Gestión de revistas")
        print("3. Gestión de usuarios")
        print("4. Gestión de préstamos")
        print("5. Cerrar sesión")
    
    def mostrar_menu_usuario(self):
        print("\n=== MENÚ USUARIO ===")
        print("1. Buscar items")
        print("2. Ver mis préstamos")
        print("3. Solicitar préstamo")
        print("4. Devolver item")
        print("5. Cerrar sesión")
    
    def mostrar_menu_gestion_libros(self):
        print("\n=== GESTIÓN DE LIBROS ===")
        print("1. Agregar libro")
        print("2. Buscar libros")
        print("3. Listar todos los libros")
        print("4. Eliminar libro")
        print("5. Volver")
    
    def mostrar_menu_gestion_revistas(self):
        print("\n=== GESTIÓN DE REVISTAS ===")
        print("1. Agregar revista")
        print("2. Buscar revistas")
        print("3. Listar todas las revistas")
        print("4. Eliminar revista")
        print("5. Volver")
    
    def mostrar_menu_gestion_usuarios(self):
        print("\n=== GESTIÓN DE USUARIOS ===")
        print("1. Agregar usuario")
        print("2. Buscar usuarios")
        print("3. Listar todos los usuarios")
        print("4. Eliminar usuario")
        print("5. Agregar bibliotecario")
        print("6. Volver")
    
    def mostrar_menu_gestion_prestamos(self):
        print("\n=== GESTIÓN DE PRÉSTAMOS ===")
        print("1. Registrar préstamo")
        print("2. Registrar devolución")
        print("3. Listar préstamos actuales")
        print("4. Listar préstamos vencidos")
        print("5. Listar préstamos por usuario")
        print("6. Volver")
    
    def ejecutar(self):
        """Ejecuta la interfaz de usuario"""
        while True:
            self.mostrar_menu_principal()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.iniciar_sesion_bibliotecario()
            elif opcion == "2":
                self.iniciar_sesion_usuario()
            elif opcion == "3":
                print("Saliendo del sistema...")
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def iniciar_sesion_bibliotecario(self):
        """Maneja el inicio de sesión como bibliotecario"""
        print("\n=== INICIAR SESIÓN COMO BIBLIOTECARIO ===")
        email = input("Email: ")
        password = input("Contraseña: ")
        
        bibliotecario = self.sistema.login_bibliotecario(email, password)
        
        if bibliotecario:
            print(f"Bienvenido/a, {bibliotecario.nombre}!")
            self.menu_bibliotecario()
        else:
            print("Email o contraseña incorrectos.")
    
    def iniciar_sesion_usuario(self):
        """Maneja el inicio de sesión como usuario"""
        print("\n=== INICIAR SESIÓN COMO USUARIO ===")
        id_usuario = input("ID de usuario: ")
        
        usuario = self.sistema.login_usuario(id_usuario)
        
        if usuario:
            print(f"Bienvenido/a, {usuario.nombre}!")
            self.menu_usuario()
        else:
            print("Usuario no encontrado.")
    
    def menu_bibliotecario(self):
        """Maneja el menú de bibliotecario"""
        while True:
            self.mostrar_menu_bibliotecario()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.menu_gestion_libros()
            elif opcion == "2":
                self.menu_gestion_revistas()
            elif opcion == "3":
                self.menu_gestion_usuarios()
            elif opcion == "4":
                self.menu_gestion_prestamos()
            elif opcion == "5":
                self.sistema.logout()
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def menu_usuario(self):
        """Maneja el menú de usuario"""
        while True:
            self.mostrar_menu_usuario()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.buscar_items_usuario()
            elif opcion == "2":
                self.ver_mis_prestamos()
            elif opcion == "3":
                self.solicitar_prestamo()
            elif opcion == "4":
                self.devolver_item_usuario()
            elif opcion == "5":
                self.sistema.logout()
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def menu_gestion_libros(self):
        """Maneja el menú de gestión de libros"""
        while True:
            self.mostrar_menu_gestion_libros()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.agregar_libro()
            elif opcion == "2":
                self.buscar_libros()
            elif opcion == "3":
                self.listar_libros()
            elif opcion == "4":
                self.eliminar_libro()
            elif opcion == "5":
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def menu_gestion_revistas(self):
        """Maneja el menú de gestión de revistas"""
        while True:
            self.mostrar_menu_gestion_revistas()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.agregar_revista()
            elif opcion == "2":
                self.buscar_revistas()
            elif opcion == "3":
                self.listar_revistas()
            elif opcion == "4":
                self.eliminar_revista()
            elif opcion == "5":
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def menu_gestion_usuarios(self):
        """Maneja el menú de gestión de usuarios"""
        while True:
            self.mostrar_menu_gestion_usuarios()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.agregar_usuario()
            elif opcion == "2":
                self.buscar_usuarios()
            elif opcion == "3":
                self.listar_usuarios()
            elif opcion == "4":
                self.eliminar_usuario()
            elif opcion == "5":
                self.agregar_bibliotecario()
            elif opcion == "6":
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    def menu_gestion_prestamos(self):
        """Maneja el menú de gestión de préstamos"""
        while True:
            self.mostrar_menu_gestion_prestamos()
            opcion = input("Seleccione una opción: ")
            
            if opcion == "1":
                self.registrar_prestamo()
            elif opcion == "2":
                self.registrar_devolucion()
            elif opcion == "3":
                self.listar_prestamos_actuales()
            elif opcion == "4":
                self.listar_prestamos_vencidos()
            elif opcion == "5":
                self.listar_prestamos_usuario()
            elif opcion == "6":
                break
            else:
                print("Opción no válida. Intente de nuevo.")
    
    # --------- Métodos para gestión de libros ---------
    
    def agregar_libro(self):
        """Interfaz para agregar un libro"""
        print("\n=== AGREGAR LIBRO ===")
        id_libro = input("ID del libro: ")
        titulo = input("Título: ")
        autor = input("Autor: ")
        
        try:
            anio = int(input("Año de publicación: "))
        except ValueError:
            print("Año no válido. Se asignará el año actual.")
            anio = datetime.datetime.now().year
        
        genero = input("Género (opcional): ")
        isbn = input("ISBN (opcional): ")
        
        resultado = self.sistema.agregar_libro(id_libro, titulo, autor, anio, genero, isbn)
        
        if resultado:
            print("Libro agregado con éxito.")
        else:
            print("Error al agregar el libro. Es posible que el ID ya exista.")
    
    def buscar_libros(self):
        """Interfaz para buscar libros"""
        print("\n=== BUSCAR LIBROS ===")
        termino = input("Término de búsqueda: ")
        
        items = self.sistema.buscar_items(termino)
        libros = [item for item in items if isinstance(item, Libro)]
        
        if libros:
            print("\nResultados de la búsqueda:")
            for libro in libros:
                print(libro.obtener_info())
        else:
            print("No se encontraron libros que coincidan con la búsqueda.")
    
    def listar_libros(self):
        """Interfaz para listar todos los libros"""
        print("\n=== LISTADO DE LIBROS ===")
        
        items = self.sistema.listar_items()
        libros = [item for item in items if isinstance(item, Libro)]
        
        if libros:
            for libro in libros:
                print(libro.obtener_info())
        else:
            print("No hay libros registrados en el sistema.")
    
    def eliminar_libro(self):
        """Interfaz para eliminar un libro"""
        print("\n=== ELIMINAR LIBRO ===")
        id_libro = input("ID del libro a eliminar: ")
        
        # Verificar que sea un libro
        item = self.sistema.gestor_items.obtener_item(id_libro)
        if not item or not isinstance(item, Libro):
            print("No se encontró un libro con ese ID.")
            return
        
        # Confirmar eliminación
        confirmacion = input(f"¿Seguro que desea eliminar '{item.titulo}'? (s/n): ")
        
        if confirmacion.lower() == 's':
            resultado = self.sistema.eliminar_item(id_libro)
            if resultado:
                print("Libro eliminado con éxito.")
            else:
                print("Error al eliminar el libro.")
        else:
            print("Operación cancelada.")
    
    # --------- Métodos para gestión de revistas ---------
    
    def agregar_revista(self):
        """Interfaz para agregar una revista"""
        print("\n=== AGREGAR REVISTA ===")
        id_revista = input("ID de la revista: ")
        titulo = input("Título: ")
        editorial = input("Editorial: ")
        
        try:
            numero = int(input("Número: "))
        except ValueError:
            print("Número no válido. Se asignará el número 1.")
            numero = 1
        
        fecha_publicacion = input("Fecha de publicación (ej: Enero 2023): ")
        
        resultado = self.sistema.agregar_revista(id_revista, titulo, editorial, numero, fecha_publicacion)
        
        if resultado:
            print("Revista agregada con éxito.")
        else:
            print("Error al agregar la revista. Es posible que el ID ya exista.")
    
    def buscar_revistas(self):
        """Interfaz para buscar revistas"""
        print("\n=== BUSCAR REVISTAS ===")
        termino = input("Término de búsqueda: ")
        
        items = self.sistema.buscar_items(termino)
        revistas = [item for item in items if isinstance(item, Revista)]
        
        if revistas:
            print("\nResultados de la búsqueda:")
            for revista in revistas:
                print(revista.obtener_info())
        else:
            print("No se encontraron revistas que coincidan con la búsqueda.")
    
    def listar_revistas(self):
        """Interfaz para listar todas las revistas"""
        print("\n=== LISTADO DE REVISTAS ===")
        
        items = self.sistema.listar_items()
        revistas = [item for item in items if isinstance(item, Revista)]
        
        if revistas:
            for revista in revistas:
                print(revista.obtener_info())
        else:
            print("No hay revistas registradas en el sistema.")
    
    def eliminar_revista(self):
        """Interfaz para eliminar una revista"""
        print("\n=== ELIMINAR REVISTA ===")
        id_revista = input("ID de la revista a eliminar: ")
        
        # Verificar que sea una revista
        item = self.sistema.gestor_items.obtener_item(id_revista)
        if not item or not isinstance(item, Revista):
            print("No se encontró una revista con ese ID.")
            return
        
        # Confirmar eliminación
        confirmacion = input(f"¿Seguro que desea eliminar '{item.titulo}'? (s/n): ")
        
        if confirmacion.lower() == 's':
            resultado = self.sistema.eliminar_item(id_revista)
            if resultado:
                print("Revista eliminada con éxito.")
            else:
                print("Error al eliminar la revista.")
        else:
            print("Operación cancelada.")
    
    # --------- Métodos para gestión de usuarios ---------
    
    def agregar_usuario(self):
        """Interfaz para agregar un usuario"""
        print("\n=== AGREGAR USUARIO ===")
        id_usuario = input("ID del usuario: ")
        nombre = input("Nombre: ")
        email = input("Email: ")
        telefono = input("Teléfono (opcional): ")
        
        resultado = self.sistema.agregar_usuario(id_usuario, nombre, email, telefono)
        
        if resultado:
            print("Usuario agregado con éxito.")
        else:
            print("Error al agregar el usuario. Es posible que el ID ya exista.")
    
    def buscar_usuarios(self):
        """Interfaz para buscar usuarios"""
        print("\n=== BUSCAR USUARIOS ===")
        termino = input("Término de búsqueda: ")
        
        usuarios = self.sistema.buscar_usuarios(termino)
        
        if usuarios:
            print("\nResultados de la búsqueda:")
            for usuario in usuarios:
                print(usuario.obtener_info())
        else:
            print("No se encontraron usuarios que coincidan con la búsqueda.")
    
    def listar_usuarios(self):
        """Interfaz para listar todos los usuarios"""
        print("\n=== LISTADO DE USUARIOS ===")
        
        usuarios = self.sistema.listar_usuarios()
        
        if usuarios:
            for usuario in usuarios:
                print(usuario.obtener_info())
        else:
            print("No hay usuarios registrados en el sistema.")
    
    def eliminar_usuario(self):
        """Interfaz para eliminar un usuario"""
        print("\n=== ELIMINAR USUARIO ===")
        id_usuario = input("ID del usuario a eliminar: ")
        
        # Verificar que el usuario exista
        usuario = self.sistema.gestor_personas.obtener_usuario(id_usuario)
        if not usuario:
            print("No se encontró un usuario con ese ID.")
            return
        
        # Confirmar eliminación
        confirmacion = input(f"¿Seguro que desea eliminar al usuario '{usuario.nombre}'? (s/n): ")
        
        if confirmacion.lower() == 's':
            resultado = self.sistema.eliminar_usuario(id_usuario)
            if resultado:
                print("Usuario eliminado con éxito.")
            else:
                print("Error al eliminar el usuario.")
        else:
            print("Operación cancelada.")
    
    def agregar_bibliotecario(self):
        """Interfaz para agregar un bibliotecario"""
        print("\n=== AGREGAR BIBLIOTECARIO ===")
        id_bibliotecario = input("ID del bibliotecario: ")
        nombre = input("Nombre: ")
        email = input("Email: ")
        password = input("Contraseña: ")
        
        resultado = self.sistema.agregar_bibliotecario(id_bibliotecario, nombre, email, password)
        
        if resultado:
            print("Bibliotecario agregado con éxito.")
        else:
            print("Error al agregar el bibliotecario. Es posible que el ID ya exista.")
    
    # --------- Métodos para gestión de préstamos ---------
    
    def registrar_prestamo(self):
        """Interfaz para registrar un préstamo"""
        print("\n=== REGISTRAR PRÉSTAMO ===")
        id_item = input("ID del ítem a prestar: ")
        id_usuario = input("ID del usuario: ")
        
        # Verificar que el item y el usuario existan
        item = self.sistema.gestor_items.obtener_item(id_item)
        usuario = self.sistema.gestor_personas.obtener_usuario(id_usuario)
        
        if not item:
            print("No se encontró un ítem con ese ID.")
            return
        
        if not usuario:
            print("No se encontró un usuario con ese ID.")
            return
        
        if not item.esta_disponible():
            print(f"El ítem '{getattr(item, 'titulo', '')}' no está disponible para préstamo.")
            return
        
        if not usuario.puede_pedir_prestado():
            print(f"El usuario '{usuario.nombre}' ha alcanzado el límite de préstamos.")
            return
        
        resultado = self.sistema.prestar_item(id_item, id_usuario)
        
        if resultado:
            print("Préstamo registrado con éxito.")
        else:
            print("Error al registrar el préstamo.")
    
    def registrar_devolucion(self):
        """Interfaz para registrar una devolución"""
        print("\n=== REGISTRAR DEVOLUCIÓN ===")
        id_item = input("ID del ítem a devolver: ")
        
        # Verificar que el item exista
        item = self.sistema.gestor_items.obtener_item(id_item)
        
        if not item:
            print("No se encontró un ítem con ese ID.")
            return
        
        if item.esta_disponible():
            print(f"El ítem '{getattr(item, 'titulo', '')}' no está prestado.")
            return
        
        resultado = self.sistema.devolver_item(id_item)
        
        if resultado:
            print("Devolución registrada con éxito.")
        else:
            print("Error al registrar la devolución.")
    
    def listar_prestamos_actuales(self):
        """Interfaz para listar préstamos actuales"""
        print("\n=== PRÉSTAMOS ACTUALES ===")
        
        prestamos = self.sistema.listar_prestamos_actuales()
        
        if prestamos:
            for prestamo in prestamos:
                print(f"Item: {prestamo['titulo_item']} (ID: {prestamo['id_item']})")
                print(f"  Usuario: {prestamo['nombre_usuario']} (ID: {prestamo['id_usuario']})")
                print(f"  Fecha de préstamo: {prestamo['fecha_prestamo']}")
                print()
        else:
            print("No hay préstamos activos en el sistema.")
    
    def listar_prestamos_vencidos(self):
        """Interfaz para listar préstamos vencidos"""
        print("\n=== PRÉSTAMOS VENCIDOS ===")
        
        prestamos = self.sistema.listar_prestamos_vencidos()
        
        if prestamos:
            for prestamo in prestamos:
                print(f"Item: {prestamo['titulo_item']} (ID: {prestamo['id_item']})")
                print(f"  Usuario: {prestamo['nombre_usuario']} (ID: {prestamo['id_usuario']})")
                print(f"  Fecha de préstamo: {prestamo['fecha_prestamo']}")
                print(f"  Días vencido: {prestamo['dias_vencido']}")
                print()
        else:
            print("No hay préstamos vencidos en el sistema.")
    
    def listar_prestamos_usuario(self):
        """Interfaz para listar préstamos de un usuario específico"""
        print("\n=== PRÉSTAMOS POR USUARIO ===")
        id_usuario = input("ID del usuario: ")
        
        # Verificar que el usuario exista
        usuario = self.sistema.gestor_personas.obtener_usuario(id_usuario)
        if not usuario:
            print("No se encontró un usuario con ese ID.")
            return
        
        prestamos = self.sistema.listar_prestamos_usuario(id_usuario)
        
        if prestamos:
            print(f"Préstamos de {usuario.nombre}:")
            for prestamo in prestamos:
                print(f"  Item: {prestamo['titulo_item']} (ID: {prestamo['id_item']})")
                print(f"  Fecha de préstamo: {prestamo['fecha_prestamo']}")
                print()
        else:
            print(f"El usuario {usuario.nombre} no tiene préstamos activos.")
    
    # --------- Métodos para usuarios normales ---------
    
    def buscar_items_usuario(self):
        """Interfaz para que un usuario busque items"""
        print("\n=== BUSCAR ITEMS ===")
        termino = input("Término de búsqueda: ")
        
        items = self.sistema.buscar_items(termino)
        
        if items:
            print("\nResultados de la búsqueda:")
            for item in items:
                print(item.obtener_info())
        else:
            print("No se encontraron items que coincidan con la búsqueda.")
    
    def ver_mis_prestamos(self):
        """Interfaz para que un usuario vea sus préstamos"""
        usuario_actual = self.sistema.obtener_usuario_actual()
        if not usuario_actual or not isinstance(usuario_actual, Usuario):
            print("Error: No hay un usuario con sesión iniciada.")
            return
        
        prestamos = self.sistema.listar_prestamos_usuario(usuario_actual.id)
        
        if prestamos:
            print("\n=== MIS PRÉSTAMOS ===")
            for prestamo in prestamos:
                print(f"Item: {prestamo['titulo_item']} (ID: {prestamo['id_item']})")
                print(f"  Fecha de préstamo: {prestamo['fecha_prestamo']}")
                print()
        else:
            print("No tienes préstamos activos.")
    
    def solicitar_prestamo(self):
        """Interfaz para que un usuario solicite un préstamo"""
        usuario_actual = self.sistema.obtener_usuario_actual()
        if not usuario_actual or not isinstance(usuario_actual, Usuario):
            print("Error: No hay un usuario con sesión iniciada.")
            return
        
        if not usuario_actual.puede_pedir_prestado():
            print("Has alcanzado el límite de préstamos permitidos.")
            return
        
        print("\n=== SOLICITAR PRÉSTAMO ===")
        
        # Mostrar items disponibles
        items_disponibles = self.sistema.listar_items_disponibles()
        if not items_disponibles:
            print("No hay items disponibles para préstamo.")
            return
        
        print("Items disponibles:")
        for i, item in enumerate(items_disponibles, 1):
            print(f"{i}. {item.obtener_info()}")
        
        try:
            seleccion = int(input("\nSeleccione un ítem (número): ")) - 1
            if seleccion < 0 or seleccion >= len(items_disponibles):
                print("Selección no válida.")
                return
            
            item_seleccionado = items_disponibles[seleccion]
            resultado = self.sistema.prestar_item(item_seleccionado.id, usuario_actual.id)
            
            if resultado:
                print("Préstamo registrado con éxito.")
            else:
                print("Error al registrar el préstamo.")
        except ValueError:
            print("Entrada no válida. Debe ingresar un número.")
    
    def devolver_item_usuario(self):
        """Interfaz para que un usuario devuelva un item"""
        usuario_actual = self.sistema.obtener_usuario_actual()
        if not usuario_actual or not isinstance(usuario_actual, Usuario):
            print("Error: No hay un usuario con sesión iniciada.")
            return
        
        prestamos = self.sistema.listar_prestamos_usuario(usuario_actual.id)
        if not prestamos:
            print("No tienes items prestados para devolver.")
            return
        
        print("\n=== DEVOLVER ITEM ===")
        print("Items prestados:")
        for i, prestamo in enumerate(prestamos, 1):
            print(f"{i}. {prestamo['titulo_item']} (ID: {prestamo['id_item']})")
        
        try:
            seleccion = int(input("\nSeleccione un ítem para devolver (número): ")) - 1
            if seleccion < 0 or seleccion >= len(prestamos):
                print("Selección no válida.")
                return
            
            id_item = prestamos[seleccion]['id_item']
            resultado = self.sistema.devolver_item(id_item)
            
            if resultado:
                print("Devolución registrada con éxito.")
            else:
                print("Error al registrar la devolución.")
        except ValueError:
            print("Entrada no válida. Debe ingresar un número.")

# ======================================================
# FUNCIÓN PRINCIPAL
# ======================================================

def main():
    # Crear el sistema
    sistema = SistemaBiblioteca()
    
    # Crear la interfaz
    interfaz = InterfazConsola(sistema)
    
    # Ejecutar la interfaz
    interfaz.ejecutar()

if __name__ == "__main__":
    main()

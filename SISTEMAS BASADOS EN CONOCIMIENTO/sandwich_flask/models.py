import json
from datetime import datetime

class Ingrediente:
    def __init__(self, nombre, precio):
        self.nombre = nombre
        self.precio = precio

    def get_nombre(self):
        return self.nombre
    
    def get_precio(self):
        return self.precio
    
class Sandwich:
    def __init__(self):
        self.nombre = ''
        self.ingredientes = []
        self.total = 0.0

    def agregar_ingrediente(self, ingrediente):
        self.ingredientes.append(ingrediente)
        self.total += ingrediente.get_precio()

    def get_ingredientes(self):
        return self.ingredientes
    
    def get_total(self):
        return self.total
    
class Compra:
    def __init__(self, sandwich):
        self.id = self._get_ultimo_id() + 1
        self.timestamp = self._obtener_timestamp()
        self.sandwich = sandwich

    def get_id(self):
        return self.id
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_sandwich(self):
        return self.sandwich
    
    def guardar_compra(self):
        print("Guardando compra...")
        datos_compra = {
            "ID": self.id,
            "Fecha Compra": self.timestamp,
            "Ingredientes": [ingrediente.get_nombre() for ingrediente in self.sandwich.get_ingredientes()],
            "Total": self.sandwich.get_total()
        }

        print(f"Datos de la compra: {datos_compra}")

        try:
            with open("compras.json", "r", encoding="utf-8") as c:
                compras = json.load(c)
        except (FileNotFoundError, json.JSONDecodeError):
            compras = []

        compras.append(datos_compra)

        with open("compras.json", "w", encoding="utf-8") as c:
            json.dump(compras, c, indent=4)
        
        print("Compra guardada!")
    
    def _get_ultimo_id(self):
        try:
            with open("compras.json", "r", encoding="utf-8") as c:
                compras = json.load(c)
            return compras[-1]["ID"]
        
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        
    def _obtener_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
class SistemaCompras:
    def __init__(self):
        self.ingredientes = {
            "Jamón": Ingrediente("Jamón", 2),
            "Queso": Ingrediente("Queso", 1.50),
            "Pollo": Ingrediente("Pollo", 2),
            "Atún": Ingrediente("Atún", 1.50),
            "Lechuga": Ingrediente("Lechuga", 1),
            "Tomate": Ingrediente("Tomate", 1),
            "Cebolla": Ingrediente("Cebolla", 1),
            "Ketchup": Ingrediente("Ketchup", 0.50),
            "Mayonesa": Ingrediente("Mayonesa", 0.50),
            "Barbacoa": Ingrediente("Barbacoa", 0.50),
            "Mostaza": Ingrediente("Mostaza", 0.50)
        }
        self.compras = self._obtener_compras()

    def _obtener_compras(self):
        try:
            with open("compras.json", "r", encoding="utf-8") as c:
                datos_compras = json.load(c)

        except (FileNotFoundError, json.JSONDecodeError):
            return []

        compras = []
        for datos_compra in datos_compras:
            sandwich = Sandwich()
            for ingrediente_nombre in datos_compra["Ingredientes"]:
                if ingrediente_nombre in self.ingredientes:
                    sandwich.agregar_ingrediente(self.ingredientes[ingrediente_nombre])
                    
            compra = Compra(sandwich)
            compra.id = datos_compra["ID"]
            compra.timestamp = datos_compra["Fecha Compra"]
            compras.append(compra)

        return compras

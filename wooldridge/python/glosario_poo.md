# Glosario de Programación Orientada a Objetos en Python

## Términos Fundamentales

### A
- **Abstracción**: Proceso de identificar las características y comportamientos esenciales de un objeto y representarlos mientras se ocultan los detalles innecesarios.
- **Atributo**: Variable que pertenece a una clase o a una instancia (objeto) de una clase.
- **Atributo de clase**: Variable que se define dentro de una clase pero fuera de cualquier método, compartida por todas las instancias.
- **Atributo de instancia**: Variable que pertenece a una instancia específica, no compartida entre objetos.

### C
- **Clase**: Plantilla o "plano" para crear objetos, define atributos y métodos.
- **Clase abstracta**: Clase que no puede ser instanciada directamente, sirve como base para otras clases.
- **Clase derivada/hija**: Clase que hereda propiedades y métodos de otra clase.
- **Composición**: Forma de combinar objetos o clases en estructuras más complejas.
- **Constructor**: Método especial `__init__` que se llama automáticamente cuando se crea un objeto.

### D
- **Decorador**: Patrón de diseño y característica de Python que permite modificar el comportamiento de funciones o métodos.
- **Dependencia**: Relación entre clases donde una clase usa otra.
- **Destructor**: Método especial `__del__` que se ejecuta cuando un objeto se elimina.

### E
- **Encapsulamiento**: Principio que agrupa datos y métodos relacionados en una unidad (clase) y restringe el acceso directo a algunos componentes.

### H
- **Herencia**: Mecanismo por el cual una clase (hija) puede heredar atributos y métodos de otra clase (padre).
- **Herencia múltiple**: Característica que permite a una clase heredar de múltiples clases padre.

### I
- **Instancia**: Objeto concreto creado a partir de una clase.
- **Interfaz**: Conjunto de métodos que una clase debe implementar.

### M
- **Método**: Función definida dentro de una clase.
- **Método abstracto**: Método declarado pero sin implementación en una clase base.
- **Método de clase**: Método que recibe la clase como primer argumento en lugar de la instancia.
- **Método de instancia**: Método que opera sobre una instancia específica de una clase.
- **Método estático**: Método que no requiere acceso a la instancia ni a la clase.
- **Mixins**: Clases diseñadas para proporcionar métodos adicionales a otras clases.
- **MRO (Method Resolution Order)**: Orden en que Python busca métodos en clases con herencia.

### O
- **Objeto**: Instancia de una clase que encapsula datos y comportamientos.
- **Overriding (Sobrescritura)**: Redefinición de un método de la clase padre en una clase hija.

### P
- **Patrón de diseño**: Solución reutilizable para problemas comunes en diseño de software.
- **Polimorfismo**: Capacidad de objetos de diferentes clases para responder al mismo mensaje o método de manera distinta.
- **Property**: Característica de Python que permite definir métodos especiales de acceso a atributos.

### R
- **Refactorización**: Proceso de restructurar código existente sin cambiar su comportamiento.

### S
- **Self**: Primer parámetro de los métodos de instancia, referencia a la instancia actual.
- **Sobrecarga de operadores**: Redefinir el significado de operadores para clases personalizadas.
- **Subclase**: Clase que hereda de otra clase.
- **Super**: Función utilizada para llamar a métodos de la clase padre.

## Principios SOLID

- **S - Principio de Responsabilidad Única (SRP)**: Una clase debe tener una y solo una razón para cambiar.
- **O - Principio Abierto/Cerrado (OCP)**: Las entidades de software deben estar abiertas para la extensión pero cerradas para la modificación.
- **L - Principio de Sustitución de Liskov (LSP)**: Los objetos de una subclase deben poder sustituir a los objetos de la superclase sin afectar la funcionalidad.
- **I - Principio de Segregación de Interfaces (ISP)**: No se debe obligar a los clientes a depender de interfaces que no usan.
- **D - Principio de Inversión de Dependencias (DIP)**: Los módulos de alto nivel no deben depender de módulos de bajo nivel. Ambos deben depender de abstracciones.

## Patrones de Diseño Comunes

- **Singleton**: Garantiza que una clase tenga solo una instancia y proporciona un punto global de acceso a ella.
- **Factory Method**: Define una interfaz para crear un objeto, pero permite a las subclases alterar el tipo de objetos que se crearán.
- **Observer**: Define una dependencia uno-a-muchos entre objetos para que cuando un objeto cambie de estado, todos sus dependientes sean notificados.
- **Strategy**: Define una familia de algoritmos, encapsula cada uno, y los hace intercambiables.
- **Adapter**: Permite que interfaces incompatibles trabajen juntas.
- **Decorator**: Añade responsabilidades adicionales a un objeto dinámicamente.

## Términos Específicos de Python

- **Duck Typing**: "Si camina como un pato y habla como un pato, entonces debe ser un pato". Característica de Python donde el tipo o clase de un objeto es menos importante que los métodos que define.
- **Dunder methods (Métodos mágicos)**: Métodos especiales con doble guion bajo al inicio y al final (como `__init__`, `__str__`).
- **ABC (Abstract Base Class)**: Módulo de Python para crear clases abstractas.
- **@property**: Decorador que permite definir propiedades en las clases.
- **@classmethod**: Decorador para definir métodos de clase.
- **@staticmethod**: Decorador para definir métodos estáticos.
- **@abstractmethod**: Decorador para definir métodos abstractos.

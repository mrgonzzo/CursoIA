from flask import Flask, render_template, url_for, request, redirect
from models import Ingrediente, Sandwich, Compra, SistemaCompras
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/creacion')
def creacion():
    sistema = SistemaCompras()
    ingredientes = sistema.ingredientes
    return render_template('creacion.html', ingredientes=ingredientes)

@app.route('/ingredientes')
def ingredientes():
    sistema = SistemaCompras()
    ingredientes = sistema.ingredientes
    return render_template('ingredientes.html', ingredientes=ingredientes)

@app.route('/registro')
def registro():
    sistema = SistemaCompras()
    compras = sistema.compras
    return render_template('registro.html', compras=compras)

@app.route('/terminar_sandwich', methods=['POST'])
def terminar_sandwich():
    ingredientes_seleccionados = request.json['ingredientes']
    sistema = SistemaCompras()

    sandwich = Sandwich()
    for nombre_ingrediente in ingredientes_seleccionados:
        if nombre_ingrediente in sistema.ingredientes:
            sandwich.agregar_ingrediente(sistema.ingredientes[nombre_ingrediente])

    compra = Compra(sandwich)
    compra.guardar_compra()

    return "Compra guardada", 200

if __name__ == "__main__":
    app.run(debug=True)
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("best_model.keras")

# Crear la aplicación Flask
app = Flask(__name__)

def procesar_imagen(imagen):
    """Convierte la imagen a escala de grises, la redimensiona, invierte colores y normaliza."""
    imagen = Image.open(io.BytesIO(imagen)).convert("L")  # Convertir a escala de grises
    imagen = imagen.resize((28, 28))  # Redimensionar a 28x28 píxeles
    
    # Invertir colores (el modelo suele esperar dígitos blancos sobre fondo negro)
    imagen = Image.eval(imagen, lambda x: 255 - x)
    
    imagen = np.array(imagen) / 255.0  # Normalizar valores (0-1)
    imagen = imagen.reshape(1, 28, 28, 1)  # Ajustar forma para el modelo
    return imagen

@app.route("/")
def home():
    """Muestra la interfaz gráfica para dibujar dígitos."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predecir():
    """Recibe una imagen, la procesa y devuelve la predicción del modelo."""
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400
    
    archivo = request.files["file"].read()
    
    try:
        imagen_procesada = procesar_imagen(archivo)
    except Exception as e:
        return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 400

    # Hacer la predicción
    try:
        prediccion = modelo.predict(imagen_procesada)
        digito_predicho = int(np.argmax(prediccion))  # Obtener el número con mayor probabilidad
        confianza = float(np.max(prediccion))  # Obtener la confianza de la predicción
        
        return jsonify({
            "digito_predicho": digito_predicho,
            "confianza": confianza
        })
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

# Ejecutar la API en el puerto 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
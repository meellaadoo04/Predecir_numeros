import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import cv2

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("best_model.keras")

# Crear la aplicación Flask
app = Flask(__name__)

def segmentar_digitos(imagen_bytes):
    """
    Convierte la imagen recibida en bytes a escala de grises, la inverte (para que los dígitos sean blancos sobre fondo negro),
    aplica umbralización, encuentra los contornos y extrae cada dígito. Se agrega padding para obtener imágenes cuadradas
    y se redimensiona cada dígito a 28x28 píxeles, normalizando los valores.
    """
    # Convertir imagen a arreglo y decodificar con OpenCV
    np_arr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    # Invertir imagen para tener dígitos en blanco
    img_inv = 255 - img
    
    # Aplicar umbralización binaria
    _, thresh = cv2.threshold(img_inv, 128, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos externos (cada contorno representa un dígito)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Ignorar pequeños ruidos
        if w < 5 or h < 5:
            continue
        # Extraer la región del dígito
        digit = thresh[y:y+h, x:x+w]
        
        # Crear imagen cuadrada agregando padding
        new_size = max(w, h)
        padded = np.full((new_size, new_size), 0, dtype=np.uint8)
        x_offset = (new_size - w) // 2
        y_offset = (new_size - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # Redimensionar a 28x28 y normalizar
        resized = cv2.resize(padded, (28, 28))
        normalized = resized.astype("float32") / 255.0
        # Ajustar forma para el modelo (28,28,1)
        normalized = np.expand_dims(normalized, axis=-1)
        
        # Se guarda junto a la coordenada x para ordenar de izquierda a derecha
        digit_images.append((x, normalized))
    
    # Ordenar las imágenes según la posición horizontal
    digit_images = sorted(digit_images, key=lambda item: item[0])
    return [img for (_, img) in digit_images]

@app.route("/")
def home():
    """Muestra la interfaz gráfica para dibujar dígitos."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predecir():
    """
    Recibe una imagen, la procesa para segmentar los dígitos y para cada uno hace la predicción
    usando el modelo entrenado. Devuelve un listado de predicciones.
    """
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400
    
    archivo = request.files["file"].read()
    
    try:
        digitos = segmentar_digitos(archivo)
        if not digitos:
            return jsonify({"error": "No se detectaron dígitos"}), 400
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 400

    resultados = []
    for img in digitos:
        # Agregar dimensión de batch
        img_expanded = np.expand_dims(img, axis=0)
        prediccion = modelo.predict(img_expanded)
        digito_predicho = int(np.argmax(prediccion))
        confianza = float(np.max(prediccion))
        resultados.append({
            "digito_predicho": digito_predicho,
            "confianza": confianza
        })
    
    return jsonify({"resultados": resultados})

# Ejecutar la API en el puerto 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

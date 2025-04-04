import requests

# URL de la API Flask
URL = "http://127.0.0.1:5000/predict"

# Ruta de la imagen que quieres enviar
IMAGEN_PATH = "numero2.png"  # Cambia esto por la imagen que quieres probar

# Cargar la imagen y enviarla a la API
with open(IMAGEN_PATH, "rb") as img_file:
    archivos = {"file": img_file}
    respuesta = requests.post(URL, files=archivos)

# Mostrar la respuesta de la API
print("Respuesta de la API:", respuesta.json())

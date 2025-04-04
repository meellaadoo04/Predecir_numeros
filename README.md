# Reconocimiento de Dígitos con Redes Convolucionales y Flask

Este proyecto implementa una API REST utilizando Flask para el reconocimiento de dígitos escritos a mano. La API recibe imágenes de dígitos (o varios dígitos) y utiliza un modelo de red neuronal convolucional entrenado con TensorFlow/Keras para predecir los números.

## Características

- Procesamiento de imágenes en escala de grises.
- Predicción de dígitos individuales o múltiples (como códigos postales).
- API REST sencilla para interactuar con el modelo.
- Respuesta en formato JSON con los resultados de las predicciones.

## Requisitos

Asegúrate de tener instalados los siguientes paquetes antes de ejecutar el proyecto:

- Python 3.8 o superior
- TensorFlow
- Flask
- NumPy
- Pillow
- OpenCV (opcional, si se utiliza segmentación de múltiples dígitos)

### Instalación de dependencias

Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install tensorflow flask numpy pillow opencv-python

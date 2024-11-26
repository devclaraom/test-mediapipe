from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image

# Inicializar Flask
app = Flask(__name__)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def obtener_hombros(imagen):
    # Convertir la imagen a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # Obtener las predicciones de los puntos clave del cuerpo
    resultados = pose.process(imagen_rgb)

    if resultados.pose_landmarks:
        # Obtener las coordenadas de los puntos clave de los hombros
        hombro_izquierdo = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hombro_derecho = resultados.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Las coordenadas son normalizadas (0-1) en relación con el tamaño de la imagen
        return (hombro_izquierdo.x, hombro_izquierdo.y), (hombro_derecho.x, hombro_derecho.y)
    else:
        return None, None

def calcular_diferencia_en_altura(hombro_izq, hombro_der, ancho_imagen, alto_imagen):
    # Obtener la diferencia en las coordenadas `y` (que son entre 0 y 1)
    diferencia_y = abs(hombro_izq[1] - hombro_der[1])
    
    # Convertir la diferencia de coordenadas `y` a píxeles
    diferencia_px = diferencia_y * alto_imagen
    
    return diferencia_px

@app.route('/calcular_hombros', methods=['POST'])
def calcular_hombros():
    # Obtener la imagen desde la solicitud
    file = request.files['image']
    img = Image.open(file.stream)
    img = np.array(img)

    # Obtener las coordenadas de los hombros
    hombro_izq, hombro_der = obtener_hombros(img)

    if hombro_izq and hombro_der:
        # Obtener las dimensiones de la imagen
        alto_imagen, ancho_imagen, _ = img.shape
        
        # Calcular la diferencia en altura en píxeles
        diferencia_px = calcular_diferencia_en_altura(hombro_izq, hombro_der, ancho_imagen, alto_imagen)
        
        # Determinar cuál hombro está más arriba
        if hombro_izq[1] < hombro_der[1]:
            resultado = {
                'mensaje': f"El hombro izquierdo está más arriba por {diferencia_px:.2f} píxeles.",
                'diferencia': diferencia_px
            }
        elif hombro_izq[1] > hombro_der[1]:
            resultado = {
                'mensaje': f"El hombro derecho está más arriba por {diferencia_px:.2f} píxeles.",
                'diferencia': diferencia_px
            }
        else:
            resultado = {
                'mensaje': "Ambos hombros están al mismo nivel.",
                'diferencia': 0
            }

        return jsonify(resultado)
    else:
        return jsonify({'mensaje': 'No se detectaron los hombros en la imagen.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
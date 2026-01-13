"""Cliente de captura de vídeo e envio de previsões para a API.

Neste cliente, capturo uma mão com MediaPipe, extraio 63 características
relativas ao pulso (invariantes à posição e normalizadas por amostra), e envio
essas características para o endpoint Flask que devolve a letra prevista e a
confiança associada. Mantenho um histórico curto das previsões e apresento a
moda para evitar flutuações instantâneas.

Configurações principais:
- CAMERA_INDEX: índice da câmara a usar.
- SERVER_URL: URL do endpoint /predict.
- CONF_THRESHOLD: confiança mínima para aceitar e acumular previsões.
- MAX_HISTORY_LEN: tamanho da janela para calcular a moda.
- REQUEST_TIMEOUT: tempo limite para pedidos à API.
"""

import cv2
import mediapipe as mp
import requests
import threading
import time
from collections import deque
from statistics import mode
from typing import List

SERVER_URL = "http://127.0.0.1:5000/predict"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.50
MAX_HISTORY_LEN = 10
REQUEST_TIMEOUT = 2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAMERA_INDEX)

current_prediction = "..."
current_confidence = ""
is_processing = False
prediction_history = deque(maxlen=MAX_HISTORY_LEN)


def extract_relative_features(landmarks) -> List[float]:
    """
    63 características relativas ao pulso e normalizadas.

    Passos para garantir consistência com o treino:
    1) Obter as coordenadas do pulso (landmark 0).
    2) Subtrair o pulso a todos os 21 landmarks (invariância a translação).
    3) Calcular o máximo absoluto por amostra e dividir todas as coordenadas por
       esse valor (redução de variação de escala).
    4) Devolver uma lista [x1,y1,z1, ..., x21,y21,z21] com 63 floats.
    """
    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
    relative_coords = []
    max_val = 0.0
    for lm in landmarks:
        rx = lm.x - wrist_x
        ry = lm.y - wrist_y
        rz = lm.z - wrist_z
        relative_coords.append((rx, ry, rz))
        max_val = max(max_val, abs(rx), abs(ry), abs(rz))
    if max_val == 0:
        max_val = 1.0
    features: List[float] = []
    for rx, ry, rz in relative_coords:
        features.extend([rx / max_val, ry / max_val, rz / max_val])
    return features


def send_to_api(features):
    """
    Envio das características para a API e atualizando o estado global.

    Corro esta função numa thread para não bloquear o ciclo de captura. 
    Se a confiança devolvida for superior ao limiar, adiciono a previsão ao
    histórico e uso a moda das últimas 10 previsões. 
    Caso contrário, não acumulo para manter estabilidade visual.
    """
    global current_prediction, current_confidence, is_processing
    
    try:
        response = requests.post(SERVER_URL, json={'features': features}, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            pred = data.get('prediction', '?')
            conf = float(data.get('confidence', 0.0))
            
            if conf > CONF_THRESHOLD:
                prediction_history.append(pred)
                
                try:
                    most_common = mode(prediction_history)
                    current_prediction = most_common
                except:
                    current_prediction = pred
                    
                current_confidence = f"{conf:.2f}"
            else:
                current_prediction = "..."
                current_confidence = f"{conf:.2f}"
    except Exception as e:
        print(f"Erro API: {e}")
    finally:
        is_processing = False

print("Client App (Threaded) iniciado.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not is_processing:
                features = extract_relative_features(hand_landmarks.landmark)
                is_processing = True
                t = threading.Thread(target=send_to_api, args=(features,))
                t.daemon = True
                t.start()

    cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Letra: {current_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {current_confidence}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('ASL Threaded Client', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

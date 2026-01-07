import cv2
import mediapipe as mp
import requests
import threading
import time

# --- CONFIGURAÇÃO ---
SERVER_URL = "http://127.0.0.1:5000/predict"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Variáveis globais partilhadas entre a câmara e a Thread da API
current_prediction = "..."
current_confidence = ""
is_processing = False  # Para impedir que enviemos 100 pedidos ao mesmo tempo

def send_to_api(features):
    """Esta função corre em paralelo (background)"""
    global current_prediction, current_confidence, is_processing
    
    try:
        response = requests.post(SERVER_URL, json={'features': features}, timeout=2)
        if response.status_code == 200:
            data = response.json()
            pred = data.get('prediction', '?')
            conf = float(data.get('confidence', 0.0))
            
            if conf > 0.50:
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

    # frame = cv2.flip(frame, 1) # Liga/Desliga conforme o teu treino
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Só enviamos novo pedido se o anterior já tiver acabado!
            if not is_processing:
                # Extrair Features Relativas
                landmarks = hand_landmarks.landmark
                wrist = landmarks[0]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
                
                features = []
                for lm in landmarks:
                    # Usar 3 coordenadas (X, Y, Z) conforme o teu treino atual
                    features.extend([
                        lm.x - wrist_x, 
                        lm.y - wrist_y, 
                        lm.z - wrist_z
                    ])
                
                # Iniciar Thread
                is_processing = True
                t = threading.Thread(target=send_to_api, args=(features,))
                t.daemon = True # Mata a thread se fechares a app
                t.start()

    # UI (Isto corre super rápido, sem travar)
    cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Letra: {current_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {current_confidence}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('ASL Threaded Client', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
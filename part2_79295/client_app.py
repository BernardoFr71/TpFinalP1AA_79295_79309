# client_app.py
import cv2
import requests
import json
import time
import numpy as np
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações da API
API_URL = "http://127.0.0.1:5000/predict"

# Configurações da Câmera
WINDOW_NAME = "ASL Detector Client"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def run_client():
    # Inicializar Extrator
    extractor = HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prediction_text = "Wait..."
    confidence_text = ""
    
    # Controle de frequência de requisições (para não sobrecarregar)
    last_req_time = 0
    REQ_INTERVAL = 0.1 # Enviar a cada 100ms

    print("Iniciando Cliente. Pressione 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # Espelhar
            
            # 1. Extração de Landmarks
            hands_data = extractor.process_image_landmarks(frame)
            
            # Desenhar landmarks
            if hands_data:
                frame = extractor.draw_landmarks(frame, hands_data)
                
                # Preparar dados para API
                # Vamos pegar a primeira mão detetada
                hand_info = hands_data[0]
                landmarks = hand_info['landmarks_normalized'] # Shape (21, 3)
                flat_landmarks = landmarks.flatten().tolist()
                
                # 2. Comunicação com API (Não bloqueante idealmente, mas simples aqui)
                curr_time = time.time()
                if curr_time - last_req_time > REQ_INTERVAL:
                    try:
                        payload = {'landmarks': flat_landmarks}
                        response = requests.post(API_URL, json=payload, timeout=0.5)
                        
                        if response.status_code == 200:
                            result = response.json()
                            pred = result.get('prediction', '?')
                            conf = result.get('confidence', 0.0)
                            
                            prediction_text = f"Letter: {pred}"
                            confidence_text = f"Conf: {conf:.2f}"
                        else:
                            print("API Error:", response.status_code)
                            
                    except Exception as e:
                        prediction_text = "API Offline"
                        print(f"Connection error: {e}")
                    
                    last_req_time = curr_time

            # 3. Visualização / Feedback Visual
            # Caixa de fundo para texto
            cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
            
            color = (0, 255, 0) if "Wait" not in prediction_text and "Offline" not in prediction_text else (0, 0, 255)
            
            cv2.putText(frame, prediction_text, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, confidence_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client()
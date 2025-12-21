# client_app.py
import cv2
import requests
import numpy as np
from hand_landmark_extractor import HandLandmarkExtractor

API_URL = "http://127.0.0.1:5002/predict" 
WINDOW_NAME = "ASL Detector"
CONFIDENCE_THRESHOLD = 0.5 

def process_landmarks_for_api(landmarks):
    """
    Aplica a mesma lógica do create_dataset: Relativo + Escala
    input: ndarray (21, 3)
    output: lista plana de 63 floats normalizados
    """
    # Converter para lista
    if isinstance(landmarks, np.ndarray):
        landmarks = landmarks.tolist()

    # 1. Relativo ao Pulso
    base_x, base_y, base_z = landmarks[0][0], landmarks[0][1], landmarks[0][2]
    relative_list = []
    for point in landmarks:
        relative_list.append([
            point[0] - base_x,
            point[1] - base_y,
            point[2] - base_z
        ])
    
    # 2. Achatar
    flat_list = [item for sublist in relative_list for item in sublist]
    
    # 3. Normalizar Escala (IMPORTANTE)
    max_value = max(list(map(abs, flat_list)))
    
    # Evitar divisão por zero (caso raro de mão colapsada num ponto)
    if max_value == 0: 
        return flat_list

    normalized_list = [n / max_value for n in flat_list]
    return normalized_list

def run_client():
    extractor = HandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    pred_text = "..."
    conf_text = ""
    color = (200, 200, 200)
    
    print("Cliente iniciado. Pressione 'q' para sair.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Espelhar imagem (opcional - remove se atrapalhar)
        frame = cv2.flip(frame, 1)

        hands_data = extractor.process_image_landmarks(frame)

        if hands_data:
            frame = extractor.draw_landmarks(frame, hands_data)
            
            # Obter landmarks brutos (21, 3)
            raw_landmarks = hands_data[0].get('landmarks_normalized')
            
            # Processar com a nova matemática
            final_data = process_landmarks_for_api(raw_landmarks)

            try:
                # Enviar para a API
                res = requests.post(API_URL, json={'landmarks': final_data}, timeout=0.1)
                
                if res.status_code == 200:
                    data = res.json()
                    letter = data['class']
                    confidence = data['confidence']
                    
                    pred_text = f"Letra: {letter}"
                    conf_text = f"Conf: {confidence:.2f}"
                    
                    if confidence > 0.8:
                        color = (0, 255, 0) # Verde forte
                    elif confidence > CONFIDENCE_THRESHOLD:
                        color = (0, 255, 255) # Amarelo
                    else:
                        color = (0, 0, 255) # Vermelho
                
            except requests.exceptions.ConnectionError:
                pred_text = "API OFF"
            except Exception:
                pass
        else:
            pred_text = "Mao nao detetada"
            conf_text = ""
            color = (200, 200, 200)

        # Desenhar Interface
        cv2.rectangle(frame, (0, 0), (250, 80), (0, 0, 0), -1)
        cv2.putText(frame, pred_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, conf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client()
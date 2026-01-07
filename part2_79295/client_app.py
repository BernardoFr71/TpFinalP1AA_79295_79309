import cv2
import requests
import time
import numpy as np
from hand_landmark_extractor import HandLandmarkExtractor

# --- Configurações ---
API_URL = "http://127.0.0.1:5000/predict"  # Certifica-te que a porta bate certo com o app.py (5000)
CAMERA_ID = 0                               # 0 para webcam integrada, 1 para externa
WINDOW_NAME = "ASL Real-Time Detector"
SKIP_FRAMES = 5                             # Enviar para API apenas a cada N frames (evita lag)
CONFIDENCE_THRESHOLD = 0.6                  # Limiar para considerar a predição válida

def make_landmarks_relative(landmarks_list):
    """
    Torna as coordenadas relativas ao pulso (ponto 0).
    Isto é CRUCIAL para garantir que o input da API corresponde ao treino.
    """
    if not landmarks_list or len(landmarks_list) < 3:
        return landmarks_list

    # O pulso é sempre os primeiros 3 valores (x, y, z)
    base_x = landmarks_list[0]
    base_y = landmarks_list[1]
    base_z = landmarks_list[2]

    relative_landmarks = []
    # Itera de 3 em 3 (x, y, z) para subtrair a base
    for i in range(0, len(landmarks_list), 3):
        relative_landmarks.append(landmarks_list[i] - base_x)
        relative_landmarks.append(landmarks_list[i+1] - base_y)
        relative_landmarks.append(landmarks_list[i+2] - base_z)

    return relative_landmarks

def run_client():
    # Inicializar Extrator
    # min_detection_confidence=0.7 ajuda a filtrar 'falsos positivos' (ruído de fundo)
    extractor = HandLandmarkExtractor(static_image_mode=False, min_detection_confidence=0.7)
    
    # Inicializar Câmara
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ Erro: Não foi possível aceder à câmara {CAMERA_ID}.")
        return

    # Variáveis de Estado da Interface
    current_letter = "..."
    current_conf = 0.0
    status_color = (200, 200, 200) # Cinzento
    frame_count = 0

    print(f"✅ Cliente iniciado. Acedendo a {API_URL}")
    print("ℹ️  Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do stream de vídeo.")
            break

        # Espelhar a imagem (opcional, mas mais natural para o utilizador)
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # 1. Extração de Landmarks
        results, landmarks = extractor.extract_landmarks(frame)

        # 2. Desenhar Esqueleto
        frame = extractor.draw_landmarks(frame, results)

        # 3. Lógica de Predição (A cada X frames e se houver mão)
        if landmarks and len(landmarks) == 63:
            if frame_count % SKIP_FRAMES == 0:
                try:
                    # Pré-processamento local (apenas coordenadas relativas)
                    processed_landmarks = make_landmarks_relative(landmarks)
                    
                    # Enviar para a API
                    payload = {'landmarks': processed_landmarks}
                    response = requests.post(API_URL, json=payload, timeout=0.5)

                    if response.status_code == 200:
                        data = response.json()
                        current_letter = data.get('letter', '?')
                        current_conf = data.get('confidence', 0.0)

                        # Atualizar cor baseado na confiança
                        if current_conf > 0.8:
                            status_color = (0, 255, 0)   # Verde (Excelente)
                        elif current_conf > CONFIDENCE_THRESHOLD:
                            status_color = (0, 255, 255) # Amarelo (Razoável)
                        else:
                            status_color = (0, 0, 255)   # Vermelho (Dúvida)
                    else:
                        print(f"⚠️ Erro API: {response.status_code}")

                except requests.exceptions.ConnectionError:
                    current_letter = "API OFF"
                    status_color = (0, 0, 255)
                except Exception as e:
                    print(f"Erro: {e}")
        else:
            # Se não detetar mão, reseta o estado visual
            if frame_count % 10 == 0: # Delay para não piscar
                current_letter = "..."
                current_conf = 0.0
                status_color = (200, 200, 200)

        # 4. Interface Gráfica (HUD)
        # Fundo preto semi-transparente para o texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 90), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Texto da Letra
        cv2.putText(frame, f"Letra: {current_letter}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2, cv2.LINE_AA)
        
        # Texto da Confiança e Status
        conf_percent = f"{current_conf*100:.1f}%" if isinstance(current_conf, float) else ""
        cv2.putText(frame, f"Confianca: {conf_percent}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Mostrar Janela
        cv2.imshow(WINDOW_NAME, frame)

        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client()
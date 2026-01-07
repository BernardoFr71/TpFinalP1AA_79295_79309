"""
Teste de Câmara Profissional (Compatível com HandLandmarkExtractor do Professor)

Funcionalidades:
- Visualização em tempo real dos landmarks.
- Cálculo de FPS (Performance).
- Gravação de dados (CSV + Imagem) com feedback visual.
- Interface gráfica (HUD) limpa.

Teclas:
- 'q': Sair
- 's': Salvar amostra atual (CSV e JPG)
"""

import cv2
import time
import os
import pandas as pd
from hand_landmark_extractor import HandLandmarkExtractor

# --- Configurações ---
WINDOW_NAME = "Teste de Camera - Deteccao de Maos"
CSV_OUTPUT_PATH = "amostras_capturadas.csv"
IMG_OUTPUT_DIR = "amostras_img"
CAM_INDEX = 0
FRAME_WIDTH = 1280  # Tenta HD se a câmara suportar
FRAME_HEIGHT = 720

# Cores (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_BG_HUD = (0, 0, 0)
COLOR_OK = (0, 255, 0)
COLOR_WARN = (0, 165, 255)

def run_camera_test():
    # 1. Configurar Extrator (Usando parâmetros do professor)
    extractor = HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        suppress_warnings=True
    )

    # 2. Inicializar Câmara
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERRO: Não foi possível abrir a câmara {CAM_INDEX}.")
        return

    # Tentar definir resolução (se a câmara não suportar, usa o padrão)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Criar pasta para imagens se não existir
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

    print("--- Teste de Câmara Iniciado ---")
    print(f"Saída de dados: {CSV_OUTPUT_PATH}")
    print("Pressione 's' para salvar amostra, 'q' para sair.")

    prev_time = time.time()
    save_feedback_timer = 0  # Contador para mostrar mensagem "Guardado"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame.")
                break

            # Espelhar imagem (UX mais natural)
            frame = cv2.flip(frame, 1)

            # 3. Processamento (Métodos do Professor)
            # Retorna lista de dicionários
            hands_data = extractor.process_image_landmarks(frame)
            
            # Desenha os pontos na imagem
            frame = extractor.draw_landmarks(frame, hands_data)

            # 4. Cálculos de Interface
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            num_hands = len(hands_data)
            status_color = COLOR_OK if num_hands > 0 else COLOR_WARN

            # 5. Desenhar HUD (Heads-Up Display)
            # Fundo semitransparente para texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (350, 110), COLOR_BG_HUD, -1)
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Texto informativo
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            
            cv2.putText(frame, f"Maos Detetadas: {num_hands}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(frame, "[Q] Sair  [S] Salvar", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 6. Lógica de Gravação ('s')
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if hands_data:
                    # Método do professor para converter para DataFrame
                    df = extractor.hands_data_to_dataframe(hands_data)
                    
                    # Adicionar timestamp ou ID se quiseres (opcional)
                    df['timestamp'] = curr_time

                    # Salvar CSV (Append mode)
                    file_exists = os.path.exists(CSV_OUTPUT_PATH)
                    df.to_csv(CSV_OUTPUT_PATH, mode='a', header=not file_exists, index=False)
                    
                    # Salvar Imagem de referência
                    img_name = f"sample_{int(curr_time)}.jpg"
                    cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, img_name), frame)
                    
                    print(f"Amostra guardada: {len(df)} linhas.")
                    save_feedback_timer = 30 # Mostrar mensagem por 30 frames
                else:
                    print("Aviso: Nenhuma mão detetada para salvar.")

            # Feedback Visual de Gravação
            if save_feedback_timer > 0:
                cv2.putText(frame, "DADOS GUARDADOS!", (width//2 - 100, height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                save_feedback_timer -= 1

            # Obter dimensões para centrar texto (opcional, usado acima)
            height, width = frame.shape[:2]

            cv2.imshow(WINDOW_NAME, frame)

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 7. Limpeza Segura
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Recursos libertados. Programa terminado.")

if __name__ == "__main__":
    run_camera_test()
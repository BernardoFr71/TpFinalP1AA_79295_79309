import cv2
import requests
import json
import threading
import time
from hand_landmark_extractor import HandLandmarkExtractor

class RealTimeASLApp:
    def __init__(self, endpoint="http://localhost:5000/predict"):
        self.server_url = endpoint
        
        # Inicializa o processador de MediaPipe (Video Mode = True para fluidez)
        self.hand_processor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1)
        
        # Variáveis de Estado da Aplicação
        self.current_label = "A aguardar..."
        self.confidence_score = 0.0
        self.is_request_active = False

    def _send_inference_request(self, input_data):
        """
        Método assíncrono para contactar a API sem congelar a webcam.
        """
        self.is_request_active = True
        try:
            # Envio do POST request
            api_response = requests.post(
                self.server_url, 
                json=input_data, 
                timeout=0.5
            )
            
            if api_response.status_code == 200:
                result_json = api_response.json()
                # Atualiza as variáveis que aparecem no ecrã
                self.current_label = result_json.get('letter', '?')
                self.confidence_score = result_json.get('confidence', 0.0)
                
        except Exception:
            # Falhas na conexão são ignoradas para não parar o vídeo
            pass
        
        # Liberta a flag para permitir novos pedidos
        self.is_request_active = False

    def start_capture(self):
        # Configuração da Webcam
        cam_stream = cv2.VideoCapture(0)
        cam_stream.set(cv2.CAP_PROP_FPS, 30)
        
        print("Transmissão iniciada. Pressiona 'q' para encerrar.")
        
        frame_counter = 0
        
        while cam_stream.isOpened():
            ret, frame = cam_stream.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # Lógica de Throttling: Processa apenas 1 em cada 5 frames
            # E apenas se não houver um pedido HTTP já em andamento
            if frame_counter % 5 == 0 and not self.is_request_active:
                
                # 1. Extração de Features
                landmarks_result = self.hand_processor.process_image_landmarks(frame)
                
                if landmarks_result:
                    # 2. Prepara os dados (Converte para DataFrame e depois Dicionário)
                    # Nota: Passamos landmarks_result[0] numa lista conforme a classe exige
                    df_features = self.hand_processor.hands_data_to_dataframe([landmarks_result[0]])
                    json_body = df_features.iloc[0].to_dict()
                    
                    # 3. Threading (Processamento em segundo plano)
                    bg_thread = threading.Thread(target=self._send_inference_request, args=(json_body,))
                    bg_thread.start()
            
            # --- Renderização da Interface (UI) ---
            
            # Caixa de fundo preta
            cv2.rectangle(frame, (5, 15), (380, 65), (20, 20, 20), -1)
            
            # Texto com a previsão
            display_text = f"Gesto: {self.current_label} ({self.confidence_score:.1%})"
            color = (0, 255, 0) if self.confidence_score > 0.5 else (0, 165, 255) # Verde se > 50%, Laranja se <
            
            cv2.putText(frame, display_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, color, 2, cv2.LINE_AA)
            
            cv2.imshow('ASL Real-Time Detector', frame)
            
            # Tecla de saída
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealTimeASLApp()
    app.start_capture()
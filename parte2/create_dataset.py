# create_dataset.py
import os
import cv2
import pandas as pd
import numpy as np
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações
DATASET_DIR = "dataset/SignAlphaSet"
OUTPUT_CSV = "hand_landmarks_dataset.csv"

def process_landmarks(landmarks):
    """
    1. Converte para coordenadas relativas ao pulso.
    2. Normaliza a escala (divide pelo maior valor) para ficar entre -1 e 1.
    """
    # Copiar para não alterar o original
    # Landmarks vem como lista de listas [[x,y,z], [x,y,z]...]
    
    # 1. Tornar relativo ao Pulso (Ponto 0)
    base_x, base_y, base_z = landmarks[0][0], landmarks[0][1], landmarks[0][2]

    relative_list = []
    for point in landmarks:
        relative_list.append([
            point[0] - base_x,
            point[1] - base_y,
            point[2] - base_z
        ])
    
    # 2. Achatar a lista para (63 values)
    flat_list = [item for sublist in relative_list for item in sublist]
    
    # 3. Normalizar pela escala (Máximo valor absoluto)
    # Isto garante que o tamanho da mão não afeta a classificação
    max_value = max(list(map(abs, flat_list)))
    
    def normalize(n):
        return n / max_value

    normalized_list = list(map(normalize, flat_list))
    return normalized_list

def create_dataset():
    extractor = HandLandmarkExtractor(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    data_list = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"Erro: Diretoria {DATASET_DIR} não encontrada.")
        return

    labels = sorted(os.listdir(DATASET_DIR))
    print(f"A processar classes: {labels}")

    for label in labels:
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir): continue
            
        print(f"--> A processar classe: {label}")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
                
            hands_data = extractor.process_image_landmarks(image)
            
            if hands_data:
                hand_info = hands_data[0] 
                landmarks = hand_info['landmarks_normalized'] # ndarray (21, 3)
                
                # Converter para lista Python para facilitar manipulação
                landmarks_list = landmarks.tolist()
                
                # --- NOVO PROCESSAMENTO ---
                final_landmarks = process_landmarks(landmarks_list)
                
                row = [label] + final_landmarks
                data_list.append(row)

    extractor.close()

    # Criar colunas
    cols = ['label']
    for i in range(21):
        cols.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
        
    df = pd.DataFrame(data_list, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConcluído! Dataset salvo em {OUTPUT_CSV} com {len(df)} amostras.")

if __name__ == "__main__":
    create_dataset()
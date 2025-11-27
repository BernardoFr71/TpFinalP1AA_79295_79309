# create_dataset.py
import os
import cv2
import pandas as pd
import numpy as np
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações
DATASET_DIR = "dataset/SignAlphaSet"  # Caminho para as pastas A, B, C...
OUTPUT_CSV = "hand_landmarks_dataset.csv"

def create_dataset():
    extractor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=1, # Geralmente datasets de letras são 1 mão
        min_detection_confidence=0.5
    )
    
    data_list = []
    
    # Verificar se diretoria existe
    if not os.path.exists(DATASET_DIR):
        print(f"Erro: Diretoria {DATASET_DIR} não encontrada.")
        return

    # Iterar sobre as pastas (A-Z)
    labels = sorted(os.listdir(DATASET_DIR))
    
    print(f"A processar classes: {labels}")

    for label in labels:
        class_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"--> A processar classe: {label}")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Ler imagem
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # Processar landmarks
            # O extrator devolve uma lista de dicionários
            hands_data = extractor.process_image_landmarks(image)
            
            if hands_data:
                # Vamos assumir a primeira mão detetada para classificação
                hand_info = hands_data[0] 
                landmarks = hand_info['landmarks_normalized'] # Shape (21, 3)
                
                # Achatamento (Flatten): converter (21, 3) para vetor de 63 valores
                flat_landmarks = landmarks.flatten().tolist()
                
                # Adicionar label e dados à lista
                # Estrutura: [label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
                row = [label] + flat_landmarks
                data_list.append(row)

    extractor.close()

    # Criar DataFrame
    # Gerar nomes das colunas: WRIST_x, WRIST_y, WRIST_z, etc.
    cols = ['label']
    landmark_names = extractor.landmark_names
    for name in landmark_names:
        cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        
    df = pd.DataFrame(data_list, columns=cols)
    
    # Salvar CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConcluído! Dataset salvo em {OUTPUT_CSV} com {len(df)} amostras.")

if __name__ == "__main__":
    create_dataset()
import os
import cv2
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações Padrão
DATASET_DIR = "dataset/SignAlphaSet"
OUTPUT_CSV = "hand_landmarks_dataset.csv"
OUTPUT_AUG_CSV = "hand_landmarks_augmented.csv"

def make_landmarks_relative(landmarks_list):
    """
    Converte coordenadas absolutas para relativas ao pulso (Landmark 0).
    """
    if not landmarks_list or len(landmarks_list) < 3:
        return landmarks_list

    base_x = landmarks_list[0]
    base_y = landmarks_list[1]
    base_z = landmarks_list[2]

    relative_landmarks = []
    # Itera de 3 em 3 (x, y, z)
    for i in range(0, len(landmarks_list), 3):
        relative_landmarks.append(landmarks_list[i] - base_x)
        relative_landmarks.append(landmarks_list[i+1] - base_y)
        relative_landmarks.append(landmarks_list[i+2] - base_z)

    return relative_landmarks

def create_dataset(input_dir=DATASET_DIR, output_file=OUTPUT_CSV):
    if not os.path.exists(input_dir):
        print(f"ERRO CRÍTICO: Diretoria '{input_dir}' não encontrada.")
        return

    # Usar parâmetros do professor
    extractor = HandLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.5)
    
    data_list = []
    labels = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"A iniciar processamento de {len(labels)} classes...")

    for label in tqdm(labels, desc="Processando Classes"):
        class_dir = os.path.join(input_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            # --- ADAPTAÇÃO AO CÓDIGO DO PROFESSOR ---
            # Retorna lista de dicionários
            hands_data = extractor.process_image_landmarks(image)
            
            if hands_data:
                # Processa apenas a primeira mão para consistência
                first_hand = hands_data[0]
                
                hand_label = first_hand.get('handedness', 'Unknown')
                
                # O professor devolve um numpy array (21, 3) em 'landmarks_normalized'
                landmarks_arr = first_hand.get('landmarks_normalized')
                
                if landmarks_arr is not None:
                    # Achatar para lista (63 floats)
                    flat_landmarks = landmarks_arr.flatten().tolist()
                    
                    # Tornar relativo ao pulso
                    processed_landmarks = make_landmarks_relative(flat_landmarks)
                    
                    row = [label, hand_label] + processed_landmarks
                    data_list.append(row)

    # Nomes das colunas: x_0, y_0, z_0 ...
    cols = ['label', 'hand']
    for i in range(21):
        cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    df = pd.DataFrame(data_list, columns=cols)
    df.to_csv(output_file, index=False)
    
    print(f"\nDataset criado com sucesso em: {output_file}")
    print(f"Total de amostras: {len(df)}")

def augment_process(input_csv, output_aug_csv, classes, n_samples):
    try:
        from augment_landmarks import main as augment_main
        print(f"\n--- A iniciar Data Augmentation ---")
        augment_main(input_csv, output_aug_csv, classes, n_samples)
    except ImportError:
        print("AVISO: 'augment_landmarks.py' não encontrado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DATASET_DIR)
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--aug-output', default=OUTPUT_AUG_CSV)
    parser.add_argument('--aug-classes', nargs='+', default=['J', 'Z'])
    parser.add_argument('--aug-n', type=int, default=500)

    args = parser.parse_args()

    create_dataset(input_dir=args.input, output_file=args.output)
    if args.augment:
        augment_process(args.output, args.aug_output, args.aug_classes, args.aug_n)
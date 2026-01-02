# create_dataset.py
import os
import cv2
import pandas as pd
import numpy as np
import argparse
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações
DATASET_DIR = "dataset/SignAlphaSet"
OUTPUT_CSV = "hand_landmarks_dataset.csv"
OUTPUT_AUG_CSV = "hand_landmarks_augmented.csv"

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

def create_dataset(output_csv=OUTPUT_CSV):
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
                    # se houver múltiplas mãos, ficar só com a primeira
                    hand_info = hands_data[0]
                    landmarks = hand_info.get('landmarks_normalized')  # ndarray (21, 3)
                    handedness = hand_info.get('handedness', 'Unknown')

                    # Converter para lista Python para facilitar manipulação
                    landmarks_list = landmarks.tolist()

                    # --- NOVO PROCESSAMENTO ---
                    final_landmarks = process_landmarks(landmarks_list)

                    # Incluir a indicação da mão (Left/Right) na segunda coluna
                    row = [label, handedness] + final_landmarks
                    data_list.append(row)

    extractor.close()

    # Criar colunas
    cols = ['label', 'hand']
    for i in range(21):
        cols.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
        
    df = pd.DataFrame(data_list, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"\nConcluído! Dataset salvo em {output_csv} com {len(df)} amostras.")


def create_and_maybe_augment(output_csv=OUTPUT_CSV, augment=False, augmented_output=OUTPUT_AUG_CSV, augment_classes=None, augment_n=1000):
    # Cria o dataset base
    create_dataset(output_csv=output_csv)

    # Se pedido, chama o script de augmentação (importa a função main de augment_landmarks)
    if augment:
        try:
            from augment_landmarks import main as augment_main
        except Exception as e:
            print(f"Erro ao importar augment_landmarks: {e}")
            return

        classes = augment_classes if augment_classes is not None else ['C','D','E']
        print(f"A iniciar augmentação para classes: {classes} (adicionando {augment_n} por classe)...")
        augment_main(input_csv=output_csv, output_csv=augmented_output, classes=classes, n_per_class=augment_n)
        print(f"Augmentação concluída. Ficheiro aumentado: {augmented_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Criar dataset de landmarks a partir de imagens e opcionalmente augmentar')
    parser.add_argument('--output', default=OUTPUT_CSV, help='CSV de saída base')
    parser.add_argument('--augment', action='store_true', help='Gerar dataset augmentado chamando augment_landmarks')
    parser.add_argument('--augmented-output', default=OUTPUT_AUG_CSV, help='CSV de saída aumentado')
    parser.add_argument('--augment-classes', nargs='+', default=['C','D','E'], help='Classes alvo para augmentação')
    parser.add_argument('--augment-n', type=int, default=1000, help='Número de amostras a adicionar por classe')
    args = parser.parse_args()

    create_and_maybe_augment(output_csv=args.output, augment=args.augment, augmented_output=args.augmented_output, augment_classes=args.augment_classes, augment_n=args.augment_n)
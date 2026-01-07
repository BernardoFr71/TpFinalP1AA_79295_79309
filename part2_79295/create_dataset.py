import os
import cv2
import pandas as pd
import argparse
import sys
from tqdm import tqdm
from hand_landmark_extractor import HandLandmarkExtractor

# Configurações Padrão
DATASET_DIR = "dataset/SignAlphaSet"
OUTPUT_CSV = "hand_landmarks_dataset.csv"
OUTPUT_AUG_CSV = "hand_landmarks_augmented.csv"

def make_landmarks_relative(landmarks_list):
    """
    Converte coordenadas absolutas para relativas ao pulso (Landmark 0).
    Isso torna o modelo invariante à posição da mão na imagem.
    
    Args:
        landmarks_list: Lista plana [x0, y0, z0, x1, y1, z1, ...]
    Returns:
        Lista com as mesmas dimensões, mas relativa ao primeiro ponto.
    """
    if not landmarks_list or len(landmarks_list) < 3:
        return landmarks_list

    # O pulso é sempre os primeiros 3 valores (x, y, z)
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
    """
    Percorre as imagens, extrai landmarks e salva num CSV.
    """
    if not os.path.exists(input_dir):
        print(f"ERRO CRÍTICO: Diretoria '{input_dir}' não encontrada.")
        return

    # Inicializar Extrator (Modo estático = True para maior precisão em imagens isoladas)
    extractor = HandLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.5)
    
    data_list = []
    
    # Obter lista de classes (pastas)
    labels = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    print(f"Classes detetadas: {labels}")
    print(f"A iniciar processamento de {len(labels)} classes...")

    for label in tqdm(labels, desc="Processando Classes"):
        class_dir = os.path.join(input_dir, label)
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Ler imagem
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Extrair Landmarks
            results, landmarks = extractor.extract_landmarks(image)
            
            if landmarks and len(landmarks) == 63:
                # 1. Obter Handedness (Esquerda/Direita)
                hand_label = "Unknown"
                if results.multi_handedness:
                    # O Google inverte (Mirror), mas para treino consistente usamos o label do output
                    hand_label = results.multi_handedness[0].classification[0].label
                
                # 2. Tornar coordenadas relativas ao pulso (Feature Engineering)
                # Nota: Não fazemos normalização de escala (divisão pelo max) aqui,
                # deixamos isso para o StandardScaler no treino (train_model.ipynb).
                processed_landmarks = make_landmarks_relative(landmarks)
                
                # 3. Criar linha de dados
                row = [label, hand_label] + processed_landmarks
                data_list.append(row)

    # Criar DataFrame e salvar CSV
    # Gerar nomes das colunas: x_0, y_0, z_0 ... x_20, y_20, z_20
    cols = ['label', 'hand']
    for i in range(21):
        cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    df = pd.DataFrame(data_list, columns=cols)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset criado com sucesso!")
    print(f"📁 Ficheiro salvo em: {output_file}")
    print(f"📊 Total de amostras: {len(df)}")
    print(f"🔢 Dimensão das features: {len(cols)}")

def augment_process(input_csv, output_aug_csv, classes, n_samples):
    """
    Função Wrapper para chamar o script de data augmentation.
    """
    try:
        from augment_landmarks import main as augment_main
        print(f"\n--- A iniciar Data Augmentation ---")
        print(f"Classes alvo: {classes}")
        print(f"Amostras a adicionar: {n_samples}")
        
        augment_main(
            input_csv=input_csv, 
            output_csv=output_aug_csv, 
            classes=classes, 
            n_per_class=n_samples
        )
        print(f"Augmentation concluída: {output_aug_csv}")
    except ImportError:
        print("AVISO: 'augment_landmarks.py' não encontrado. Augmentation ignorada.")
    except Exception as e:
        print(f"ERRO na Augmentation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fase 1: Criar Dataset de Landmarks')
    
    parser.add_argument('--input', default=DATASET_DIR, help='Caminho da pasta do dataset original')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Caminho do ficheiro CSV de saída')
    
    # Argumentos de Augmentation (Opcionais)
    parser.add_argument('--augment', action='store_true', help='Executar data augmentation após a criação')
    parser.add_argument('--aug-output', default=OUTPUT_AUG_CSV, help='Caminho do CSV aumentado')
    parser.add_argument('--aug-classes', nargs='+', default=['J', 'Z'], help='Classes para aumentar (ex: J Z)')
    parser.add_argument('--aug-n', type=int, default=500, help='Número de amostras a adicionar por classe')

    args = parser.parse_args()

    # 1. Criar Dataset Base
    create_dataset(input_dir=args.input, output_file=args.output)

    # 2. Augmentation (se solicitado)
    if args.augment:
        augment_process(args.output, args.aug_output, args.aug_classes, args.aug_n)
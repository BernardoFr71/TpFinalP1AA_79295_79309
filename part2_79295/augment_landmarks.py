import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def augment_sample(flat_landmarks, rot_deg_range=15, scale_range=(0.9, 1.1), noise_std=0.02):
    """
    Gera uma variação sintética de uma amostra de landmarks.
    Aplica: Rotação (Eixo Z), Escala e Ruído Gaussiano.
    """
    # 1. Reconstruir estrutura (21 pontos, 3 coordenadas)
    # Garante que é float para permitir operações matemáticas
    data = np.array(flat_landmarks, dtype=float).reshape(-1, 3)

    # 2. Rotação (Simular inclinação da mão no plano da câmara/Eixo Z)
    # Como os dados são relativos ao pulso (0,0,0), rodamos em torno da origem.
    theta = np.deg2rad(np.random.uniform(-rot_deg_range, rot_deg_range))
    c, s = np.cos(theta), np.sin(theta)
    
    # Matriz de rotação em torno de Z
    rotation_matrix = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Aplicar rotação (dot product)
    data = np.dot(data, rotation_matrix)

    # 3. Escala (Simular mão maior/menor ou mais perto/longe)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    data = data * scale

    # 4. Ruído (Simular imperfeições do sensor/tremor)
    noise = np.random.normal(0, noise_std, data.shape)
    data = data + noise

    return data.flatten()

def main(input_csv, output_csv, classes, n_per_class):
    if not os.path.exists(input_csv):
        print(f"ERRO: Ficheiro de entrada '{input_csv}' não encontrado.")
        return

    print(f"A carregar dataset original: {input_csv}...")
    df = pd.read_csv(input_csv)

    # Validação de Colunas Obrigatórias
    if 'label' not in df.columns:
        raise ValueError("O CSV de entrada tem de ter a coluna 'label'.")

    # Identificar colunas de features (todas exceto label e hand)
    # Isto garante compatibilidade com x_0, y_0, z_0...
    non_feature_cols = ['label', 'hand']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    if len(feature_cols) != 63:
        print(f"AVISO: Esperava 63 colunas de features, encontrei {len(feature_cols)}.")

    # Preparar lista de dados finais (começa com cópia dos originais)
    final_data = df.to_dict('records')
    
    # Se classes não forem especificadas, aumentar todas
    if not classes:
        classes = df['label'].unique().tolist()
        print("Nenhuma classe especificada. A aumentar TODAS as classes.")

    print(f"\n--- Iniciar Data Augmentation ---")
    print(f"Alvo: Adicionar {n_per_class} amostras por classe: {classes}")

    for cls in tqdm(classes, desc="Processando Classes"):
        # Filtrar dados da classe atual
        cls_df = df[df['label'] == cls]
        
        if cls_df.empty:
            print(f"Aviso: Classe '{cls}' não existe no dataset original. Ignorada.")
            continue

        # Converter para lista de arrays para performance
        existing_features = cls_df[feature_cols].values
        existing_metadata = cls_df[non_feature_cols].values # label e hand
        
        n_existing = len(existing_features)
        
        # Gerar novas amostras
        for i in range(n_per_class):
            # Escolher aleatoriamente uma amostra base (bootstrap)
            idx = np.random.randint(0, n_existing)
            base_feats = existing_features[idx]
            base_meta = existing_metadata[idx] # [label, hand] se hand existir
            
            # Aplicar transformações
            aug_feats = augment_sample(base_feats)
            
            # Construir nova linha
            new_row = dict(zip(feature_cols, aug_feats))
            
            # Adicionar metadados (label, hand)
            # Assume que non_feature_cols está alinhado com base_meta
            for meta_col, meta_val in zip(non_feature_cols, base_meta):
                new_row[meta_col] = meta_val
            
            final_data.append(new_row)

    # Criar DataFrame final e salvar
    df_out = pd.DataFrame(final_data)
    
    # Reordenar colunas para garantir consistência (label, hand, x_0...)
    ordered_cols = [c for c in df.columns if c in df_out.columns]
    df_out = df_out[ordered_cols]
    
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    
    print(f"\nConcluído!")
    print(f"Dataset original: {len(df)} amostras")
    print(f"Dataset aumentado: {len(df_out)} amostras")
    print(f"Guardado em: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para Data Augmentation de Landmarks ASL")
    
    parser.add_argument('--input', default='hand_landmarks_dataset.csv', help='CSV de entrada')
    parser.add_argument('--output', default='hand_landmarks_augmented.csv', help='CSV de saída')
    # nargs='*' permite passar 0 ou mais argumentos. Se 0, o script assume todas as classes.
    parser.add_argument('--classes', nargs='*', help='Classes a aumentar (ex: J Z). Se vazio, aumenta todas.')
    parser.add_argument('--n_per_class', type=int, default=500, help='Quantas amostras adicionar por classe')
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.classes, args.n_per_class)
"""augment_landmarks.py
Script para aumentar o dataset de landmarks gerado por create_dataset.py.
Gera amostras sintéticas por pequenas rotações, escalas e ruído gaussiano.

Uso:
python augment_landmarks.py --input hand_landmarks_dataset.csv --output hand_landmarks_augmented.csv --classes C D E --n_per_class 1000

"""
import argparse
import pandas as pd
import numpy as np
import os


def augment_sample(flat_landmarks, n_aug=1, rot_deg_range=15, scale_range=(0.95,1.05), noise_std=0.02):
    """Gera n_aug amostras a partir de um vetor de 63 valores.
    flat_landmarks: list ou np.array shape (63,)
    Retorna lista de arrays (63,)
    """
    arr = np.array(flat_landmarks).reshape(21,3)
    aug_list = []
    for _ in range(n_aug):
        # Rotação em torno do pulso (origem), aplicar a x,y
        angle = np.deg2rad(np.random.uniform(-rot_deg_range, rot_deg_range))
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s],[s, c]])
        xy = arr[:, :2]
        xy_rot = xy.dot(R.T)
        # Escala
        scale = np.random.uniform(scale_range[0], scale_range[1])
        xy_rot = xy_rot * scale
        # z: aplicar pequena escala e ruído
        z = arr[:,2] * scale + np.random.normal(0, noise_std, size=(21,))
        # adicionar ruído gaussiano a x,y
        xy_rot += np.random.normal(0, noise_std, size=xy_rot.shape)
        new_arr = np.hstack([xy_rot, z.reshape(-1,1)])
        aug_list.append(new_arr.flatten())
    return aug_list


def main(input_csv, output_csv, classes, n_per_class):
    df = pd.read_csv(input_csv)
    # Columns: label, lm0_x, lm0_y, lm0_z, ..., lm20_z
    if 'label' not in df.columns:
        raise ValueError('input CSV must have column label')
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

    out_rows = []
    # Copy original rows (preserve 'hand' column if present)
    for _, row in df.iterrows():
        out_rows.append(row.values.tolist())

    # For each target class, generate augmented samples until reach n_per_class additional samples
    for cls in classes:
        cls_rows = df[df['label'] == cls]
        if cls_rows.empty:
            print(f'Warning: classe {cls} não encontrada no dataset original')
            continue
        existing = cls_rows.values.tolist()
        current_aug = 0
        idx = 0
        print(f'Augmenting class {cls}: existing {len(existing)} samples -> adding {n_per_class} samples')
        while current_aug < n_per_class:
            sample = existing[idx % len(existing)]
            # Detect if 'hand' column exists (label, hand, features...) or not (label, features...)
            if 'hand' in df.columns:
                label = sample[0]
                hand = sample[1]
                flat = sample[2:]
                aug = augment_sample(flat, n_aug=1)[0]
                out_rows.append([label, hand] + aug.tolist())
            else:
                label = sample[0]
                flat = sample[1:]
                aug = augment_sample(flat, n_aug=1)[0]
                out_rows.append([label] + aug.tolist())
            current_aug += 1
            idx += 1
    # Criar DataFrame (preservar coluna 'hand' se existia no input)
    if 'hand' in df.columns:
        cols = ['label', 'hand'] + [f'lm{i}_{axis}' for i in range(21) for axis in ['x','y','z']]
    else:
        cols = ['label'] + [f'lm{i}_{axis}' for i in range(21) for axis in ['x','y','z']]
    df_out = pd.DataFrame(out_rows, columns=cols)
    df_out.to_csv(output_csv, index=False)
    print(f'Augmented dataset salvo em {output_csv} com {len(df_out)} amostras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='hand_landmarks_dataset.csv')
    parser.add_argument('--output', default='hand_landmarks_augmented.csv')
    parser.add_argument('--classes', nargs='+', default=['C','D','E'])
    parser.add_argument('--n_per_class', type=int, default=1000)
    args = parser.parse_args()
    main(args.input, args.output, args.classes, args.n_per_class)

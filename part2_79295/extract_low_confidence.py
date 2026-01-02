"""extract_low_confidence.py
Carrega o modelo salvo em models/ e o dataset (aumentado se existir).
Seleciona amostras cuja confiança máxima (predict_proba) é menor que um limiar e salva em models/low_confidence_samples.csv

Uso:
python extract_low_confidence.py --threshold 0.8 --output models/low_confidence_samples.csv
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler_hand_sign.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

DEFAULT_INPUTS = ['hand_landmarks_augmented.csv', 'hand_landmarks_dataset.csv']


def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def main(threshold, output_csv, input_csv=None):
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Modelo não encontrado em {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

    # Choose input CSV
    if input_csv:
        csv_path = input_csv
    else:
        csv_path = load_first_existing(DEFAULT_INPUTS)
    if csv_path is None:
        raise FileNotFoundError('Nenhum CSV de dataset encontrado (procure hand_landmarks_augmented.csv ou hand_landmarks_dataset.csv)')

    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError('O CSV precisa da coluna label')

    cols = [c for c in df.columns if c != 'label']
    X = df[cols].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            # fallback: try without scaler
            pass

    if not hasattr(model, 'predict_proba'):
        raise AttributeError('Modelo não tem método predict_proba; não é possível calcular confiança.')

    probas = model.predict_proba(X)
    confidences = np.max(probas, axis=1)
    preds_idx = np.argmax(probas, axis=1)

    # Try to convert preds to labels
    if encoder is not None:
        try:
            preds = encoder.inverse_transform(preds_idx)
        except Exception:
            preds = preds_idx.astype(str)
    else:
        preds = preds_idx.astype(str)

    out_df = df.copy()
    out_df['predicted'] = preds
    out_df['pred_idx'] = preds_idx
    out_df['confidence'] = confidences

    low_df = out_df[out_df['confidence'] < threshold].copy()

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    low_df.to_csv(output_csv, index=False)
    print(f'Salvo {len(low_df)} amostras com confiança < {threshold} em {output_csv} (base: {csv_path})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--output', default='models/low_confidence_samples.csv')
    parser.add_argument('--input', default=None)
    args = parser.parse_args()
    main(args.threshold, args.output, args.input)

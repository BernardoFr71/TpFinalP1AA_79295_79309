# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

MODEL_PATH = 'models/best_model.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
SCALER_PATH = 'models/scaler_hand_sign.pkl'
model = None
encoder = None
scaler = None

# Nomes das colunas esperados pelo modelo (igual ao create_dataset)
COL_NAMES = []
for i in range(21):
    COL_NAMES.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

def load_model():
    global model, encoder, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(ENCODER_PATH):
            encoder = joblib.load(ENCODER_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        logging.info("Modelo, encoder e scaler carregados (se existentes).")
    except Exception as e:
        logging.error(f"Erro ao carregar artefactos: {e}")

load_model()


@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não carregado'}), 500
    try:
        data = request.get_json()
        landmarks_list = data.get('landmarks')

        if not landmarks_list or len(landmarks_list) != 63:
            return jsonify({'error': 'Input inválido: envie 63 floats (relativo + normalizado)'}), 400

        # Criar DataFrame com nomes das colunas
        df_input = pd.DataFrame([landmarks_list], columns=COL_NAMES)

        # Aplicar scaler se houver (o notebook de treino guarda o scaler)
        X = df_input.values
        if scaler is not None:
            X = scaler.transform(X)

        # Previsão
        pred_idx = model.predict(X)[0]

        # Se o encoder existir, traduzir para letra
        pred_label = str(pred_idx)
        if encoder is not None:
            try:
                pred_label = encoder.inverse_transform([pred_idx])[0]
            except Exception:
                # Se o modelo já retornar etiquetas string, ignore
                pred_label = str(pred_idx)

        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)[0]
            confidence = float(np.max(probas))

        logging.info(f"Predição: {pred_label} | Confiança: {confidence:.2f}")

        return jsonify({"class": pred_label, "confidence": confidence}), 200

    except Exception as e:
        logging.error(f"Erro: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
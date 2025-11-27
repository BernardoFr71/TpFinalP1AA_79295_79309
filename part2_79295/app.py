# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd # Importante para tirar o warning
import logging
import os

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

MODEL_PATH = 'models/best_model.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
model = None
encoder = None

# Nomes das colunas esperados pelo modelo (igual ao create_dataset)
COL_NAMES = []
for i in range(21):
    COL_NAMES.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

def load_model():
    global model, encoder
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)
        logging.info("Modelo carregado.")
    except Exception as e:
        logging.error(f"Erro: {e}")

load_model()

@app.route("/predict", methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Modelo off'}), 500
    try:
        data = request.get_json()
        landmarks_list = data.get('landmarks') # Recebe já em relativo do cliente
        
        if not landmarks_list or len(landmarks_list) != 63:
            return jsonify({'error': 'Input inválido'}), 400
            
        # CORREÇÃO DO WARNING: Criar DataFrame com nomes das colunas
        df_input = pd.DataFrame([landmarks_list], columns=COL_NAMES)
        
        # Previsão
        pred_idx = model.predict(df_input)[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]
        
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(df_input)[0]
            confidence = float(np.max(probas))

        logging.info(f"Predição: {pred_label} | Confiança: {confidence:.2f}")

        return jsonify({"class": pred_label, "confidence": confidence}), 200

    except Exception as e:
        logging.error(f"Erro: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
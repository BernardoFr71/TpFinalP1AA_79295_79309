# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
import os

# 1. Configuração de Logging (como no ficheiro do professor)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Caminhos dos ficheiros
MODEL_PATH = 'models/best_model.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

# Variáveis globais para o modelo
model = None
encoder = None

def load_model():
    """Carrega o modelo e o encoder do disco."""
    global model, encoder
    try:
        # Verifica se os ficheiros existem
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            logging.error("Ficheiros de modelo não encontrados. Execute a Fase 2 primeiro.")
            return False

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
            
        logging.info("Modelo e Encoder carregados com sucesso.")
        
        # Log das classes (como no exemplo do professor)
        logging.info(f"Classes treinadas: {encoder.classes_}")
        return True
    except Exception as e:
        logging.error(f"Erro ao carregar modelo: {e}")
        return False

# Carregar no arranque
load_model()

# Rota de Ajuda/Uso (Inspirado no endpoint 'usage' do professor)
@app.route("/usage")
@app.route("/")
def usage():
    return jsonify({
        "info": "API de Classificação de Língua Gestual (ASL)",
        "endpoint": "/predict",
        "method": "POST",
        "format_expected": {
            "landmarks": "Lista de 63 floats (x,y,z para 21 pontos)"
        },
        "example_python_requests": "requests.post(url, json={'landmarks': [0.5, 0.2, ...]})"
    })

# Rota de Predição
@app.route("/predict", methods=['POST'])
def predict():
    if not model or not encoder:
        return jsonify({'error': 'Modelo não carregado', 'code': 500}), 500

    try:
        # Obter JSON
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nenhum dado JSON recebido', 'code': 400}), 400

        landmarks_list = data.get('landmarks')
        
        # Validação do input
        if not landmarks_list or len(landmarks_list) != 63:
            return jsonify({
                'error': f'Input inválido. Esperados 63 valores, recebidos {len(landmarks_list) if landmarks_list else 0}.',
                'code': 400
            }), 400
            
        # Converter para numpy array 2D
        instance = np.array([landmarks_list], dtype=float)
        
        # Previsão
        pred_idx = model.predict(instance)[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]
        
        # Probabilidade (Confiança)
        confidence = 0.0
        details = {}
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(instance)[0]
            confidence = float(np.max(probas))
            
            # Detalhes (como o professor fez em "details")
            # Mapear cada classe à sua probabilidade
            details = {cls: float(prob) for cls, prob in zip(encoder.classes_, probas)}

        logging.info(f"Predição: {pred_label} | Confiança: {confidence:.2f}")

        return jsonify({
            "class": pred_label,
            "confidence": confidence,
            "details": details, # Probabilidade de todas as outras letras
            "code": 200
        }), 200

    except Exception as e:
        logging.error(f"Erro na predição: {e}")
        return jsonify({'error': str(e), 'code': 500}), 500

# Tratamento de erro 404 (Igual ao do professor)
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "code": 404,
        "message": "Rota não encontrada. Consulte /usage."
    }), 404

if __name__ == '__main__':
    # run the API !
    app.run(
        host='0.0.0.0',
        port=5002, 
        debug=True
    )
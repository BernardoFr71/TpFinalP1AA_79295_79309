import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Inicialização da App ---
app = Flask(__name__)

# --- Caminhos dos Artefactos ---
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_hand_sign.pkl')

# --- Variáveis Globais ---
model = None
encoder = None
scaler = None

def load_artifacts():
    """
    Carrega o modelo treinado, o scaler e o encoder.
    Deve ser executado no início da aplicação.
    """
    global model, encoder, scaler
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
            
        model = joblib.load(MODEL_PATH)
        logging.info(f"Modelo carregado: {type(model).__name__}")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logging.info("Scaler carregado.")
        else:
            logging.warning("Scaler não encontrado. A precisão pode ser afetada.")

        if os.path.exists(ENCODER_PATH):
            encoder = joblib.load(ENCODER_PATH)
            logging.info("Label Encoder carregado.")
        else:
            logging.warning("Encoder não encontrado. Serão retornados IDs numéricos.")

    except Exception as e:
        logging.critical(f"ERRO CRÍTICO ao carregar artefactos: {e}")
        # Não impedimos o app de rodar, mas o /predict falhará

def make_landmarks_relative(landmarks_list):
    """
    Replica a lógica do 'create_dataset.py'.
    Torna as coordenadas relativas ao pulso (ponto 0).
    Isto é CRUCIAL para garantir que o input da API corresponde ao input de treino.
    """
    if len(landmarks_list) < 3:
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

# Carregar artefactos no arranque
load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint da Fase 3: Recebe landmarks, processa e classifica.
    """
    if model is None:
        return jsonify({'error': 'Modelo não carregado no servidor.'}), 503

    try:
        # 1. Receber e Validar Dados
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'JSON inválido. Campo "landmarks" obrigatório.'}), 400

        raw_landmarks = data['landmarks']

        if len(raw_landmarks) != 63:
            return jsonify({'error': f'Esperados 63 valores (21 pontos * 3 coords). Recebidos {len(raw_landmarks)}.'}), 400

        # 2. Pré-processamento (Pipeline idêntico ao create_dataset.py)
        # Passo A: Tornar relativo ao pulso
        relative_landmarks = make_landmarks_relative(raw_landmarks)
        
        # Passo B: Formatar para array NumPy (1 amostra, 63 features)
        features = np.array([relative_landmarks])

        # Passo C: Scaling (Pipeline idêntico ao train_model.ipynb)
        if scaler:
            features = scaler.transform(features)

        # 3. Predição
        prediction_idx = model.predict(features)[0]

        # 4. Cálculo de Confiança
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            # Pega a probabilidade da classe vencedora
            probs = model.predict_proba(features)
            confidence = float(np.max(probs))
        
        # 5. Descodificação (Index -> Letra)
        predicted_label = str(prediction_idx)
        if encoder:
            predicted_label = encoder.inverse_transform([prediction_idx])[0]

        # Log para debug (opcional)
        logging.info(f"Input processado -> Predição: {predicted_label} ({confidence:.2f})")

        # [cite_start]6. Resposta JSON [cite: 138]
        return jsonify({
            'letter': predicted_label,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        logging.error(f"Erro durante a predição: {str(e)}")
        return jsonify({'error': 'Erro interno no servidor ao processar predição.'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint simples para verificar se a API está online."""
    return jsonify({'status': 'online', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    # Porta 5000 é o padrão do Flask
    # host='0.0.0.0' permite acesso externo se necessário
    print("API de Deteção de Língua Gestual a iniciar...")
    app.run(host='0.0.0.0', port=5000, debug=False)
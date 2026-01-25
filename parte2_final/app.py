from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

""" Inicialização do servidor """
server = Flask(__name__)
CORS(server)

""" --- 1. CARREGAMENTO DOS ARTEFACTOS (CÉREBRO, TRADUTOR E SCALER) --- 
        Carregar o Modelo (O Cérebro)
        Carregar o Encoder da Mão (O Tradutor de 'Left'/'Right')
        Carregar o Scaler (Normalizador dos dados) """

with open('best_model_asl.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('hand_encoder.pkl', 'rb') as f:
    hand_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)


""" --- 2. DEFINIÇÃO DAS ROTAS --- """

@server.route('/predict', methods=['POST'])
def get_inference():
    try:
        """ Receção dos dados do cliente """
        input_data = request.get_json() 
        
        if not input_data:
            return jsonify({'error': 'Nenhum dado recebido'}), 400
            
        """ Converter para DataFrame (Tabela de 1 linha) """
        df_input = pd.DataFrame([input_data]) 
        
        """ Transforma 'Left'/'Right' em 0 ou 1 """
        if 'hand' in df_input.columns:
            df_input['hand'] = hand_encoder.transform(df_input['hand'])
            
        """ Aplica a mesma matemática usada no treino (StandardScaler) """
        X_final = feature_scaler.transform(df_input)
        
        """ O modelo olha para os dados normalizados e decide a letra """
        predicted_letter = classifier.predict(X_final)[0]
        
        """ Calcular a confiança (0 a 1) """
        probs = classifier.predict_proba(X_final)[0]
        score = float(np.max(probs)) 
        
        return jsonify({
            'letter': str(predicted_letter),
            'confidence': score
        })
        
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@server.route('/status', methods=['GET'])
def check_status():
    """Rota simples para ver se o servidor está vivo"""
    return jsonify({
        'system': 'online', 
        'ready_to_predict': True
    })

if __name__ == '__main__':
    # Arranca o servidor na porta 5000
    print("Servidor a rodar na porta 5000...")
    server.run(host='0.0.0.0', port=5000)
import joblib 
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- 1. Configuração e Carregamento dos Artefactos ---
# Caminhos absolutos ou relativos (garante que a pasta 'models' está ao lado do app.py)
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler_hand_sign.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

print("A carregar artefactos com Joblib...")

try:
    # O joblib.load aceita o caminho do ficheiro diretamente (não precisa de 'with open')
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
        
    print("Artefactos carregados com sucesso!")

except FileNotFoundError:
    print(f"Erro Crítico: Não foi possível encontrar os ficheiros em '{MODEL_PATH}'.")
    print("Certifique-se de que correu o notebook de treino e a pasta 'models' existe.")
    model, scaler, encoder = None, None, None
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    model, scaler, encoder = None, None, None

# --- 2. Rota de Previsão ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not encoder:
        return jsonify({'error': 'Modelos não estão carregados no servidor.'}), 500

    try:
        # A. Receber dados JSON
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Formato inválido. Chave "features" necessária.'}), 400
        
        input_data = data['features']
        
        # B. Pré-processamento
        # Converter para array numpy e redimensionar para 2D (1 linha, N colunas)
        features_array = np.array(input_data).reshape(1, -1)
        
        # Aplicar a normalização (StandardScaler) usada no treino
        scaled_features = scaler.transform(features_array)
        
        # C. Previsão
        prediction_index = model.predict(scaled_features)[0]
        
        # D. Descodificar a Label (Converter número 0, 1... para "A", "B"...)
        # O encoder espera uma lista, por isso passamos [prediction_index]
        prediction_label = encoder.inverse_transform([prediction_index])[0]
        
        # Confiança (se o modelo suportar)
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(scaled_features)
                confidence = float(np.max(probs))
            except:
                confidence = 0.0 # Alguns modelos como SVM linear podem não ter proba habilitado por defeito

        # E. Retornar Resultado
        return jsonify({
            'prediction': prediction_label,
            'confidence': f"{confidence:.2f}",
            'raw_index': int(prediction_index)
        })

    except Exception as e:
        return jsonify({'error': f"Erro na previsão: {str(e)}"}), 500

# --- 3. Executar a App ---
if __name__ == '__main__':
    # host='0.0.0.0' permite acesso de outros dispositivos na mesma rede
    app.run(debug=True, host='0.0.0.0', port=5000)
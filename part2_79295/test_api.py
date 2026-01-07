import requests
import json
import random

# URL onde a tua API está a correr (definido no app.py)
url = 'http://127.0.0.1:5000/predict'

NUM_FEATURES = 63 # 21 landmarks * 3 coordenadas (x, y, z)

# Gerar dados aleatórios para testar se a API aceita o pedido
dummy_features = [random.uniform(0, 1) for _ in range(NUM_FEATURES)]

payload = {
    'features': dummy_features
}

print(f"A enviar {NUM_FEATURES} features para a API...")

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("\nSucesso! Resposta da API:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"\nErro {response.status_code}:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\nErro: Não foi possível conectar. O app.py está a correr no outro terminal?")
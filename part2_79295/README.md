Part2 - ASL Hand Sign Detection

Este diretório contém o código para extrair landmarks de mãos, treinar um modelo de classificação das letras A-Z do dataset SignAlphaSet, expor o modelo via API Flask e executar um cliente em tempo real.

Estrutura principal:
- `create_dataset.py`: percorre `dataset/SignAlphaSet` e extrai landmarks usando `HandLandmarkExtractor`, produzindo `hand_landmarks_dataset.csv` (label + 63 features relativas e normalizadas).
- `hand_landmark_extractor.py`: utilitário para extrair landmarks (MediaPipe) e converter para DataFrame.
- `train_model.ipynb`: notebook com pipeline completo de treino, comparação de modelos, GridSearch leve, avaliação e salvamento de artefactos.
- `app.py`: API Flask que carrega `models/best_model.pkl`, `models/scaler_hand_sign.pkl` e `models/label_encoder.pkl` e fornece `/predict`.
- `client_app.py`: cliente que captura vídeo da câmara, extrai landmarks e envia para a API em tempo real.
- `models/`: pasta onde os artefactos treinados são guardados (`best_model.pkl`, `scaler_hand_sign.pkl`, `label_encoder.pkl`).

Como usar (resumo):
1. Criar dataset de landmarks (se ainda não existir):

```bash
python create_dataset.py
```

2. Abrir e executar o notebook `train_model.ipynb` para treino e afinação. Ou executar o script de treino integrado:

```bash
python -c "import train_script; train_script.run()"  # opcional
```

3. Iniciar a API Flask:

```bash
python app.py
```

4. Abrir o cliente em tempo real (num terminal separado):

```bash
python client_app.py
```

Boas práticas e notas:
- Se algumas letras apresentarem confiança baixa, considerar: a) recolher mais imagens para essa classe; b) aplicar augmentação (rotação, brilho, traslação); c) afinar hiperparâmetros (GridSearch mais extenso) ou usar ensembles; d) inspecionar erros com `confusion_matrix` e analisar exemplos mal classificados.
- As letras J e Z correspondem a gestos com movimento — podem não ser classificadas corretamente com imagens estáticas.

Contacto: documentação e explicações detalhadas no `train_model.ipynb`.

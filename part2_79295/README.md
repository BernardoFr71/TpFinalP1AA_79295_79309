# Trabalho Prático - Parte 2: Deteção de Língua Gestual (ASL)

Este projeto implementa um sistema completo de Machine Learning para a deteção e classificação de caracteres da Língua Gestual Americana (ASL) em tempo real. O sistema abrange desde o processamento de imagem e extração de características até à disponibilização do modelo via API e interface cliente.

## Estrutura do Projeto

O projeto está organizado conforme as 4 fases do desenvolvimento:

* **Fase 1: Dataset & Extração**
    * `hand_landmark_extractor.py`: Classe utilitária baseada no MediaPipe para extração de 21 *landmarks* (x, y, z) das mãos.
    * `create_dataset.py`: Processa as imagens originais e gera o dataset estruturado (`hand_landmarks_dataset.csv`).
    * `augment_landmarks.py`: (Opcional) Gera dados sintéticos para aumentar o dataset e melhorar a robustez do modelo.

* **Fase 2: Treino e Avaliação**
    * `train_model.ipynb`: Notebook Jupyter que contém todo o pipeline de ML: pré-processamento, comparação de modelos (RF, SVM, KNN, MLP), otimização (GridSearch) e avaliação. Gera os ficheiros na pasta `models/`.

* **Fase 3: API Backend**
    * `app.py`: Servidor Flask que carrega o modelo treinado e expõe um endpoint `/predict` para classificação via JSON.

* **Fase 4: Aplicação Cliente**
    * `client_app.py`: Aplicação de visão computacional que captura vídeo da webcam, processa a imagem localmente e consulta a API para obter a classificação em tempo real.

---

## Instalação e Configuração

1.  **Pré-requisitos:** Certifique-se de que tem o Python (3.8+) instalado.
2.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dados:** Coloque a pasta com as imagens originais (dataset *SignAlphaSet*) em:
    `dataset/SignAlphaSet/` (pastas A, B, C... dentro desta diretoria).

---

## 🛠️ Guia de Execução (Passo a Passo)

Siga a ordem abaixo para garantir o funcionamento correto do sistema.

### 1. Criar o Dataset (Fase 1)
Extrai as características geométricas das mãos das imagens originais.
```bash
python create_dataset.py

```

*Saída:* Cria o ficheiro `hand_landmarks_dataset.csv`.

### 2. Aumentar o Dataset (Opcional)

Se desejar melhorar a precisão (especialmente em letras difíceis ou com poucos dados), gere dados sintéticos (rotação, escala e ruído).

```bash
# Exemplo: Adicionar 500 amostras a TODAS as classes
python augment_landmarks.py --n_per_class 500

# Exemplo: Aumentar apenas classes específicas
python augment_landmarks.py --classes M N S --n_per_class 1000

```

*Saída:* Cria o ficheiro `hand_landmarks_augmented.csv`.

### 3. Treinar o Modelo (Fase 2)

Abra o Jupyter Notebook para treinar e selecionar o melhor modelo.

```bash
jupyter notebook train_model.ipynb

```

* Execute todas as células sequencialmente.
* **Importante:** O notebook irá gerar e salvar três ficheiros na pasta `models/`:
1. `best_model.pkl` (O Modelo)
2. `scaler_hand_sign.pkl` (Normalizador)
3. `label_encoder.pkl` (Codificador de etiquetas)



### 4. Iniciar a API (Fase 3)

Inicie o servidor backend. Mantenha este terminal aberto.

```bash
python app.py

```

*O servidor iniciará em `http://127.0.0.1:5000`.*

### 5. Executar o Cliente (Fase 4)

Num **novo terminal**, inicie a aplicação de deteção em tempo real.

```bash
python client_app.py

```

* Pressione **'q'** para sair da aplicação.

---

## 📝 Notas Importantes

* **Letras Dinâmicas (J e Z):** Estas letras envolvem movimento na língua gestual. Como este modelo baseia-se em *frames* estáticos (fotos), a classificação destas letras poderá ser menos precisa.
* **Iluminação e Fundo:** Para melhores resultados na aplicação cliente, utilize um fundo neutro e boa iluminação.
* **Câmara:** O script `test_camera.py` está incluído caso precise de diagnosticar problemas com a webcam sem correr o modelo.

---

## 👤 Autores

Trabalho realizado no âmbito da Unidade Curricular de Aprendizagem Automática.


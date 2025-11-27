Aqui tens um ficheiro `README.md` completo e profissional, formatado em Markdown.

Podes criar um ficheiro chamado **`README.md`** na pasta do teu projeto e colar lá este conteúdo. Ele serve como documentação oficial do teu trabalho, explicando como instalar, configurar e executar cada fase.

-----

````markdown
# Projeto de Aprendizagem Automática: Deteção de Língua Gestual (ASL)

Este projeto implementa um sistema completo de Machine Learning para a deteção e classificação de caracteres da Língua Gestual Americana (ASL) em tempo real. O sistema é composto por um pipeline de extração de características (landmarks das mãos), treino de múltiplos modelos de classificação, uma API REST para servir o modelo e uma aplicação cliente para captura de vídeo.

**Trabalho Prático - Parte 2**

---

## 📋 Índice
1. [Requisitos e Instalação](#1-requisitos-e-instalação)
2. [Estrutura do Projeto](#2-estrutura-do-projeto)
3. [Preparação do Dataset](#3-preparação-do-dataset)
4. [Execução Passo a Passo](#4-execução-passo-a-passo)
    - [Fase 1: Extração de Dados](#fase-1-extração-de-dados-create_datasetpy)
    - [Fase 2: Treino do Modelo](#fase-2-treino-do-modelo-train_modelipynb)
    - [Fase 3: API Flask](#fase-3-api-server-apppy)
    - [Fase 4: Aplicação Cliente](#fase-4-cliente-camera-client_apppy)
5. [Detalhes Técnicos](#5-detalhes-técnicos)

---

## 1. Requisitos e Instalação

### Pré-requisitos
* **Python 3.10** (Obrigatório devido à compatibilidade do MediaPipe).
* **Dataset SignAlphaSet** (Imagens A-Z).

### Instalação das Dependências
Abra o terminal na pasta do projeto e execute o seguinte comando para instalar todas as bibliotecas necessárias:

```bash
py -3.10 -m pip install opencv-python mediapipe pandas numpy scikit-learn flask requests matplotlib seaborn notebook
````

-----

## 2\. Estrutura do Projeto

Certifique-se de que a sua pasta de trabalho (`part2_79295`) segue exatamente esta estrutura:

```text
part2_79295/
│
├── dataset/
│   └── SignAlphaSet/        <-- (Pasta que deve criar)
│       ├── A/               <-- (Colocar aqui as pastas extraídas do zip)
│       ├── B/
│       └── ... (até Z)
│
├── models/                  <-- (Gerada automaticamente na Fase 2)
│   ├── best_model.pkl
│   └── label_encoder.pkl
│
├── hand_landmark_extractor.py  <-- (Classe auxiliar fornecida)
├── create_dataset.py           <-- (Script Fase 1)
├── train_model.ipynb           <-- (Notebook Fase 2)
├── app.py                      <-- (Servidor API Fase 3)
├── client_app.py               <-- (Cliente Webcam Fase 4)
└── README.md
```

-----

## 3\. Preparação do Dataset

Antes de iniciar, é necessário popular a pasta `dataset/SignAlphaSet` com as imagens.

1.  Faça o download do dataset aqui: [Mendeley Data - SignAlphaSet](https://data.mendeley.com/datasets/8fmvr9m98w/1).
2.  Extraia o ficheiro `.zip`.
3.  Copie as pastas **A** a **Z** para dentro de `part2_79295/dataset/SignAlphaSet/`.

-----

## 4\. Execução Passo a Passo

### Fase 1: Extração de Dados (`create_dataset.py`)

**Objetivo:** Processar todas as imagens, extrair os 21 landmarks da mão (x, y, z) usando MediaPipe e salvar num ficheiro CSV.

**Comando:**

```bash
py -3.10 create_dataset.py
```

  * **Input:** Imagens em `dataset/SignAlphaSet/`.
  * **Output:** Ficheiro `hand_landmarks_dataset.csv`.
  * **Nota:** Se o output disser "0 amostras", verifique se as pastas das letras estão no local correto.

### Fase 2: Treino do Modelo (`train_model.ipynb`)

**Objetivo:** Carregar o CSV, treinar diferentes algoritmos (Random Forest, SVM, KNN, MLP), otimizar hiperparâmetros (GridSearch) e salvar o melhor modelo.

**Comando:**
Para abrir o notebook:

```bash
py -3.10 -m notebook train_model.ipynb
```

  * Execute todas as células sequencialmente.
  * O notebook irá comparar a precisão dos modelos e gerar gráficos.
  * **Output:** Cria a pasta `models/` contendo `best_model.pkl` e `label_encoder.pkl`.

### Fase 3: API Server (`app.py`)

**Objetivo:** Iniciar um servidor Flask que expõe o modelo treinado através de um endpoint REST.

**Comando:**

```bash
py -3.10 app.py
```

  * O servidor iniciará em `http://127.0.0.1:5002`.
  * **Mantenha este terminal aberto** enquanto usa o cliente.
  * Rota principal: `POST /predict` (Recebe JSON com landmarks, devolve classificação).

### Fase 4: Cliente Camera (`client_app.py`)

**Objetivo:** Aplicação que usa a webcam para capturar a mão do utilizador, extrair landmarks, enviar para a API e mostrar o resultado.

**Comando (num novo terminal):**

```bash
py -3.10 client_app.py
```

  * Certifique-se que o servidor (Fase 3) está a correr.
  * Aponte a mão para a câmara.
  * Pressione **'q'** para sair.

-----

## 5\. Detalhes Técnicos

### Pipeline de Processamento

1.  **Input:** Frame de vídeo ou Imagem.
2.  **Pré-processamento:** Conversão para RGB e deteção de mão (MediaPipe Hands).
3.  **Extração de Features:** 21 pontos chave (landmarks) normalizados.
4.  **Achatamento (Flatten):** Matriz (21, 3) convertida para vetor de 63 valores.
5.  **Inferência:** O vetor é enviado para o modelo (ex: Random Forest) que devolve a classe (Letra).

### Modelos Testados

  * **Random Forest:** Robusto a ruído, bom desempenho geral.
  * **SVM (Support Vector Machine):** Eficiente em espaços de alta dimensão.
  * **KNN (K-Nearest Neighbors):** Simples, baseado em similaridade.
  * **MLP (Multi-Layer Perceptron):** Rede neuronal para captar relações não-lineares complexas.

### API Specification

**Endpoint:** `/predict`
**Method:** `POST`
**Body (JSON):**

```json
{
  "landmarks": [0.54, 0.32, 0.0, ... ] // Lista de 63 floats
}
```

**Response (JSON):**

```json
{
  "class": "A",
  "confidence": 0.95,
  "code": 200
}
```

-----

**Autor:** [Bernardo Freitas] - [79295]
**Unidade Curricular:** Aprendizagem Automática

```
```
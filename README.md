# TP Final - Aprendizagem Automática (AA)
## Previsão de Preços de Carros Usados

**Autores:** 79295 & 79309  
**Competição:** Kaggle - Car Price Prediction  
**Data:** Novembro 2025

---

## 📋 Descrição do Projeto

Este projeto implementa modelos de regressão avançados para **prever preços de carros usados** utilizando técnicas de Machine Learning com scikit-learn. O trabalho inclui:

- ✅ **7 algoritmos de regressão** com cross-validation e otimização de hiperparâmetros (GridSearchCV)
- ✅ **Feature Engineering** robusto (extração de HP/cilindrada, age, brand/model grouping)
- ✅ **Pré-processamento inteligente** com Pipeline e ColumnTransformer
- ✅ **Ensemble Stacking** para melhorar RMSE
- ✅ **Seeds fixadas** para reprodutibilidade garantida
- ✅ **Execução sem erros** end-to-end

---

## 🏆 Resultados

### Modelos Treinados
1. **Random Forest** - RMSE: **68,534.80** ⭐ (Melhor)
2. **XGBoost** - RMSE: 69,450.23
3. **Linear Regression** - RMSE: 71,204.56
4. **K-Nearest Neighbors** - RMSE: 72,340.12
5. **Decision Tree** - RMSE: 73,890.45
6. **SVR** - RMSE: 75,123.67
7. **MLP (Neural Network)** - RMSE: 76,005.34
8. **Stacking Ensemble** - RMSE: 68,654.70 (não superou RF)

### Score em Kaggle
- **Public Score:** 76,528.925158 (Top ~15% do leaderboard)
- **Dataset:** 188,533 amostras de treino, ~125k de teste

---

## 📁 Estrutura do Projeto

```
TpFinalP1AA_79295_79309/
├── README.md                           # Este ficheiro
├── requirements.txt                    # Dependências Python
├── data/
│   ├── train.csv                      # Dados de treino (188.5k amostras)
│   ├── test.csv                       # Dados de teste (125.7k amostras)
│   └── sample_submission.csv          # Template de submissão
├── notebooks/
│   └── TP_PART1_AA_79295_79309.ipynb # Notebook principal (executable)
├── models/
│   ├── RF_best.joblib                # Melhor modelo (Random Forest)
│   ├── preprocessor.joblib           # ColumnTransformer para preprocessing
│   ├── training_results_summary.csv  # Resumo de todos os modelos
│   └── submission_sample.csv         # Previsões para Kaggle
└── part1_info/
    ├── PART1_RESULTADOS_FINAIS.md    # Análise detalhada de resultados
    ├── PART1_TRANSFORMACOES_APLICADAS.md
    └── PART1_GUIA_RAPIDO.md          # Troubleshooting & FAQ
```

---

## 🚀 Quick Start

### 1. Instalação
```bash
# Clonar repositório
git clone https://github.com/BernardoFr71/TpFinalP1AA_79295_79309.git
cd TpFinalP1AA_79295_79309

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar o Notebook
```bash
# Abrir Jupyter Notebook
jupyter notebook notebooks/TP_PART1_AA_79295_79309.ipynb
```

**Opções de Execução (controladas pela variável `MODE`):**
- `MODE='quick'` → Execução rápida (CV=3, pequenas grids) - 5 minutos ⚡
- `MODE='subset'` → Amostra com grids completas (20k amostras) - 20 minutos ⏱️
- `MODE='full'` → Dataset completo com otimização máxima - 2-4 horas 🏃 (recomendado)

### 3. Gerar Submissão
A submissão é gerada automaticamente pela última célula do notebook:
```
models/submission_sample.csv
```
Pronta para upload no Kaggle!

---

## 🔧 Features Implementadas

### Feature Engineering
| Feature | Descrição |
|---------|-----------|
| `hp` | Horsepower extraída de `engine` |
| `displ_l` | Cilindrada em litros |
| `age` | Idade do veículo (ano atual - modelo) |
| `brand_grp` | Top 20 brands + 'other' |
| `brand_freq` | Frequency encoding de brand |
| `model_grp` | Top 50 modelos + 'other' |
| `model_freq` | Frequency encoding de model |
| `accident_flag` | Flag binária (yes/no → 1/0) |
| `clean_title_flag` | Flag binária de título limpo |

### Pré-processamento
- **Numéricos:** Imputação mediana + StandardScaler
- **Categóricos:** Imputação modo + OneHotEncoder
- **Sanitização:** Infinities e valores extremos (>1e12) → NaN

---

## 📊 Modelos Treinados

Cada modelo foi otimizado com **GridSearchCV** usando **5-fold cross-validation**:

### Linear Regression
```python
params = {'fit_intercept': [True, False]}
```

### K-Nearest Neighbors
```python
params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance']
}
```

### Decision Tree
```python
params = {
    'max_depth': [5, 10, 20, 30, 50, None],
    'min_samples_leaf': [1, 2, 3, 5, 10]
}
```

### Random Forest ⭐
```python
params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 3, 5]
}
```

### XGBoost
```python
params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}
```

### SVR
```python
params = {
    'C': [0.1, 1.0, 10.0, 50.0],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}
```

### MLP (Redes Neuronais)
```python
params = {
    'hidden_layer_sizes': [(50,), (100,), (200,100), (200,150,50)],
    'alpha': [0.0001, 0.001, 0.01]
}
```

---

## 🔬 Técnicas Avançadas

### 1. **Cross-Validation**
- 5-fold KFold com shuffle aleatória
- Scores: RMSE médio ± desvio padrão

### 2. **Stacking Ensemble**
- Base learners: Random Forest + XGBoost + SVR
- Meta-learner: Ridge (α=1.0)
- Resultado: RMSE 68,654.70 (não superou RF single)

### 3. **Log Transformation**
- Target: `log1p(price)` durante treino
- Inversão: `expm1(predictions)` para RMSE final
- Benefício: Estabiliza variância em distribuições skewed

### 4. **Reproducibilidade**
- `RANDOM_STATE=42` fixado globalmente
- `PYTHONHASHSEED`, `np.random.seed`, `random.seed`
- Garantido: Mesmos resultados em execuções repetidas

---

## 📈 Métricas de Avaliação

- **RMSE (Root Mean Squared Error):** Erro médio quadrático (unidades originárias = $)
- **R² Score:** Proporção de variância explicada
- **CV Scores:** Validação cruzada com σ (desvio padrão)

Exemplo de saída:
```
name                rmse      r2       cv_rmse_mean  cv_rmse_std
Random Forest       68534.80  0.8234   70123.45      1245.67
XGBoost            69450.23  0.8189   71034.56      1389.34
...
```

---

## ⚙️ Dependências

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
xgboost>=1.5
matplotlib>=3.4
seaborn>=0.11
```

Instala automaticamente com:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Problema: "OneHotEncoder: unexpected keyword argument 'sparse'"
**Solução:** O código detecta automaticamente a versão do scikit-learn e usa `sparse_output` quando necessário.

### Problema: "File too large" ao fazer git push
**Solução:** Ficheiros `.joblib` não são commitados (veja `.gitignore`). Execute o notebook localmente para gerar modelos.

### Problema: Notebook muito lento
**Solução:** Use `MODE='quick'` ou `MODE='subset'` para testes rápidos. Full mode recomendado apenas com CPU multi-core.

Mais detalhes em: `part1_info/PART1_GUIA_RAPIDO.md`

---

## 📝 Observações Importantes

1. **Seeds Fixadas:** Reprodutibilidade garantida (mesmos resultados sempre)
2. **Sem Erros:** Notebook executa sequencialmente do início ao fim sem exceções
3. **Artifacts Salvos:** Modelos em `.joblib`, preprocessor, e submissão em CSV
4. **Log Transform:** TARGET_LOG=True por padrão (pode ser desativado)
5. **RMSE vs Kaggle:** Hold-out RMSE (68,534.80) vs Public Kaggle (76,528.93) indica distribuição diferente entre treino/teste

---

## 📬 Contacto & Info

- **Alunos:** 79295, 79309
- **Instituição:** Universidade do Algarve
- **Curso:** Aprendizagem Automática
- **Kaggle:** [Ver Submissão](https://www.kaggle.com/competitions/25-26-lesti-u-alg-pt-preco-de-carros-usados/leaderboard)

---

## 📜 Licença

Este projeto é para fins académicos. Sinta-se livre para usar como referência.
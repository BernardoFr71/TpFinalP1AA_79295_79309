# 📚 GUIA DE USO RÁPIDO

## Como Usar o Projeto

### 1. **Arquivo Principal**
```
notebooks/TP1_part1_refactor.ipynb
```
- Notebook completo pronto para execução
- Já foi executado com sucesso
- Modo: **FULL** (máxima otimização)

### 2. **Executar do Zero**

#### Opção A: VS Code / Jupyter
```
cd notebooks/
jupyter notebook TP1_part1_refactor.ipynb
```
Depois pressione "Run All Cells" ou execute célula por célula.

#### Opção B: Python Script
```bash
jupyter nbconvert --to notebook --execute TP1_part1_refactor.ipynb
```

#### Opção C: Dentro do notebook
Quando abrir o notebook, execute cada célula em sequência (Ctrl+Enter em cada célula).

---

### 3. **Outputs Gerados**

Após execução, encontrará:
```
../models/
├── RF_best.joblib              ⭐ Melhor modelo
├── XGB_best.joblib
├── KNN_best.joblib
├── DTree_best.joblib
├── Linear_best.joblib
├── preprocessor.joblib         📊 Pipeline pré-processamento
├── training_results_summary.csv    📈 Resultados detalhados
└── submission_sample.csv           📤 Previsões para Kaggle

../data/
├── train.csv                   (esperado estar aqui)
└── test.csv                    (para gerar submissão)
```

---

### 4. **Resultados**

Ver os resultados no notebook:
1. Abra célula **"Mostrar melhor resultado"** → mostra top 5 modelos
2. Abra célula **"Plot RMSE por modelo"** → visualiza desempenho
3. Abra célula **"Distribuição do target"** → análise EDA
4. Abra célula **"Submissão"** → arquivo pronto para Kaggle

---

### 5. **Submissão Kaggle**

Arquivo pronto: `../models/submission_sample.csv`

**Formato esperado**:
```csv
id,prediction
0,85000
1,92000
...
```

**Uso no Kaggle**:
1. Abra a competição
2. Clique em "Submit Predictions"
3. Upload do arquivo `submission_sample.csv`
4. Submeta!

---

### 6. **Modificar Configuração**

Se quiser customizar (não recomendado pois já está otimizado):

Abra célula **"Execução - parâmetros"** e altere:
```python
MODE = 'full'      # Opções: 'quick' (rápido), 'subset' (amostra), 'full' (completo)
CV_FOLDS = 5       # Número de folds da validação cruzada
TRAIN_N_JOBS = -1  # -1 = todos CPUs; 1 = sequencial (mais lento)
TARGET_LOG = True  # True = modelar log(price); False = price direto
```

---

### 7. **Troubleshooting**

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: xgboost` | Execute célula 2 novamente (instala XGBoost) |
| `FileNotFoundError: ../data/train.csv` | Coloque arquivo train.csv em `../data/` |
| `MemoryError` | Use MODE='quick' ou MODE='subset' |
| `Grid search muito lento` | Use MODE='quick' em vez de 'full' |
| Validação RMSE > 70k | Dados podem estar diferentes; re-execute |

---

### 8. **Principais Features do Notebook**

✅ **Feature Engineering**
- Extração HP, cilindrada da coluna engine
- Limpeza de mileage
- Cálculo de age (anos do veículo)
- Flags de accident/clean_title
- Grouping inteligente de brand/model

✅ **Preprocessing Robusto**
- Imputação de valores missing
- Normalização de features numéricas
- One-hot encoding de categóricas
- Sanitização de infinities

✅ **7 Algoritmos**
- Linear Regression
- KNN (k-Nearest Neighbors)
- Decision Tree
- Random Forest ⭐
- XGBoost
- SVR (Support Vector Regression)
- MLP (Neural Networks)

✅ **GridSearch Completo**
- Validação cruzada (5-fold)
- Múltiplas hiperparâmetros por modelo
- Salva melhor modelo encontrado
- Retorna RMSE, R², best_params

✅ **Reprodutibilidade**
- Seeds fixadas (RANDOM_STATE=42)
- Resultados idênticos em múltiplas execuções
- Logging detalhado

---

### 9. **Performance Esperado**

Tempo de execução:
- **quick mode**: ~5-10 minutos
- **subset mode**: ~20-30 minutos (20k amostras)
- **full mode**: 2-4 HORAS (188k amostras, 7 modelos)

RMSE esperado:
- **Linear**: 69,844
- **RF**: 68,535 ⭐ (melhor)
- **XGB**: 68,772

---

### 10. **Próximos Passos Avançados**

Se quiser melhorar ainda mais:

#### Adicionar Stacking
```python
from sklearn.ensemble import StackingRegressor
# Usar RF, XGB como base learners
# Ridge como meta-learner
```

#### Feature Engineering Adicional
```python
# Polinômios de features
# Interações (HP * displ_l)
# Log transforms de features
```

#### Tuning Fino
```python
# Aumentar n_estimators em RF
# Explorar max_features
# Ajustar learning_rate XGB
```

---

### 11. **Ficheiros de Documentação**

- `RESULTADOS_FINAIS.md` — Sumário executivo dos resultados
- `TRANSFORMACOES_APLICADAS.md` — Detalhes técnicos das correções
- `README.md` — Visão geral do projeto
- `requirements.txt` — Dependências (para pip install)

---

### 12. **Contato / Dúvidas**

Se o notebook não executar:
1. Verificar se `train.csv` existe em `../data/`
2. Verificar Python 3.7+ instalado
3. Instalar: `pip install -r requirements.txt`
4. Re-executar o notebook

---

**Data**: Novembro 20, 2025  
**Status**: ✅ Production Ready  
**Última Atualização**: Transformação completa + execução MODE='full'

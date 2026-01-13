# Trabalho Prático 1 — Previsão de Preços de Carros Usados

**Autores:** Bernardo Freitas (79295) — Afonso Figueiredo (79309)

**Disciplina:** Aprendizagem Automática — Ano Letivo 2025/2026

---

**Resumo:**
- **Objetivo:** Desenvolver e validar um modelo de regressão para prever o preço de carros usados, minimizando o RMSE. O trabalho foca-se numa solução replicável e justificada metodologicamente, privilegiando um modelo XGBoost otimizado com pré-processamento encapsulado por *pipelines* do scikit-learn.
- **Abordagem:** Imputação hierárquica de atributos motor (HP, Litros), engenharia de atributos, codificação de variáveis categóricas, normalização robusta e otimização de hiperparâmetros via validação cruzada.

---

**Conteúdo deste repositório**
- **Enunciado:** [enunciado_TP.pdf](enunciado_TP.pdf)
- **Caderno principal:** [parte1/notebooks/TP1_79295_79309.ipynb](parte1/notebooks/TP1_79295_79309.ipynb)
- **Dados (não incluídos no remoto quando apropriado):** [data/](data)
- **Modelos e artefactos:** [parte1/models/](parte1/models)
- **Scripts auxiliares e ficheiros entregues:** [FicheirosConsulta/](FicheirosConsulta)

---

**Descrição detalhada da metodologia**

- **Pré-processamento:**
  - **Extração de motor:** função `clean_engine` que extrai `HP`, `Liters` e `Cylinders` via expressões regulares a partir da string `engine`.
  - **Normalização de quilometragem:** limpeza de formatos textuais e conversão para numérico, criação de `milage` e `log_milage` (`log1p`).
  - **Idade do veículo:** cálculo a partir de `model_year` (ex.: `age = 2025 - model_year`).
  - **Agrupamentos/flags:** `brand_model` (chave composta), `is_super_luxury` (lista de marcas de luxo), `hp_per_liter` e outras features derivadas.

- **Imputação Hierárquica:** Para `HP` e `Liters` usa-se a ordem de preferência: mediana por `(brand, model)` → mediana por `brand` → mediana global (valores calculados a partir do conjunto de treino). Valores ausentes em `Cylinders` e `milage` são preenchidos pela mediana global do treino.

- **Codificação:** `LabelEncoder` aplicado por coluna combinando `train` e `test` para garantir consistência.

- **Pipeline e normalização:** `ColumnTransformer` + `Pipeline` com `SimpleImputer(strategy='median')` seguido de `RobustScaler` para mitigar outliers.

- **Modelo:** `XGBRegressor` (objective `reg:squarederror`) integrado num `Pipeline` do scikit-learn. Estratégia de treino:
  - **Transformação do alvo:** `y = log1p(price)` para estabilidade e simetria.
  - **Validação:** divisão treino/val 80/20 com `random_state=42` e `KFold` para a RandomizedSearch.
  - **Otimização:** `RandomizedSearchCV` sobre hiperparâmetros do XGBoost (ex.: `learning_rate` baixo 0.007–0.01, `n_estimators` elevado — 4000 em modo `full`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`).

---

**Métricas e análise**
- **Métrica principal:** RMSE sobre preços originais (após `expm1` da predição) — apresentada no notebook principal.
- **Visualizações:** distribuições das previsões, scatter Real vs Previsto (com e sem outliers) e zoom sobre 99% dos dados. Ver secção de Análise Visual no notebook.

---

**Reproduzibilidade — passos para correr localmente**

1. **Criar ambiente Python (recomendado):**

   - Criar virtualenv (Windows Powershell):

     ```powershell
     python -m venv venv_310
     .\venv_310\Scripts\Activate.ps1
     pip install -U pip
     pip install -r requirements.txt
     ```

2. **Abrir e executar o notebook:**

   - Iniciar Jupyter e abrir [parte1/notebooks/TP1_79295_79309.ipynb](parte1/notebooks/TP1_79295_79309.ipynb).

     ```powershell
     jupyter notebook
     ```

   - Executar células na ordem indicada. Para execução completa em modo `full` (otimização extensa) experimente alterar a variável `MODE = 'quick'` para `'full'` conforme documentação no notebook — note que `full` pode demorar muito tempo.

3. **Gerar submissão:** seguir a secção *Submissão* do notebook que gera `submission.csv` a partir do `best_model` treinado.

4. **Guardar/usar o modelo treinado:** o notebook grava o modelo final em `parte1/models/XGB_Final_Optimized.joblib` quando apropriado.

---

**Boas práticas e notas importantes**

- **Ignorar ficheiros pesados:** a raíz do repositório contém um `.gitignore` configurado para evitar envio de ficheiros de dados, modelos e artefactos binários (ex.: `*.joblib`, `*.pkl`, `parte1/data/`, `parte1/models/`).
- **Backup antes de operações destrutivas:** antes de executar `git clean` ou remoções recursivas, faça compressão/backup da pasta (ex.: `Compress-Archive`).
- **Histórico remoto divergente:** caso existam históricos não relacionados entre branches remotas, recomenda-se criar uma branch intermédia (`merge-extra-from-feature`) e abrir Pull Request para revisão, em vez de forçar `push --force` para `main`.

---

**Estrutura de ficheiros (resumo)**
- [README.md](README.md) — este ficheiro.
- [requirements.txt](requirements.txt) — dependências do projecto.
- [enunciado_TP.pdf](enunciado_TP.pdf) — enunciado da Parte 1.
- [parte1/notebooks/TP1_79295_79309.ipynb](parte1/notebooks/TP1_79295_79309.ipynb) — notebook principal com todo o fluxo.
- [parte1/models/](parte1/models) — pasta de modelos (ignoradas pelo `.gitignore` quando apropriado).
- [data/](data) — conjunto de dados (normalmente não comitado ao repositório remoto para respeitar políticas de partilha).

---

Se quiserem, faço também:
- um ficheiro `CONTRIBUTING.md` com instruções de fluxo Git (branches, PR, revisão),
- ou um script `run_train.py` para executar treino de forma não interactiva (batch).

Contacto para dúvidas e revisão: Bernardo Freitas / Afonso Figueiredo.
# TpFinalP1AA_79295_79309

Projeto de Aprendizagem Automática para prever preços de carros usados (Kaggle).

## Estrutura
- data/: Dados do Kaggle.
- notebooks/: Notebooks Jupyter.
- submissions/: Arquivos de submissão.

## Instruções
- Clone o repo.
- Rode o notebook em notebooks/TP1_79295_79309.ipynb.
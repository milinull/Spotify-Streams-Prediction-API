# 🎵 Spotify Streams Prediction API

Projeto Django + Machine Learning para prever a quantidade de streams futuros de músicas baseando-se em dados históricos dos charts diários do Spotify.

---

## 📦 Funcionalidades

- 📈 Previsão de streams futuros de qualquer música no histórico.
- 🔍 Análise de tendências históricas (melhor dia da semana, força de tendência).
- 🔄 Scraping diário dos dados do Spotify via GitHub Actions.
- 🛠️ Treinamento automático de modelo de regressão.
- 🗂️ API REST para acesso fácil às informações.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Django 5.1.7** + **Django REST Framework**
- **Scikit-Learn 1.6.1** (ML)
- **BeautifulSoup 4.13.3** (Scraping)
- **Pandas 2.2.3** e **Numpy 2.2.4** (Dados)
- **GitHub Actions** (Automação diária)
- **Joblib** (Persistência de modelo)

---

## 📋 Como Funciona

### 1. Coleta de Dados

- Script `scripts/scrap_spotify_chart.py` faz scraping diário do ranking global do Spotify.
- Armazena as músicas do dia no banco de dados (modelo `SQLite3`).

### 2. Treinamento do Modelo

- Script `ML/train_spotify_model.py` treina um modelo de regressão baseado nos dados históricos.
- O modelo é salvo em `ML/spotify_streams_model.joblib`.

### 3. APIs Disponíveis

- **Prever streams futuros** (`/predict/`)
- **Analisar tendências históricas** (`/analyze-trends/`)

---

## 🤖 Explicação dos Códigos de Machine Learning e Treinamento

### `ML/ml_predictor.py`

Classe principal: **`StreamsPredictor`**

| Função | O que faz |
|:---|:---|
| `load_or_create_model()` | Carrega o modelo salvo ou cria um novo ensemble de regressão (Gradient Boosting + Random Forest + Ridge). |
| `load_metrics()` | Carrega as métricas do último treinamento salvo em `metrics.json`. |
| `train(spotify_data)` | Treina o modelo com os dados históricos do Spotify armazenados no banco de dados. |
| `predict_future_streams(song_title, artist, days_to_predict=7)` | Faz previsões de streams para os próximos dias para uma música específica. |
| `analyze_song_trends(song_title, artist)` | Analisa o comportamento histórico da música: tendência de crescimento ou queda, melhores dias da semana, projeções futuras simples. |
| `_prepare_training_data(spotify_data)` | Prepara e cria as features de entrada (X) e o target (y) a partir dos dados históricos para o treinamento. |
| `_prepare_prediction_features(song_df)` | Constrói os vetores de features necessários para fazer uma previsão de streams futuros. |
| `_evaluate_prediction_quality(song_df)` | Avalia se uma previsão será confiável (alta, média ou baixa confiança), baseada na estabilidade histórica da música. |
| `_simple_prediction(song_df, days_to_predict)` | Faz uma previsão básica (linear) se não houver dados históricos suficientes para usar o modelo complexo. |
| `_get_metrics_dict()` | Retorna as métricas de avaliação do modelo no formato de dicionário (MAE, RMSE, R²) para facilitar o envio via API. |

---

### Estrutura do Modelo

- **Pipeline**:
  - Padronização: `StandardScaler`
  - Regressão Ensemble: /`VotingRegressor` com
    - `GradientBoostingRegressor`
    - `RandomForestRegressor`
    - `Ridge`

- **Features usadas**:
  - Posição atual, streams anteriores, tendência recente, dia da semana, se é fim de semana, etc.

---

### Previsões

- Se o histórico é robusto ➔ Previsão normal via ML.
- Se histórico é pequeno ➔ Previsão simples baseada em média de variação.
- Correção automática para fins de semana (+5% streams).
- Indicação de confiança da previsão (alta, média ou baixa).

---

## 🧩 Estrutura do Projeto

```
/ML/
    ml_predictor.py     # Classe principal de Machine Learning
    spotify_streams_model.joblib  # Modelo salvo após treino
    train_spotify_model.py # Treinamento do modelo

/scripts/
    scrap_spotify_charts.py # Scraping e importação de dados

/data_csv/
    /original/           # CSVs brutos baixados
    /processed/          # CSVs processados e corrigidos

/.github/workflows/
    daily_scraper.yml    # GitHub Action para scraping automático
```

---

## 🚀 Como Rodar Localmente

1. **Clone o repositório:**

```bash
git clone https://github.com/milinull/Spotify-Streams-Prediction-API
```

2. **Crie o ambiente virtual:**

```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

4. **Ajuste configurações do Django** se necessário (banco de dados, etc).

5. **Execute as migrações:**

```bash
python manage.py migrate
```

6. **Coleta de dados inicial:**

```bash
python scripts/get_spotify_charts.py
```

7. **Treinamento inicial do modelo:**

```bash
python scripts/train_spotify_model.py
```

8. **Rodar o servidor:**

```bash
python manage.py runserver
```

---

## 📚 Endpoints Disponíveis

| Método | Rota             | Descrição                              |
|:------:|:-----------------|:--------------------------------------|
| POST   | `/predict/`       | Faz previsão de streams futuros       |
| POST   | `/analyze-trends/`| Analisa tendências históricas         |
| GET    | `/charts/`        | Lista os charts históricos            |

---

### Exemplo de Request `/predict/`

```json
{
  "title": "Blinding Lights",
  "artist": "The Weeknd",
  "days": 7
}
```

### Exemplo de Request `/analyze-trends/`

```json
{
  "title": "Blinding Lights",
  "artist": "The Weeknd"
}
```

---

## ⚙️ Automação Diária (GitHub Actions)

O scraping e atualização do banco de dados são automáticos através do **GitHub Actions**!

Arquivo de workflow: `.github/workflows/daily_scraper.yml`

### O que ele faz:

- Todo dia às **18:00 BRT** (`21:00 UTC`) o GitHub Actions:
  - Roda o script `scrap_spotify_chart.py`
  - Atualiza os dados no repositório
  - Faz commit automático com a mensagem `"update diário automático"`

### Exemplo do Workflow:

```yaml
name: Daily Spotify Chart Scraper

permissions:
  contents: write

on:
  schedule:
    - cron: '0 21 * * *'
  workflow_dispatch:

jobs:
  run-script:
    runs-on: windows-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run scraping script
      env:
        DJANGO_SETTINGS_MODULE: setup.settings
      run: python scripts\scrap_spotify_chart.py

    - name: Commit and Push
      run: |
        git config --global user.name "github-actions[bot]"
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        git add .
        git commit -m "update diário automático" || echo "Sem alterações para commit"
        git push
```

---

## 📜 Observações Importantes

- O modelo precisa ser re-treinado periodicamente para melhor desempenho.
- Se a música não for encontrada no histórico, a API retorna erro 404.
- A previsão considera padrões semanais para melhorar a acurácia.

---
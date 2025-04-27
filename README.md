# Spotify Streams Predictor API

Uma API Django desenvolvida para coletar, analisar e prever dados de streams de músicas no Spotify utilizando técnicas de Machine Learning.

![Spotify API](https://img.shields.io/badge/API-Spotify-1DB954)
![Django](https://img.shields.io/badge/Framework-Django-092E20)
![Python](https://img.shields.io/badge/Language-Python-3776AB)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-F7931E)

## 📋 Índice

- [Spotify Streams Predictor API](#spotify-streams-predictor-api)
  - [📋 Índice](#-índice)
  - [🔍 Visão Geral](#-visão-geral)
  - [✨ Funcionalidades](#-funcionalidades)
  - [📁 Estrutura do Projeto](#-estrutura-do-projeto)
  - [🛠 Tecnologias Utilizadas](#-tecnologias-utilizadas)
  - [🤖 Modelo de Machine Learning](#-modelo-de-machine-learning)
  - [⚙️ Instalação e Configuração](#️-instalação-e-configuração)
    - [Pré-requisitos](#pré-requisitos)
    - [Passos para Instalação](#passos-para-instalação)
  - [📡 Uso da API](#-uso-da-api)
    - [Endpoints](#endpoints)
    - [Exemplos de Requisições](#exemplos-de-requisições)
      - [Previsão de Streams](#previsão-de-streams)
      - [Análise de Tendências](#análise-de-tendências)
  - [📊 Fluxo de Dados](#-fluxo-de-dados)
  - [📊 Features do Modelo](#-features-do-modelo)
  - [👨‍💻 Manutenção e Atualização](#-manutenção-e-atualização)
    - [Atualização Diária dos Dados](#atualização-diária-dos-dados)
    - [Retreinamento do Modelo](#retreinamento-do-modelo)

## 🔍 Visão Geral

O **Spotify Streams Predictor** é uma API desenvolvida para coletar dados diários das músicas mais ouvidas globalmente no Spotify, processá-los e utilizar algoritmos de machine learning para prever tendências futuras de streams. A aplicação permite analisar o comportamento histórico de músicas específicas e fazer projeções precisas para os próximos dias.

## ✨ Funcionalidades

- **Coleta Automatizada**: Extração automática de dados do Kworb.net com informações diárias do Spotify Charts
- **Processamento de Dados**: Limpeza e estruturação dos dados coletados para análise
- **Visualização de Charts**: API RESTful para consulta das músicas no ranking
- **Previsão de Streams**: Modelo de machine learning para previsão de streams futuros
- **Análise de Tendências**: Detecção de padrões e tendências nas performances das músicas
- **Intervalos de Confiança**: Avaliação da qualidade das previsões com margens de erro

## 📁 Estrutura do Projeto

```
spotify-streams-predictor/
├── api_charts/                  
│   ├── models.py                
│   ├── serializers.py           
│   ├── views.py                                
├── ML/                          
│   ├── ml_predictor.py          # Implementação do modelo de previsão
│   ├── metrics.json             # Métricas de performance do modelo
│   └── spotify_streams_model.joblib  # Modelo treinado serializado
│   └── train_spotify_model.py   # Script para treinamento do modelo
├── scripts/                     
│   ├── scrap_spotify_charts.py    # Script para obtenção dos dados
├── data_csv/                    
│   ├── original/                # CSV originais coletados
│   └── processed/               # CSV processados para uso
├── setup/                       
│   ├── settings.py              
│   ├── urls.py                  
│   └── wsgi.py                  
└── manage.py                    
```

## 🛠 Tecnologias Utilizadas

- **[Django](https://www.djangoproject.com/)**: Framework web
- **[Django REST Framework](https://www.django-rest-framework.org/)**: Framework para APIs REST
- **[Scikit-Learn](https://scikit-learn.org/)**: Biblioteca de machine learning
- **[Pandas](https://pandas.pydata.org/)**: Manipulação e análise de dados
- **[NumPy](https://numpy.org/)**: Computação numérica
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)**: Web scraping
- **[Joblib](https://joblib.readthedocs.io/)**: Serialização de modelos ML
- **[SciPy](https://www.scipy.org/)**: Biblioteca científica para análise estatística

## 🤖 Modelo de Machine Learning

O sistema utiliza um modelo ensemble sofisticado composto por três algoritmos complementares:

1. **Gradient Boosting Regressor**: Para capturar padrões complexos não-lineares
2. **Random Forest Regressor**: Para lidar com diferentes tipos de dados e evitar overfitting
3. **Ridge Regression**: Para estabelecer uma base linear robusta

Os três modelos trabalham em conjunto através de um **VotingRegressor** que combina suas previsões para obter resultados mais precisos e estáveis. O pipeline completo inclui:

- Pré-processamento com StandardScaler para normalização dos dados
- Extração de features temporais (dia da semana, fim de semana)
- Cálculo de médias móveis (3 e 7 dias)
- Detecção de tendências recentes
- Ajustes sazonais para dias da semana

## ⚙️ Instalação e Configuração

### Pré-requisitos

- Python 3.12
- pip (gerenciador de pacotes Python)
- Banco de dados PostgreSQL (recomendado) ou SQLite

### Passos para Instalação

1. **Clone o repositório**
   ```bash
   git clone https://github.com/milinull/Spotify-Streams-Prediction-API.git
   cd spotify-streams-predictor
   ```

2. **Crie e ative um ambiente virtual**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # No Windows
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure o banco de dados**
   ```bash
   python manage.py migrate
   ```

5. **Colete os dados iniciais**
   ```bash
   python scripts/scrap_spotify_charts.py
   ```

6. **Treine o modelo**
   ```bash
   python scripts/train_spotify_model.py
   ```

7. **Inicie o servidor**
   ```bash
   python manage.py runserver
   ```

## 📡 Uso da API

### Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/charts/` | GET | Lista todas as entradas de charts |
| `/api/charts/?search=artista` | GET | Busca por artista ou título |
| `/api/charts/?chart_date=2025-04-20` | GET | Filtra por data específica |
| `/api/charts/?position=1` | GET | Filtra por posição no ranking |
| `/api/predict/` | POST | Faz previsão de streams futuros |
| `/api/analyze-trends/` | POST | Analisa tendências históricas |

### Exemplos de Requisições

#### Previsão de Streams

```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"title": "Cruel Summer", "artist": "Taylor Swift", "days": 7}'
```

**Resposta:**
```json
{
  "current_streams": 3254698,
  "current_date": "2025-04-25",
  "predictions": [
    {
      "date": "2025-04-26",
      "predicted_streams": 3289412,
      "confidence_interval": {
        "lower": 3102456,
        "upper": 3476368
      }
    },
    ...
  ],
  "metrics": {
    "mae": 45863.22,
    "rmse": 62914.58,
    "r2": 0.9432,
    "description": {
      "mae": "Erro Médio Absoluto (menor é melhor)",
      "rmse": "Raiz do Erro Quadrático Médio (menor é melhor)",
      "r2": "Coeficiente de Determinação (mais próximo de 1 é melhor)"
    }
  },
  "prediction_quality": {
    "confidence": "alta",
    "reason": "Streams estáveis ao longo do tempo",
    "trend": "ascendente",
    "variability": {
      "coefficient_of_variation": 0.0812,
      "standard_deviation": 264892
    }
  }
}
```

#### Análise de Tendências

```bash
curl -X POST http://localhost:8000/api/analyze-trends/ \
  -H "Content-Type: application/json" \
  -d '{"title": "Cruel Summer", "artist": "Taylor Swift"}'
```

**Resposta:**
```json
{
  "song_info": {
    "title": "Cruel Summer",
    "artist": "Taylor Swift",
    "days_on_chart": 312,
    "peak_position": 1,
    "peak_streams": 4578932,
    "average_streams": 3245621
  },
  "trend_analysis": {
    "recent_direction": "crescente",
    "trend_strength": 0.87,
    "weekly_pattern": {
      "best_day": "Sábado",
      "worst_day": "Quarta",
      "daily_averages": {
        "Segunda": 3102458,
        "Terça": 2987654,
        "Quarta": 2876543,
        "Quinta": 3056789,
        "Sexta": 3456789,
        "Sábado": 3876543,
        "Domingo": 3654321
      }
    }
  },
  "linear_projection": [
    {
      "date": "2025-04-26",
      "projected_streams": 3315467
    },
    ...
  ]
}
```

## 📊 Fluxo de Dados

O sistema opera através do seguinte fluxo:

1. **Coleta**: O script `scrap_spotify_charts.py` extrai dados diários do Kworb.net
2. **Processamento**: Os dados são limpos, transformados e normalizados
3. **Armazenamento**: As informações são salvas no banco de dados
4. **Treinamento**: O modelo é treinado periodicamente com os dados acumulados
5. **Previsão**: Quando solicitado, o modelo faz previsões baseadas nos padrões aprendidos
6. **Análise**: Métricas e estatísticas são calculadas para avaliar a qualidade das previsões

## 📊 Features do Modelo

As principais features utilizadas pelo modelo de previsão incluem:

- **Posição atual e anterior no ranking**
- **Quantidade de streams atual e anterior**
- **Dias na parada**
- **Posição de pico**
- **Multiplicador (quando disponível)**
- **Variação de streams entre dias**
- **Média semanal de streams**
- **Variação da média semanal**
- **Média móvel de 3 dias**
- **Dia da semana (0-6)**
- **Flag de fim de semana**
- **Tendência recente (inclinação da curva)**

## 👨‍💻 Manutenção e Atualização

### Atualização Diária dos Dados

Para manter o banco de dados atualizado, configure um cronjob ou task scheduler para executar:

```bash
python scripts/scrap_spotify_charts.py
```

### Retreinamento do Modelo

Recomenda-se retreinar o modelo periodicamente para incorporar novos dados:

```bash
python scripts/train_spotify_model.py
```
# Spotify Streams Predictor API

Uma API Django desenvolvida para coletar, analisar e prever dados de streams de músicas no Spotify utilizando técnicas de Machine Learning avançadas com arquitetura modular.

![Spotify API](https://img.shields.io/badge/API-Spotify-1DB954)
![Django](https://img.shields.io/badge/Framework-Django-092E20)
![Python](https://img.shields.io/badge/Language-Python-3776AB)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-F7931E)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57)

## 📋 Índice

- [Spotify Streams Predictor API](#spotify-streams-predictor-api)
  - [📋 Índice](#-índice)
  - [🔍 Visão Geral](#-visão-geral)
  - [✨ Funcionalidades](#-funcionalidades)
  - [🏗️ Arquitetura Modular](#️-arquitetura-modular)
    - [Classes Principais](#classes-principais)
  - [📁 Estrutura do Projeto](#-estrutura-do-projeto)
  - [🛠 Tecnologias Utilizadas](#-tecnologias-utilizadas)
    - [Core Framework](#core-framework)
    - [Machine Learning Stack](#machine-learning-stack)
    - [Utilitários](#utilitários)
  - [🤖 Sistema de Machine Learning](#-sistema-de-machine-learning)
    - [Modelo Ensemble Avançado](#modelo-ensemble-avançado)
    - [Pipeline Completo](#pipeline-completo)
    - [Features Inteligentes (13 dimensões)](#features-inteligentes-13-dimensões)
  - [⚙️ Instalação e Configuração](#️-instalação-e-configuração)
    - [Pré-requisitos](#pré-requisitos)
    - [Passos para Instalação](#passos-para-instalação)
  - [📡 Uso da API](#-uso-da-api)
    - [Endpoints Disponíveis](#endpoints-disponíveis)
    - [Exemplos de Requisições](#exemplos-de-requisições)
      - [🎯 Previsão de Streams](#-previsão-de-streams)
      - [📊 Análise de Tendências](#-análise-de-tendências)
      - [📈 Métricas do Modelo](#-métricas-do-modelo)
  - [📊 Features e Análises](#-features-e-análises)
    - [Engenharia de Features Automatizada](#engenharia-de-features-automatizada)
    - [Análises Estatísticas Avançadas](#análises-estatísticas-avançadas)
    - [Ajustes Contextuais](#ajustes-contextuais)
  - [🔧 Manutenção](#-manutenção)
    - [Atualização Diária dos Dados](#atualização-diária-dos-dados)
    - [Retreinamento do Modelo](#retreinamento-do-modelo)

## 🔍 Visão Geral

O **Spotify Streams Predictor** é uma API desenvolvida para coletar dados diários das músicas mais ouvidas globalmente no Spotify, processá-los e utilizar algoritmos de machine learning para prever tendências futuras de streams. A aplicação permite analisar o comportamento histórico de músicas específicas e fazer projeções precisas para os próximos dias.

## ✨ Funcionalidades

- **🔄 Coleta Automatizada**: Extração automática de dados do Kworb.net
- **🧠 IA Avançada**: Sistema modular com múltiplos algoritmos de ML
- **📈 Previsões Inteligentes**: Previsões com intervalos de confiança e ajustes contextuais
- **📊 Análise Completa**: Tendências, padrões semanais e projeções lineares
- **🎯 Alta Precisão**: Sistema ensemble com validação cruzada
- **📋 API RESTful**: Endpoints organizados e documentados
- **💾 Armazenamento Local**: Banco SQLite3 para desenvolvimento ágil

## 🏗️ Arquitetura Modular

O sistema foi completamente estruturado com arquitetura orientada a objetos:

### Classes Principais

- **`FeatureEngine`**: Centraliza toda engenharia de features
  - Features temporais (dia da semana, fim de semana)
  - Rolling features (médias móveis 3d/7d, tendências)
  - Features de posição e variação
  - Preparação automatizada para ML

- **`ModelManager`**: Gerencia operações do modelo ML
  - Criação de modelos ensemble
  - Carregamento/salvamento automático
  - Treinamento com validação
  - Métricas de performance

- **`StreamsAnalyzer`**: Análises estatísticas especializadas
  - Análise de tendências recentes
  - Padrões semanais detalhados
  - Projeções lineares
  - Avaliação de qualidade

- **`StreamsPredictor`**: Classe principal unificada
  - Interface simplificada
  - Integração com Django
  - Previsões contextuais
  - Fallbacks inteligentes

## 📁 Estrutura do Projeto

```
spotify-streams-predictor/
├── api_charts/                  
│   ├── models.py               # Modelo SpotifyChart
│   ├── serializers.py          # Serializers DRF
│   ├── views.py                # Views com integração ML
├── ML/                          
│   ├── ml_predictor.py         # Sistema ML modular completo
│   ├── metrics.json            # Métricas salvas automaticamente
│   └── spotify_streams_model.joblib  # Modelo ensemble serializado
├── scripts/                     
│   ├── scrap_spotify_charts.py  # Coleta de dados
│   └── train_spotify_model.py   # Script de treinamento
├── setup/                       
│   ├── settings.py             # Configurações Django
│   └── urls.py                 # URLs principais
├── db.sqlite3                   # Banco SQLite3
└── manage.py                   
```

## 🛠 Tecnologias Utilizadas

### Core Framework
- **[Django 5.x](https://www.djangoproject.com/)**: Framework web robusto
- **[Django REST Framework](https://www.django-rest-framework.org/)**: APIs RESTful
- **[SQLite3](https://sqlite.org/)**: Banco de dados integrado

### Machine Learning Stack
- **[Scikit-Learn](https://scikit-learn.org/)**: Algoritmos ML e pipelines
- **[Pandas](https://pandas.pydata.org/)**: Manipulação de dados
- **[NumPy](https://numpy.org/)**: Computação numérica
- **[SciPy](https://www.scipy.org/)**: Análises estatísticas avançadas
- **[Joblib](https://joblib.readthedocs.io/)**: Serialização otimizada

### Utilitários
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)**: Web scraping
- **[Django-Filter](https://django-filter.readthedocs.io/)**: Filtros avançados

## 🤖 Sistema de Machine Learning

### Modelo Ensemble Avançado

O sistema utiliza um **VotingRegressor** combinando três algoritmos especializados:

1. **Gradient Boosting Regressor**
   - `n_estimators=150, learning_rate=0.05`
   - `max_depth=5, subsample=0.8`
   - Especializado em padrões não-lineares complexos

2. **Random Forest Regressor**
   - `n_estimators=200, max_depth=10`
   - `min_samples_leaf=2`
   - Robusto contra overfitting

3. **Ridge Regression**
   - `alpha=1.0`
   - Base linear estável

### Pipeline Completo
```python
Pipeline([
    ('scaler', StandardScaler()),      # Normalização
    ('regressor', VotingRegressor)     # Modelo ensemble
])
```

### Features Inteligentes (13 dimensões)

- **Posicionais**: Posição atual/anterior, variação
- **Temporais**: Streams atual/anterior, diferencial
- **Sazonais**: Dias no chart, posição de pico
- **Técnicos**: Multiplicador, médias móveis 3d/7d
- **Contextuais**: Dia da semana, fim de semana
- **Tendência**: Slope de 3 dias

## ⚙️ Instalação e Configuração

### Pré-requisitos

- Python 3.12+
- pip (gerenciador de pacotes)

### Passos para Instalação

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/spotify-streams-predictor.git
   cd spotify-streams-predictor
   ```

2. **Crie ambiente virtual**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Instale dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure banco SQLite**
   ```bash
   python manage.py migrate
   ```

5. **Colete dados iniciais**
   ```bash
   python scripts/scrap_spotify_charts.py
   ```

6. **Treine o modelo**
   ```bash
   python train_spotify_model.py
   ```

7. **Inicie o servidor**
   ```bash
   python manage.py runserver
   ```

## 📡 Uso da API

### Endpoints Disponíveis

| Endpoint | Método | Descrição | Parâmetros |
|----------|--------|-----------|------------|
| `/charts/` | GET | Lista charts com filtros | `search`, `chart_date`, `position` |
| `/predict/` | POST | Previsão de streams | `title`, `artist`, `days` |
| `/analyze-trends/` | POST | Análise de tendências | `title`, `artist` |
| `/simple-return/` | POST | Dados históricos simples | `title`, `artist` |
| `/model-metrics/` | GET | Métricas do modelo atual | - |

### Exemplos de Requisições

#### 🎯 Previsão de Streams

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Anti-Hero", 
    "artist": "Taylor Swift", 
    "days": 5
  }'
```

**Resposta Completa:**
```json
{
  "current_streams": 4521387,
  "current_date": "2024-05-24",
  "predictions": [
    {
      "date": "2024-05-25",
      "predicted_streams": 4587234,
      "confidence_interval": {
        "lower": 4234567,
        "upper": 4939901
      }
    }
  ],
  "metrics": {
    "mae": 52341.22,
    "rmse": 71829.45,
    "r2": 0.9387,
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
      "coefficient_of_variation": 0.0723,
      "standard_deviation": 326547
    }
  }
}
```

#### 📊 Análise de Tendências

```bash
curl -X POST http://localhost:8000/analyze-trends/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Flowers", 
    "artist": "Miley Cyrus"
  }'
```

**Resposta:**
```json
{
  "song_info": {
    "title": "Flowers",
    "artist": "Miley Cyrus"
  },
  "song_stats": {
    "days_on_chart": 156,
    "peak_position": 1,
    "peak_streams": 5234567,
    "average_streams": 3876543
  },
  "trend_analysis": {
    "direction": "decrescente",
    "strength": 0.67
  },
  "weekly_patterns": {
    "best_day": "Sábado",
    "worst_day": "Terça",
    "daily_averages": {
      "Segunda": 3654321,
      "Terça": 3234567,
      // ... outros dias
    }
  },
  "linear_projection": [
    {
      "date": "2024-05-25",
      "projected_streams": 3825647
    }
    // ... próximos 7 dias
  ]
}
```

#### 📈 Métricas do Modelo

```bash
curl http://localhost:8000/model-metrics/
```

**Resposta:**
```json
{
  "model_metrics": {
    "mae": 45863.22,
    "rmse": 62914.58,
    "r2": 0.9432,
    "description": {
      "mae": "Erro Médio Absoluto (menor é melhor)",
      "rmse": "Raiz do Erro Quadrático Médio (menor é melhor)",
      "r2": "Coeficiente de Determinação (mais próximo de 1 é melhor)"
    }
  },
  "model_status": "trained"
}
```

## 📊 Features e Análises

### Engenharia de Features Automatizada

O `FeatureEngine` calcula automaticamente:

- **Temporais**: Dia da semana, fim de semana
- **Rolling**: Médias móveis 3d/7d, diferenças
- **Posicionais**: Variações, dias desde pico
- **Trends**: Tendências de curto prazo

### Análises Estatísticas Avançadas

O `StreamsAnalyzer` oferece:

- **Tendências Recentes**: Regressão linear dos últimos dados
- **Padrões Semanais**: Performance por dia da semana
- **Projeções**: Extrapolação baseada em tendências
- **Qualidade**: Avaliação da confiabilidade das previsões

### Ajustes Contextuais

O sistema aplica automaticamente:

- **Boost de Fim de Semana**: +5% para sábados/domingos
- **Proteção contra Quedas**: Limita quedas bruscas
- **Intervalos de Confiança**: Aumentam com o horizonte temporal
- **Fallbacks**: Previsões simples para dados insuficientes

## 🔧 Manutenção

### Atualização Diária dos Dados

Para manter o banco de dados atualizado, configure um cronjob ou task scheduler para executar:

```bash
python scripts/scrap_spotify_charts.py
```

### Retreinamento do Modelo
```bash
python ML/train_spotify_model.py
```

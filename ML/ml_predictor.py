import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from django.apps import apps


class FeatureEngine:
    """
    Centraliza a engenharia de features para análise e predição de dados dos charts do Spotify.

    Esta classe oferece métodos para calcular características temporais, tendências,
    comportamento de posição e preparação de dados para modelos de machine learning.
    """
    
    @staticmethod
    def calculate_temporal_features(df):
        """
        Calcula features temporais com base na data do chart.

        Adiciona colunas com o dia da semana (0 = segunda, 6 = domingo)
        e um indicador binário de fim de semana.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo a coluna 'chart_date'.

        Retorna:
            pd.DataFrame: DataFrame com novas colunas 'day_of_week' e 'is_weekend'.
        """

        df = df.copy()
        df['chart_date'] = pd.to_datetime(df['chart_date'])
        df['day_of_week'] = df['chart_date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        return df
    
    @staticmethod
    def calculate_rolling_features(df, streams_col='streams'):
        """
        Calcula médias móveis e tendências de streams.

        Gera colunas de média móvel de 3 e 7 dias, além de uma tendência
        com base na variação média dos streams nos últimos 3 dias.

        Parâmetros:
            df (pd.DataFrame): DataFrame contendo dados de streams.
            streams_col (str): Nome da coluna de streams a ser utilizada.

        Retorna:
            pd.DataFrame: DataFrame com colunas 'rolling_3d', 'rolling_7d' e 'trend_3d'.
        """

        df = df.copy()
        df['rolling_3d'] = df[streams_col].rolling(window=3, min_periods=1).mean()
        df['rolling_7d'] = df[streams_col].rolling(window=7, min_periods=1).mean()
        df['trend_3d'] = df[streams_col].diff().rolling(window=3, min_periods=1).mean().fillna(0)
        return df
    
    @staticmethod
    def calculate_position_features(df):
        """
        Calcula features relacionadas à posição da música no chart.

        Adiciona colunas com a variação diária de posição e um contador
        acumulado de dias desde a primeira aparição de cada música.

        Parâmetros:
            df (pd.DataFrame): DataFrame com coluna 'position'.

        Retorna:
            pd.DataFrame: DataFrame com colunas 'position_change' e 'days_since_peak'.
        """

        df = df.copy()
        df['position_change'] = df['position'].diff().fillna(0)
        df['days_since_peak'] = df.groupby(['title', 'artist']).cumcount()
        return df
    
    @staticmethod
    def prepare_ml_features(df):
        """
        Prepara vetores de features e targets para treinamento de modelos de machine learning.

        Para cada música, calcula features agregadas por dia, como posição, streams,
        médias móveis, tendência e dados temporais. Cria um vetor de entrada por dia,
        e o valor alvo é o número de streams do respectivo dia.

        Parâmetros:
            df (pd.DataFrame): DataFrame com dados históricos das músicas.

        Retorna:
            Tuple[np.ndarray, np.ndarray]: Vetores de features e targets (streams).
        """

        features = []
        targets = []
        
        # Agrupar por música
        df['song_id'] = df['title'] + '-' + df['artist']
        
        for song_id in df['song_id'].unique():
            song_data = df[df['song_id'] == song_id].sort_values('chart_date')
            
            if len(song_data) <= 1:
                continue
                
            # Calcular todas as features
            song_data = FeatureEngine.calculate_temporal_features(song_data)
            song_data = FeatureEngine.calculate_rolling_features(song_data)
            song_data = FeatureEngine.calculate_position_features(song_data)
            
            # Criar vectors de features para cada dia (exceto o primeiro)
            for i in range(1, len(song_data)):
                current = song_data.iloc[i]
                previous = song_data.iloc[i-1]
                
                # Garantir que não há valores NaN
                current_position = float(current.get('position', 0))
                previous_position = float(previous.get('position', 0))
                previous_streams = float(previous.get('streams', 0))
                current_streams = float(current.get('streams', 0))
                days = float(current.get('days', i)) if pd.notna(current.get('days')) else float(i)
                peak = float(current.get('peak', current_position)) if pd.notna(current.get('peak')) else current_position
                multiplier = float(current.get('multiplier', 0)) if pd.notna(current.get('multiplier')) else 0.0
                
                # Features calculadas com proteção contra NaN
                rolling_7d = float(current.get('rolling_7d', current_streams)) if pd.notna(current.get('rolling_7d')) else current_streams
                rolling_3d = float(current.get('rolling_3d', current_streams)) if pd.notna(current.get('rolling_3d')) else current_streams
                trend_3d = float(current.get('trend_3d', 0)) if pd.notna(current.get('trend_3d')) else 0.0
                
                streams_diff = current_streams - previous_streams
                day_of_week = float(current.get('day_of_week', 0))
                is_weekend = float(current.get('is_weekend', 0))
                
                feature_vector = [
                    current_position,
                    previous_position,
                    previous_streams,
                    days,
                    peak,
                    multiplier,
                    streams_diff,
                    rolling_7d,
                    rolling_7d - rolling_3d,
                    rolling_3d,
                    day_of_week,
                    is_weekend,
                    trend_3d
                ]
                
                # Verificar se há NaN no vetor final
                if not any(pd.isna(val) or np.isinf(val) for val in feature_vector):
                    features.append(feature_vector)
                    targets.append(float(current_streams))
        
        return np.array(features), np.array(targets)
    
    @staticmethod
    def create_prediction_features(song_data, future_date):
        """
        Gera o vetor de features mais recente para realizar uma predição de streams futuros.

        Utiliza o histórico da música para calcular todas as features disponíveis e
        adiciona informações temporais com base na data futura.

        Parâmetros:
            song_data (pd.DataFrame): Histórico de dados da música.
            future_date (str ou datetime): Data futura para a qual se deseja predizer.

        Retorna:
            np.ndarray ou None: Vetor de features com uma única amostra para predição,
            ou None se os dados forem insuficientes.
        """

        if len(song_data) == 0:
            return None
            
        # Preparar dados
        song_data = FeatureEngine.calculate_temporal_features(song_data)
        song_data = FeatureEngine.calculate_rolling_features(song_data)
        song_data = FeatureEngine.calculate_position_features(song_data)
        
        current = song_data.iloc[-1]
        previous = song_data.iloc[-2] if len(song_data) > 1 else current
        
        # Features temporais da data futura
        future_dt = pd.to_datetime(future_date)
        day_of_week = future_dt.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        feature_vector = [
            current['position'],
            previous['position'], 
            current['streams'],
            current.get('days', len(song_data)),
            current.get('peak', current['position']),
            current.get('multiplier', 0) or 0,
            current['streams'] - previous['streams'] if len(song_data) > 1 else 0,
            current['rolling_7d'],
            current['rolling_7d'] - current['rolling_3d'],
            current['rolling_3d'],
            day_of_week,
            is_weekend,
            current['trend_3d']
        ]
        
        return np.array([feature_vector])


class ModelManager:
    """Gerencia todas as operações relacionadas ao modelo de Machine Learning,
    incluindo criação, treinamento, salvamento, carregamento, predição e avaliação.
    """
    
    def __init__(self, model_path='spotify_streams_model.joblib'):
        """
        Inicializa a classe ModelManager.

        Args:
            model_path (str): Caminho para salvar ou carregar o modelo.
        """
        self.model_path = model_path
        self.model = None
        self.metrics = {'mae': None, 'rmse': None, 'r2': None}
    
    def create_model(self):
        """
        Cria um modelo ensemble composto por três algoritmos:
        Gradient Boosting, Random Forest e Ridge Regression, combinados via VotingRegressor.

        Returns:
            sklearn.pipeline.Pipeline: Pipeline com escalador + ensemble regressivo.
        """
        models = [
            ('gb', GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=5, 
                subsample=0.8, random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42
            )),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        ensemble = VotingRegressor(estimators=models)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ensemble)
        ])
        
        return self.model
    
    def load_model(self):
        """
        Tenta carregar o modelo salvo do disco.

        Returns:
            bool: True se o modelo for carregado com sucesso, False caso contrário.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Modelo carregado com sucesso!")
                return True
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
        return False
    
    def save_model(self):
        """
        Salva o modelo atual no caminho definido.
        """
        if self.model:
            joblib.dump(self.model, self.model_path)
    
    def train(self, X, y):
        """
        Treina o modelo com os dados fornecidos e calcula métricas de avaliação.

        Divide os dados em 80% treino e 20% teste, realiza o treinamento
        e salva tanto o modelo quanto as métricas.

        Args:
            X (np.array): Vetores de features.
            y (np.array): Valores alvo (streams).

        Returns:
            dict: Informações sobre o treinamento, incluindo métricas e tamanhos dos conjuntos.
        """
        if self.model is None:
            self.create_model()
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.save_model()
        self._save_metrics()
        
        return {
            "metrics": self.get_metrics_dict(),
            "training_size": len(X_train),
            "testing_size": len(X_test)
        }
    
    def predict(self, X):
        """
        Realiza predições usando o modelo treinado.

        Args:
            X (np.array): Vetores de features para predição.

        Returns:
            np.array: Valores preditos.

        Raises:
            ValueError: Se o modelo ainda não foi treinado.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)
    
    def _save_metrics(self):
        """
        Salva as métricas do modelo treinado em um arquivo JSON.
        """
        metrics_path = os.path.join('Metrics', 'metrics.json')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_metrics(self):
        """
        Carrega métricas salvas previamente do arquivo JSON.
        """
        metrics_path = 'ML/Metrics/metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
    
    def get_metrics_dict(self):
        """
        Retorna as métricas de avaliação do modelo formatadas e explicadas.

        Returns:
            dict: Dicionário com métricas e descrições.
        """
        return {
            "mae": round(float(self.metrics['mae']), 2) if self.metrics['mae'] else 0,
            "rmse": round(float(self.metrics['rmse']), 2) if self.metrics['rmse'] else 0,
            "r2": round(float(self.metrics['r2']), 4) if self.metrics['r2'] else 0,
            "description": {
                "mae": "Erro Médio Absoluto (menor é melhor)",
                "rmse": "Raiz do Erro Quadrático Médio (menor é melhor)",
                "r2": "Coeficiente de Determinação (mais próximo de 1 é melhor)"
            }
        }


class StreamsAnalyzer:
    """Realiza análises estatísticas e projeções relacionadas ao desempenho de uma música nas paradas.

    Essa classe oferece métodos para identificar padrões, tendências e realizar previsões baseadas nos
    dados históricos de streams.
    """
    
    @staticmethod
    def analyze_trends(song_data):
        """
        Analisa tendências gerais de desempenho com base nos dados históricos da música.

        Executa cálculo de estatísticas básicas, prepara os dados temporais e realiza uma projeção linear
        simples com base nos últimos dias.

        Args:
            song_data (pd.DataFrame): DataFrame com o histórico da música.

        Returns:
            dict: Dicionário contendo estatísticas da música e projeções futuras.
        """
        if len(song_data) < 3:
            return {"error": "Dados insuficientes para análise"}
        
        df = song_data.copy()
        df = FeatureEngine.calculate_temporal_features(df)
        
        # Estatísticas básicas
        stats_basic = {
            "days_on_chart": len(df),
            "peak_position": int(df['position'].min()),
            "peak_streams": int(df['streams'].max()),
            "average_streams": int(df['streams'].mean())
        }
        
        # Análise de tendência (últimos 7 dias)
        recent_df = df.tail(min(7, len(df)))
#        trend_analysis = StreamsAnalyzer._analyze_recent_trend(recent_df)
        
        # Padrões semanais
#        weekly_patterns = StreamsAnalyzer._analyze_weekly_patterns(df)
        
        # Projeção linear
        linear_projection = StreamsAnalyzer._calculate_linear_projection(recent_df)
        
        return {
            "song_stats": stats_basic,
#            "trend_analysis": trend_analysis,
#            "weekly_patterns": weekly_patterns,
            "linear_projection": linear_projection
        }
    
    # @staticmethod
    # def _analyze_recent_trend(recent_df):
    #     """
    #     Analisa a direção e força da tendência recente de streams.

    #     Realiza uma regressão linear nos últimos dias para identificar se os streams estão
    #     aumentando, diminuindo ou estáveis, e qual a força dessa tendência.

    #     Args:
    #         recent_df (pd.DataFrame): Subconjunto mais recente dos dados.

    #     Returns:
    #         dict: Direção ('crescente', 'decrescente', 'estável') e força (float de 0 a 1).
    #     """
    #     if len(recent_df) < 3:
    #         return {"direction": "indeterminado", "strength": 0.0}
        
    #     x = np.arange(len(recent_df))
    #     y = recent_df['streams'].values
    #     slope, _, r_value, _, _ = stats.linregress(x, y)
        
    #     direction = "crescente" if slope > 0 else "decrescente" if slope < 0 else "estável"
    #     strength = abs(r_value)
        
    #     return {
    #         "direction": direction,
    #         "strength": round(float(strength), 2)
    #     }

    # @staticmethod
    # def _analyze_weekly_patterns(df):
    #     """
    #     Identifica padrões semanais de streams, como melhores e piores dias da semana.

    #     Agrupa os dados por dia da semana e calcula as médias de streams,
    #     retornando o dia mais forte e o mais fraco.

    #     Args:
    #         df (pd.DataFrame): DataFrame com colunas temporais e de streams.

    #     Returns:
    #         dict: Melhor dia, pior dia e médias por dia da semana.
    #     """
    #     if len(df) < 7:
    #         return {"best_day": "indeterminado", "worst_day": "indeterminado", "daily_averages": {}}
        
    #     day_avg = df.groupby('day_of_week')['streams'].mean()
    #     days_map = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 
    #                4: "Sexta", 5: "Sábado", 6: "Domingo"}
        
    #     daily_averages = {days_map[day]: int(avg) for day, avg in day_avg.items()}
    #     best_day = days_map[day_avg.idxmax()]
    #     worst_day = days_map[day_avg.idxmin()]
        
    #     return {
    #         "best_day": best_day,
    #         "worst_day": worst_day,
    #         "daily_averages": daily_averages
    #     }

    @staticmethod
    def _calculate_linear_projection(recent_df, days=7):
        """
        Realiza uma projeção linear simples para prever o número de streams futuros.

        Utiliza regressão linear com base nos últimos dias e projeta valores para os próximos `days`.

        Args:
            recent_df (pd.DataFrame): Subconjunto mais recente dos dados históricos.
            days (int): Número de dias futuros para projetar.

        Returns:
            list: Lista de dicionários com datas e valores projetados de streams.
        """
        if len(recent_df) < 3:
            return []
        
        x = np.arange(len(recent_df))
        y = recent_df['streams'].values
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        projections = []
        last_date = recent_df['chart_date'].iloc[-1]
        
        for i in range(1, days + 1):
            future_date = last_date + timedelta(days=i)
            projected_streams = max(0, int(intercept + slope * (len(recent_df) + i - 1)))
            
            projections.append({
                "date": future_date.strftime('%Y-%m-%d'),
                "projected_streams": projected_streams
            })
        
        return projections


class StreamsPredictor:
    """
    Classe responsável por treinar e utilizar modelos de machine learning
    para prever streams futuros de músicas no Spotify, com base em dados históricos.
    
    Funcionalidades:
    - Treinamento de modelo com dados históricos
    - Predição de streams futuros com modelo ML ou abordagem simples
    - Análise estatística de tendências de músicas
    - Retorno de dados históricos recentes
    """
    
    def __init__(self, model_path='spotify_streams_model.joblib'):
        """
        Inicializa a classe, carregando ou criando o modelo de predição.

        Parâmetros:
        - model_path (str): Caminho para o arquivo do modelo salvo.
        """

        self.model_manager = ModelManager(model_path)
        self.model_manager.load_model() or self.model_manager.create_model()
        self.model_manager.load_metrics()
    
    def train(self, spotify_data):
        """
        Treina o modelo de predição com os dados históricos fornecidos.

        Parâmetros:
        - spotify_data (dict): Dicionário com os dados históricos da música.

        Retorna:
        - dict: Resultado do treinamento ou mensagem de erro.
        """

        df = pd.DataFrame(list(spotify_data.values()))
        
        if df.empty:
            return {"error": "Dados insuficientes para treinamento"}
        
        # Preparar features usando o FeatureEngine
        X, y = FeatureEngine.prepare_ml_features(df)
        
        if len(X) == 0:
            return {"error": "Não foi possível gerar features válidas"}
        
        return self.model_manager.train(X, y)
    
    def predict_future_streams(self, song_title, artist, days_to_predict=7):
        """
        Realiza a predição de streams futuros para uma música específica.

        Parâmetros:
        - song_title (str): Título da música.
        - artist (str): Nome do artista.
        - days_to_predict (int): Número de dias futuros a prever (default: 7).

        Retorna:
        - dict: Predições futuras, métricas e qualidade da previsão, ou mensagens de erro/alerta.
        """

        # Buscar dados da música
        song_data = self._get_song_data(song_title, artist)
        if isinstance(song_data, dict) and "error" in song_data:
            return song_data
        
        # Verificar se modelo está treinado
        if self.model_manager.model is None:
            return {
                "error": "Modelo não treinado. Execute o treinamento primeiro.",
                "instructions": "Execute 'python train_spotify_model.py' para treinar o modelo."
            }
        
        # Se poucos dados, usar predição simples
        if len(song_data) < 3:
            return {
                "warning": "Poucos dados históricos para previsão precisa",
                "predictions": self._simple_prediction(song_data, days_to_predict),
                "confidence": "baixa",
                "metrics": self.model_manager.get_metrics_dict()
            }
        
        # Fazer predições ML
        predictions = self._ml_predictions(song_data, days_to_predict)
        quality = self._evaluate_prediction_quality(song_data)
        
        return {
            "current_streams": int(song_data['streams'].iloc[-1]),
            "current_date": song_data['chart_date'].iloc[-1].strftime('%Y-%m-%d'),
            "predictions": predictions,
            "metrics": self.model_manager.get_metrics_dict(),
            "prediction_quality": quality
        }
    
    def analyze_song_trends(self, song_title, artist):
        """
        Analisa tendências históricas da música em termos estatísticos.

        Parâmetros:
        - song_title (str): Título da música.
        - artist (str): Nome do artista.

        Retorna:
        - dict: Análise estatística da evolução dos streams da música.
        """

        song_data = self._get_song_data(song_title, artist)
        if isinstance(song_data, dict) and "error" in song_data:
            return song_data
        
        analysis = StreamsAnalyzer.analyze_trends(song_data)
        analysis["song_info"] = {"title": song_title, "artist": artist}
        
        return analysis
    
    def simple_return(self, song_title, artist):
        """
        Retorna os dados históricos recentes da música, limitando aos últimos 7 dias.

        Parâmetros:
        - song_title (str): Título da música.
        - artist (str): Nome do artista.

        Retorna:
        - list: Lista de dicionários contendo data e streams.
        """
                
        song_data = self._get_song_data(song_title, artist)
        if isinstance(song_data, dict) and "error" in song_data:
            return []
        
        recent_data = song_data.tail(min(7, len(song_data)))
        return [
            {
                "date": row['chart_date'].strftime('%Y-%m-%d'),
                "streams": row['streams']
            }
            for _, row in recent_data.iterrows()
        ]
    
    def _get_song_data(self, song_title, artist):
        """
        Recupera os dados históricos de uma música específica a partir do banco de dados.

        Parâmetros:
        - song_title (str): Título da música.
        - artist (str): Nome do artista.

        Retorna:
        - DataFrame: Dados da música ordenados por data, ou dicionário com erro.
        """

        SpotifyChart = apps.get_model('api_charts', 'SpotifyChart')
        
        song_data = SpotifyChart.objects.filter(
            title=song_title, artist=artist
        ).order_by('chart_date')
        
        if not song_data.exists():
            return {"error": "Música não encontrada no histórico"}
        
        df = pd.DataFrame(list(song_data.values()))
        df['chart_date'] = pd.to_datetime(df['chart_date'])
        
        return df
    
    def _ml_predictions(self, song_data, days_to_predict):
        """
        Gera predições de streams usando o modelo de machine learning.

        Parâmetros:
        - song_data (DataFrame): Dados históricos da música.
        - days_to_predict (int): Quantidade de dias a prever.

        Retorna:
        - list: Lista de predições com data, streams e intervalo de confiança.
        """

        predictions = []
        current_data = song_data.copy()
        
        for i in range(days_to_predict):
            # Data futura
            last_date = current_data['chart_date'].iloc[-1]
            next_date = last_date + timedelta(days=1)
            
            # Criar features para predição
            features = FeatureEngine.create_prediction_features(current_data, next_date)
            if features is None:
                break
            
            # Fazer predição
            predicted_streams = max(0, int(self.model_manager.predict(features)[0]))
            
            # Ajustes contextuais
            predicted_streams = self._apply_contextual_adjustments(
                predicted_streams, next_date, current_data
            )
            
            # Calcular intervalo de confiança
            confidence_interval = self._calculate_confidence_interval(
                predicted_streams, i
            )
            
            prediction = {
                "date": next_date.strftime('%Y-%m-%d'),
                "predicted_streams": predicted_streams
            }
            
            if confidence_interval:
                prediction["confidence_interval"] = confidence_interval
            
            predictions.append(prediction)
            
            # Atualizar dados para próxima iteração
            current_data = self._update_data_for_next_iteration(
                current_data, next_date, predicted_streams
            )
        
        return predictions
    
    def _apply_contextual_adjustments(self, predicted_streams, date, song_data):
        """
        Aplica ajustes contextuais às predições (como fim de semana ou suavização).

        Parâmetros:
        - predicted_streams (int): Valor previsto de streams.
        - date (datetime): Data da predição.
        - song_data (DataFrame): Dados históricos da música.

        Retorna:
        - int: Valor ajustado da predição.
        """

        # Ajuste de fim de semana
        if date.weekday() >= 5:  # Sábado ou Domingo
            predicted_streams = int(predicted_streams * 1.05)
        
        # Evitar quedas bruscas no primeiro dia
        if len(song_data) > 0:
            last_streams = song_data['streams'].iloc[-1]
            if predicted_streams < last_streams * 0.7:
                predicted_streams = int(max(predicted_streams, last_streams * 0.7))
        
        return predicted_streams
    
    def _calculate_confidence_interval(self, predicted_streams, day_offset):
        """
        Calcula o intervalo de confiança para uma predição de streams.

        Parâmetros:
        - predicted_streams (int): Valor previsto.
        - day_offset (int): Quantos dias no futuro está a predição.

        Retorna:
        - dict | None: Intervalo inferior e superior, ou None se não aplicável.
        """

        if self.model_manager.metrics['rmse'] is None:
            return None
        
        # Aumentar incerteza com o tempo
        uncertainty_factor = 1 + (0.08 * day_offset)
        margin = self.model_manager.metrics['rmse'] * 1.96 * uncertainty_factor
        
        return {
            "lower": max(0, int(predicted_streams - margin)),
            "upper": int(predicted_streams + margin)
        }
    
    def _update_data_for_next_iteration(self, current_data, next_date, predicted_streams):
        """
        Atualiza os dados históricos com a predição mais recente para alimentar a próxima iteração.

        Parâmetros:
        - current_data (DataFrame): Histórico atual.
        - next_date (datetime): Próxima data a ser adicionada.
        - predicted_streams (int): Streams previstos para a nova data.

        Retorna:
        - DataFrame: Novo histórico com a linha predita adicionada.
        """

        # Criar nova linha com predição
        new_row = current_data.iloc[-1].copy()
        new_row['chart_date'] = next_date
        new_row['streams'] = predicted_streams
        new_row['days'] = new_row.get('days', len(current_data)) + 1
        
        # Adicionar ao dataframe
        new_df = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return new_df
    
    def _simple_prediction(self, song_data, days_to_predict):
        """
        Realiza predições simples com base em tendência linear dos dados históricos.

        Parâmetros:
        - song_data (DataFrame): Dados históricos da música.
        - days_to_predict (int): Número de dias a prever.

        Retorna:
        - list: Lista de predições por dia com streams previstos.
        """

        if len(song_data) < 2:
            # Apenas um ponto - manter valor constante
            last_streams = song_data['streams'].iloc[-1]
            last_date = song_data['chart_date'].iloc[-1]
            
            predictions = []
            for i in range(days_to_predict):
                next_date = last_date + timedelta(days=i+1)
                predictions.append({
                    "date": next_date.strftime('%Y-%m-%d'),
                    "predicted_streams": int(last_streams)
                })
            return predictions
        
        # Calcular tendência média
        song_data['streams_diff'] = song_data['streams'].diff()
        avg_change = song_data['streams_diff'].dropna().mean()
        
        last_streams = song_data['streams'].iloc[-1]
        last_date = song_data['chart_date'].iloc[-1]
        
        predictions = []
        for i in range(days_to_predict):
            next_date = last_date + timedelta(days=i+1)
            predicted_streams = max(0, int(last_streams + avg_change * (i + 1)))
            predictions.append({
                "date": next_date.strftime('%Y-%m-%d'),
                "predicted_streams": predicted_streams
            })
        
        return predictions
    
    def _evaluate_prediction_quality(self, song_data):
        """
        Avalia a qualidade das predições com base em variabilidade dos dados e métricas do modelo.

        Parâmetros:
        - song_data (DataFrame): Dados históricos da música.

        Retorna:
        - dict: Avaliação da confiança, motivo, tendência e variabilidade.
        """
        
        if len(song_data) < 5:
            return {
                "confidence": "baixa",
                "reason": "Poucos dados históricos disponíveis",
                "trend": "indeterminado"
            }
        
        # Calcular variabilidade
        cv = song_data['streams'].std() / song_data['streams'].mean()
        
        # Determinar tendência recente
        recent = song_data.tail(5)
        changes = recent['streams'].diff().dropna()
        positive_changes = sum(1 for x in changes if x > 0)
        
        if positive_changes >= len(changes) * 0.7:
            trend = "ascendente"
        elif positive_changes <= len(changes) * 0.3:
            trend = "descendente"
        else:
            trend = "estável"
        
        # Determinar confiança
        if cv < 0.1:
            confidence = "alta"
            reason = "Streams estáveis ao longo do tempo"
        elif cv > 0.3:
            confidence = "baixa" 
            reason = "Alta variabilidade nos streams"
        else:
            confidence = "média"
            reason = "Baseado nos dados históricos disponíveis"
        
        # Ajustar com base no R²
        if self.model_manager.metrics['r2'] and self.model_manager.metrics['r2'] > 0.8:
            confidence = "alta" if confidence != "baixa" else "média"
        elif self.model_manager.metrics['r2'] and self.model_manager.metrics['r2'] < 0.5:
            confidence = "baixa" if confidence != "alta" else "média"
        
        return {
            "confidence": confidence,
            "reason": reason,
            "trend": trend,
            "variability": {
                "coefficient_of_variation": round(float(cv), 4) if not np.isnan(cv) else None,
                "standard_deviation": int(song_data['streams'].std()) if not np.isnan(song_data['streams'].std()) else None
            }
        }
    

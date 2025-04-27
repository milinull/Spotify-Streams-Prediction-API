# ML/ml_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib
import os
import json

class StreamsPredictor:
    def __init__(self, model_path='spotify_streams_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.load_or_create_model()
        self.metrics = {
            'mae': None,
            'rmse': None,
            'r2': None
        }
        self.load_metrics()

    def load_metrics(self):
        """Carrega métricas do arquivo JSON em scripts/metrics.json, se existir"""
        metrics_path = os.path.join('ML', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
                #print("Métricas carregadas com sucesso:", self.metrics)
        else:
            #print("Arquivo metrics.json não encontrado no diretório scripts.")
            pass
        
    def load_or_create_model(self):
        """Carrega o modelo existente ou cria um novo com arquitetura melhorada"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Modelo carregado com sucesso!")
                return True
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
        
        # Se não conseguiu carregar, cria um novo modelo
        print("Criando novo modelo")
        
        # Criar um ensemble de modelos
        models = [
            ('gb', GradientBoostingRegressor(
                n_estimators=150, 
                learning_rate=0.05, 
                max_depth=5, 
                subsample=0.8, 
                random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_leaf=2,
                random_state=42
            )),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        # Criar um modelo de votação
        ensemble = VotingRegressor(estimators=models)
        
        # Criar pipeline completa
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ensemble)
        ])
        
        return False 

    def analyze_song_trends(self, song_title, artist):
        """Analisa tendências históricas da música para melhorar previsões"""
        from django.apps import apps
        SpotifyChart = apps.get_model('api_charts', 'SpotifyChart')
        
        # Buscar histórico da música
        song_data = SpotifyChart.objects.filter(
            title=song_title, 
            artist=artist
        ).order_by('chart_date')
        
        if not song_data.exists():
            return {"error": "Música não encontrada no histórico"}
        
        # Converter para DataFrame
        song_df = pd.DataFrame(list(song_data.values()))
        song_df['chart_date'] = pd.to_datetime(song_df['chart_date'])
        
        # Calcular dias na parada
        days_on_chart = len(song_df)
        
        # Calcular estatísticas básicas
        peak_position = song_df['position'].min()  # Menor número = posição mais alta
        peak_streams = song_df['streams'].max()
        avg_streams = song_df['streams'].mean()
        
        # Calcular tendência recente (últimos 7 dias)
        recent_df = song_df.tail(min(7, len(song_df)))
        
        if len(recent_df) >= 3:
            # Ajustar uma linha de tendência
            from scipy import stats
            x = np.arange(len(recent_df))
            y = recent_df['streams'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calcular métricas de tendência
            trend_direction = "crescente" if slope > 0 else "decrescente" if slope < 0 else "estável"
            trend_strength = abs(r_value)  # Força da correlação
            
            # Projeção linear simples para 7 dias
            future_projection = []
            last_date = recent_df['chart_date'].iloc[-1]
            last_streams = recent_df['streams'].iloc[-1]
            
            for i in range(1, 8):
                projected_streams = intercept + slope * (len(recent_df) + i - 1)
                projected_streams = max(0, int(projected_streams))
                future_date = last_date + timedelta(days=i)
                
                future_projection.append({
                    "date": future_date.strftime('%Y-%m-%d'),
                    "projected_streams": projected_streams
                })
        else:
            trend_direction = "indeterminado"
            trend_strength = 0.0
            future_projection = []
        
        # Verificar sazonalidade (dia da semana)
        if len(song_df) >= 7:
            song_df['day_of_week'] = song_df['chart_date'].dt.dayofweek
            day_avg = song_df.groupby('day_of_week')['streams'].mean().to_dict()
            
            # Converter para formato legível
            days_map = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 
                    4: "Sexta", 5: "Sábado", 6: "Domingo"}
            
            weekday_pattern = {days_map[day]: int(avg) for day, avg in day_avg.items()}
            best_day = days_map[max(day_avg.items(), key=lambda x: x[1])[0]]
            worst_day = days_map[min(day_avg.items(), key=lambda x: x[1])[0]]
        else:
            weekday_pattern = {}
            best_day = "indeterminado"
            worst_day = "indeterminado"
        
        return {
            "song_info": {
                "title": song_title,
                "artist": artist,
                "days_on_chart": days_on_chart,
                "peak_position": int(peak_position),
                "peak_streams": int(peak_streams),
                "average_streams": int(avg_streams)
            },
            "trend_analysis": {
                "recent_direction": trend_direction,
                "trend_strength": round(float(trend_strength), 2) if 'trend_strength' in locals() else 0.0,
                "weekly_pattern": {
                    "best_day": best_day,
                    "worst_day": worst_day,
                    "daily_averages": weekday_pattern
                }
            },
            "linear_projection": future_projection
        }

    def train(self, spotify_data):
        """Treina o modelo com dados históricos"""
        # Preparar features (X) e target (y)
        features, targets = self._prepare_training_data(spotify_data)
        
        if len(features) == 0 or len(targets) == 0:
            return {"error": "Dados insuficientes para treinamento"}
            
        # Dividir em treino e teste (80/20)
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Treinar o modelo
        self.model.fit(X_train, y_train)
        
        # Calcular métricas
        y_pred = self.model.predict(X_test)
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)
        self.metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        self.metrics['r2'] = r2_score(y_test, y_pred)
        
        # Salvar o modelo treinado
        joblib.dump(self.model, self.model_path)
            
        return {
            "metrics": self._get_metrics_dict(),
            "training_size": len(X_train),
            "testing_size": len(X_test)
        }
        
    def predict_future_streams(self, song_title, artist, days_to_predict=7):
        """Prever streams para os próximos dias"""
        # Verificar se o modelo já foi treinado
        if not os.path.exists(self.model_path):
            return {
                "error": "Modelo não treinado. Execute o treinamento primeiro.",
                "instructions": "Execute 'python train_spotify_model.py' para treinar o modelo."
            }
        # Carregar métricas, se necessário
        if self.metrics['mae'] is None:
            self.load_metrics()

        from django.apps import apps
        SpotifyChart = apps.get_model('api_charts', 'SpotifyChart')
        
        # Buscar histórico da música
        song_data = SpotifyChart.objects.filter(
            title=song_title, 
            artist=artist
        ).order_by('chart_date')
        
        if not song_data.exists():
            return {"error": "Música não encontrada no histórico"}
        
        # Converter para DataFrame
        song_df = pd.DataFrame(list(song_data.values()))
        
        # Se temos poucos dados, pode ser difícil fazer previsões precisas
        if len(song_df) < 3:
            return {
                "warning": "Poucos dados históricos para previsão precisa", 
                "predictions": self._simple_prediction(song_df, days_to_predict),
                "confidence": "baixa",
                "metrics": self._get_metrics_dict()
            }
        
        # Preparar dados para previsão
        last_date = song_df['chart_date'].max()
        last_streams = song_df.loc[song_df['chart_date'] == last_date, 'streams'].values[0]
        
        predictions = []
        
        # Criar cópia do dataframe para não modificar o original
        prediction_df = song_df.copy()
        
        for i in range(days_to_predict):
            next_date = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=i+1) if isinstance(last_date, str) else last_date + timedelta(days=i+1)
            next_date_str = next_date.strftime('%Y-%m-%d')
            
            # Adicionar informações sazonais
            day_of_week = next_date.weekday()  # 0-6 (Segunda-Domingo)
            is_weekend = 1 if day_of_week >= 5 else 0  # Fim de semana = 1, Dia útil = 0
            
            # Criar nova entrada para predição
            last_row = prediction_df.iloc[-1].copy()
            prev_row = prediction_df.iloc[-2].copy() if len(prediction_df) > 1 else last_row
            
            # Calcular médias móveis atualizadas
            recent_streams = prediction_df['streams'].tail(min(7, len(prediction_df))).values
            rolling_avg_3d = np.mean(recent_streams[-3:]) if len(recent_streams) >= 3 else np.mean(recent_streams)
            rolling_avg_7d = np.mean(recent_streams) if len(recent_streams) > 0 else last_streams
            
            # Calcular tendência recente (últimos 3-5 dias)
            recent_trend = 0
            if len(recent_streams) >= 3:
                # Tendência calculada como inclinação média dos últimos dias
                recent_changes = np.diff(recent_streams[-min(5, len(recent_streams)):])
                recent_trend = np.mean(recent_changes) if len(recent_changes) > 0 else 0
            
            # Preparar features para a predição atual
            feature_vector = [
                last_row['position'],
                prev_row['position'],
                last_row['streams'],  # Atualizar para o valor mais recente
                last_row['days'] + 1,  # Incrementar dias na parada
                last_row['peak'],
                last_row['multiplier'] if pd.notna(last_row['multiplier']) else 0,
                last_row['streams'] - prev_row['streams'] if pd.notna(prev_row['streams']) else 0,  # Calcular mudança real
                rolling_avg_7d,  # Média semanal atualizada
                rolling_avg_7d - rolling_avg_3d if pd.notna(rolling_avg_3d) else 0,  # Tendência semanal
                rolling_avg_3d  # Média móvel de 3 dias
            ]
            
            # Adicionar features adicionais para o modelo
            # Dia da semana (0-6)
            feature_vector.append(day_of_week)
            # É fim de semana?
            feature_vector.append(is_weekend)
            # Tendência recente (inclinação)
            feature_vector.append(recent_trend)
            
            # Fazer previsão
            X_pred = np.array([feature_vector[:10]])  # Usar apenas as features originais para compatibilidade
            predicted_streams = max(0, int(self.model.predict(X_pred)[0]))
            
            # Ajustar previsão com base em insights adicionais
            if is_weekend:
                # Streams tendem a aumentar nos fins de semana
                weekend_factor = 1.05  # Ajuste conforme necessário
                predicted_streams = int(predicted_streams * weekend_factor)
            
            # Considerar a tendência recente para evitar quedas bruscas
            if i == 0 and predicted_streams < last_streams * 0.7:
                # Limitar queda brusca no primeiro dia
                predicted_streams = int(max(predicted_streams, last_streams * 0.7))
            
            # Calcular intervalo de confiança
            confidence_interval = None
            if self.metrics['rmse'] is not None:
                # Aumentar a incerteza conforme prevemos mais no futuro
                uncertainty_factor = 1 + (0.08 * i)  # 8% mais incerteza a cada dia
                margin = self.metrics['rmse'] * 1.96 * uncertainty_factor
                confidence_interval = {
                    "lower": max(0, int(predicted_streams - margin)),
                    "upper": int(predicted_streams + margin)
                }
            
            prediction_data = {
                "date": next_date_str,
                "predicted_streams": predicted_streams
            }
            
            if confidence_interval:
                prediction_data["confidence_interval"] = confidence_interval
                
            predictions.append(prediction_data)
            
            # Adicionar previsão ao dataframe para usar nas próximas iterações
            new_row = last_row.copy()
            new_row['chart_date'] = next_date_str
            new_row['streams'] = predicted_streams
            new_row['days'] = last_row['days'] + 1
            new_row['streams_change'] = predicted_streams - last_row['streams']
            
            # Atualizar posição com base na tendência (simulado)
            if new_row['streams_change'] > 0:
                new_row['position'] = max(1, new_row['position'] - 1)  # Subir no ranking
            elif new_row['streams_change'] < 0:
                new_row['position'] = new_row['position'] + 1  # Descer no ranking
                
            # Atualizar week_streams
            new_row['week_streams'] = rolling_avg_7d
            
            # Adicionar a nova linha ao dataframe
            prediction_df = pd.concat([prediction_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Calcular qualidade da previsão com base em histórico recente
        prediction_quality = self._evaluate_prediction_quality(song_df)
        
        return {
            "current_streams": int(last_streams),
            "current_date": last_date.strftime('%Y-%m-%d') if not isinstance(last_date, str) else last_date,
            "predictions": predictions,
            "metrics": self._get_metrics_dict(),
            "prediction_quality": prediction_quality
        }
 
    def _evaluate_prediction_quality(self, song_df):
        """Avalia a qualidade da previsão com base nos dados históricos"""
        if len(song_df) < 5:
            return {
                "confidence": "baixa",
                "reason": "Poucos dados históricos disponíveis"
            }
            
        # Verifique a estabilidade dos streams
        streams_std = song_df['streams'].std()
        streams_mean = song_df['streams'].mean()
        cv = streams_std / streams_mean if streams_mean > 0 else float('inf')
        
        # Calcular tendência recente
        recent = song_df.tail(5)
        trend = "estável"
        
        if len(recent) >= 3:
            recent_changes = recent['streams'].diff().dropna()
            positive_changes = sum(1 for x in recent_changes if x > 0)
            negative_changes = sum(1 for x in recent_changes if x < 0)
            
            if positive_changes >= len(recent_changes) * 0.7:
                trend = "ascendente"
            elif negative_changes >= len(recent_changes) * 0.7:
                trend = "descendente"
        
        # Determinar confiança
        confidence = "média"
        reason = "Baseado nos dados históricos disponíveis"
        
        if cv < 0.1:
            confidence = "alta"
            reason = "Streams estáveis ao longo do tempo"
        elif cv > 0.3:
            confidence = "baixa"
            reason = "Alta variabilidade nos streams"
            
        # Ajustar confiança com base nas métricas do modelo
        if self.metrics['r2'] is not None:
            if self.metrics['r2'] > 0.8:
                confidence = "alta" if confidence != "baixa" else "média"
            elif self.metrics['r2'] < 0.5:
                confidence = "baixa" if confidence != "alta" else "média"
        
        return {
            "confidence": confidence,
            "reason": reason,
            "trend": trend,
            "variability": {
                "coefficient_of_variation": round(float(cv), 4) if not np.isnan(cv) and not np.isinf(cv) else None,
                "standard_deviation": int(streams_std) if not np.isnan(streams_std) else None
            }
        }
    
    def _get_metrics_dict(self):
        """Retorna as métricas do modelo em formato adequado para JSON"""
        return {
            "mae": round(float(self.metrics['mae']), 2) if self.metrics['mae'] is not None else 0,
            "rmse": round(float(self.metrics['rmse']), 2) if self.metrics['rmse'] is not None else 0,
            "r2": round(float(self.metrics['r2']), 4) if self.metrics['r2'] is not None else 0,
            "description": {
                "mae": "Erro Médio Absoluto (menor é melhor)",
                "rmse": "Raiz do Erro Quadrático Médio (menor é melhor)",
                "r2": "Coeficiente de Determinação (mais próximo de 1 é melhor)"
            }
        }
    
    def _simple_prediction(self, song_df, days_to_predict):
        """Previsão simples baseada na média de variação"""
        if len(song_df) < 2:
            # Se temos apenas um dia, mantemos o mesmo valor
            last_streams = song_df['streams'].iloc[-1]
            last_date = song_df['chart_date'].iloc[-1]
            
            predictions = []
            for i in range(days_to_predict):
                next_date = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=i+1) if isinstance(last_date, str) else last_date + timedelta(days=i+1)
                next_date_str = next_date.strftime('%Y-%m-%d')
                predictions.append({
                    "date": next_date_str, 
                    "predicted_streams": int(last_streams)
                })
            return predictions
        
        # Calcular média de variação diária
        song_df['streams_diff'] = song_df['streams'].diff()
        avg_daily_change = song_df['streams_diff'].dropna().mean()
        
        # Fazer previsões simples
        last_streams = song_df['streams'].iloc[-1]
        last_date = song_df['chart_date'].iloc[-1]
        
        predictions = []
        for i in range(days_to_predict):
            next_date = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=i+1) if isinstance(last_date, str) else last_date + timedelta(days=i+1)
            next_date_str = next_date.strftime('%Y-%m-%d')
            predicted_streams = int(last_streams + avg_daily_change * (i + 1))
            # Garantir que não seja negativo
            predicted_streams = max(0, predicted_streams)
            predictions.append({
                "date": next_date_str,
                "predicted_streams": predicted_streams
            })
        
        return predictions
    
    def _prepare_training_data(self, spotify_data):
        """Prepara dados para treinamento com features adicionais"""
        df = pd.DataFrame(list(spotify_data.values()))
        
        # Agrupar por música (título + artista)
        df['song_id'] = df['title'] + '-' + df['artist']
        
        # Adicionar informações de dia da semana
        df['chart_date'] = pd.to_datetime(df['chart_date'])
        df['day_of_week'] = df['chart_date'].dt.dayofweek  # 0-6 (Segunda-Domingo)
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        features = []
        targets = []
        
        # Para cada música única
        for song_id in df['song_id'].unique():
            song_df = df[df['song_id'] == song_id].sort_values('chart_date')
            
            if len(song_df) <= 1:
                continue  # Pular músicas com apenas um dia de dados
                
            # Calcular tendências e médias móveis
            song_df['rolling_avg_3d'] = song_df['streams'].rolling(window=3).mean().fillna(song_df['streams'])
            song_df['rolling_avg_7d'] = song_df['streams'].rolling(window=7).mean().fillna(song_df['streams'])
            
            # Calcular tendência (diferença das médias)
            song_df['recent_trend'] = song_df['streams'].diff().rolling(window=3).mean().fillna(0)
                
            # Criar features para cada dia com histórico
            for i in range(1, len(song_df)):
                # Features básicas
                row = song_df.iloc[i]
                prev_row = song_df.iloc[i-1]
                
                feature_vector = [
                    row['position'],
                    prev_row['position'],
                    prev_row['streams'],
                    row['days'],
                    row['peak'],
                    row['multiplier'] if pd.notna(row['multiplier']) else 0,
                    row['streams_change'] if pd.notna(row['streams_change']) else 0,
                    row['week_streams'],
                    row['week_streams_change'] if pd.notna(row['week_streams_change']) else 0,
                    row['rolling_avg_3d']
                ]
                
                features.append(feature_vector)
                targets.append(row['streams'])
                
        return np.array(features), np.array(targets)

    def _prepare_prediction_features(self, song_df):
        """Prepara features para previsão"""
        if len(song_df) <= 1:
            # Não temos dados históricos suficientes
            # Criar vetor de features com valores padrão
            return np.array([[
                song_df['position'].iloc[-1],
                song_df['position'].iloc[-1],
                song_df['streams'].iloc[-1],
                song_df['days'].iloc[-1],
                song_df['peak'].iloc[-1],
                song_df['multiplier'].iloc[-1] if pd.notna(song_df['multiplier'].iloc[-1]) else 0,
                0,  # streams_change
                song_df['week_streams'].iloc[-1],
                0,  # week_streams_change
            ]])
        
        # Ordenar por data
        song_df = song_df.sort_values('chart_date')
        
        # Criar features para o último dia disponível
        last_row = song_df.iloc[-1]
        prev_row = song_df.iloc[-2]

        rolling_avg = song_df['streams'].rolling(window=3).mean().iloc[-1]
        
        feature_vector = [
            last_row['position'],
            prev_row['position'],
            prev_row['streams'],
            last_row['days'],
            last_row['peak'],
            last_row['multiplier'] if pd.notna(last_row['multiplier']) else 0,
            last_row['streams_change'] if pd.notna(last_row['streams_change']) else 0,
            last_row['week_streams'],
            last_row['week_streams_change'] if pd.notna(last_row['week_streams_change']) else 0,
            rolling_avg
        ]
        
        return np.array([feature_vector])
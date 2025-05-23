# train_spotify_model.py
import os
import django
import sys
from pathlib import Path
import json

# Configurar o ambiente Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'setup.settings')
django.setup()

from api_charts.models import SpotifyChart
from ML.ml_predictor import StreamsPredictor

def train_model():
    """Treina o modelo com a nova estrutura modularizada"""
    print("Iniciando treinamento do modelo...")
    
    # Buscar todos os dados
    all_data = SpotifyChart.objects.all()
    print(f"Total de registros: {all_data.count()}")
    
    if all_data.count() == 0:
        print("ERRO: Sem dados para treinar!")
        return
    
    # Criar instância do predictor com a nova estrutura
    predictor = StreamsPredictor(model_path='C:/Users/Brian/Desktop/spotify_dataset_v3/ML/spotify_streams_model.joblib')
    
    # Treinar o modelo
    training_result = predictor.train(all_data)
    
    # Verificar se houve erro no treinamento
    if "error" in training_result:
        print(f"ERRO no treinamento: {training_result['error']}")
        return
    
    print("Modelo treinado com sucesso!")
    print("Métricas do modelo:")
    print(f"  MAE (Erro Médio Absoluto): {training_result['metrics']['mae']}")
    print(f"  RMSE (Raiz do Erro Quadrático): {training_result['metrics']['rmse']}")
    print(f"  R² (Coeficiente de Determinação): {training_result['metrics']['r2']}")
    print(f"  Tamanho do conjunto de treino: {training_result['training_size']}")
    print(f"  Tamanho do conjunto de teste: {training_result['testing_size']}")
    
    # As métricas já são salvas automaticamente pela nova estrutura
    print("\nMétricas salvas automaticamente em 'ML/metrics.json'")
    print("Modelo salvo automaticamente em 'ML/spotify_streams_model.joblib'")
    print("\nAgora você pode usar a API de previsão!")

if __name__ == "__main__":
    train_model()
# train_spotify_model.py
import os
import django
import sys
from pathlib import Path
import json

# Configurar o ambiente Django
# Adicione o caminho do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar o Django para usar o arquivo de configurações do projeto
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'setup.settings')
django.setup()

from api_charts.models import SpotifyChart
from ML.ml_predictor import StreamsPredictor

def train_model():
    print("Iniciando treinamento do modelo...")
    
    all_data = SpotifyChart.objects.all()
    print(f"Total de registros: {all_data.count()}")
    
    if all_data.count() == 0:
        print("ERRO: Sem dados para treinar!")
        return
        
    predictor = StreamsPredictor()
    training_result = predictor.train(all_data)
    #print(training_result)
    
    print("Modelo treinado com sucesso!")
    print(f"Métricas:")
    print(f"MAE: {training_result['metrics']['mae']}")
    print(f"RMSE: {training_result['metrics']['rmse']}")
    print(f"R²: {training_result['metrics']['r2']}")

    # Salvar métricas em um arquivo JSON
    metrics_path = Path('metrics.json')
    with metrics_path.open('w') as f:
        json.dump(training_result['metrics'], f)
    
    print("Agora você pode usar a API de previsão!")

if __name__ == "__main__":
    train_model()
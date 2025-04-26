# management/commands/train_streams_model.py
from django.core.management.base import BaseCommand
from api_charts.models import SpotifyChart
from ML.ml_predictor import StreamsPredictor  # Caminho atualizado

class Command(BaseCommand):
    help = 'Treina o modelo de previsão de streams'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Iniciando treinamento do modelo...'))
        
        # Obter todos os dados
        all_data = SpotifyChart.objects.all()
        
        if all_data.count() == 0:
            self.stdout.write(self.style.ERROR('Nenhum dado encontrado para treinar o modelo!'))
            return
            
        predictor = StreamsPredictor()
        training_result = predictor.train(all_data)
        
        if 'error' in training_result:
            self.stdout.write(self.style.ERROR(f"Erro no treinamento: {training_result['error']}"))
            return
            
        # Mostrar métricas
        self.stdout.write(self.style.SUCCESS('Modelo treinado e salvo com sucesso!'))
        self.stdout.write(f"Métricas do modelo:")
        self.stdout.write(f"MAE: {training_result['metrics']['mae']}")
        self.stdout.write(f"RMSE: {training_result['metrics']['rmse']}")
        self.stdout.write(f"R²: {training_result['metrics']['r2']}")
        self.stdout.write(f"Dados de treino: {training_result['training_size']}")
        self.stdout.write(f"Dados de teste: {training_result['testing_size']}")
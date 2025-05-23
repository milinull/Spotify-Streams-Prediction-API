# serializers.py
from rest_framework import serializers
from .models import SpotifyChart

class SpotifyChartSerializer(serializers.ModelSerializer):
    """Serializer para o modelo SpotifyChart"""
    class Meta:
        model = SpotifyChart
        fields = '__all__'

class StreamPredictionRequestSerializer(serializers.Serializer):
    """Serializer para requisições de predição de streams"""
    title = serializers.CharField(
        max_length=255,
        help_text="Título da música"
    )
    artist = serializers.CharField(
        max_length=255,
        help_text="Nome do artista"
    )
    days = serializers.IntegerField(
        min_value=1, 
        max_value=30, 
        default=7,
        help_text="Número de dias para predição (1-30)"
    )

class SimpleReturnSerializer(serializers.Serializer):
    """Serializer para retorno simples de dados históricos"""
    title = serializers.CharField(
        max_length=255,
        help_text="Título da música"
    )
    artist = serializers.CharField(
        max_length=255,
        help_text="Nome do artista"
    )

# Novos serializers para resposta (opcionais, para melhor documentação da API)
class PredictionResponseSerializer(serializers.Serializer):
    """Serializer para resposta de predição"""
    date = serializers.DateField()
    predicted_streams = serializers.IntegerField()
    confidence_interval = serializers.DictField(required=False)

class ModelMetricsResponseSerializer(serializers.Serializer):
    """Serializer para resposta de métricas do modelo"""
    mae = serializers.FloatField(help_text="Erro Médio Absoluto")
    rmse = serializers.FloatField(help_text="Raiz do Erro Quadrático Médio")
    r2 = serializers.FloatField(help_text="Coeficiente de Determinação")
    description = serializers.DictField()

class TrainingResponseSerializer(serializers.Serializer):
    """Serializer para resposta de treinamento"""
    message = serializers.CharField()
    training_result = serializers.DictField()

class TrendAnalysisResponseSerializer(serializers.Serializer):
    """Serializer para resposta de análise de tendências"""
    song_info = serializers.DictField()
    song_stats = serializers.DictField()
    trend_analysis = serializers.DictField()
    weekly_patterns = serializers.DictField()
    linear_projection = serializers.ListField()
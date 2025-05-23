# views.py
from rest_framework import viewsets, filters, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from .models import SpotifyChart
from .serializers import *
from ML.ml_predictor import StreamsPredictor

class SpotifyChartViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet para fornecer endpoints de leitura para Spotify Charts
    """
    queryset = SpotifyChart.objects.all()
    serializer_class = SpotifyChartSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    
    filterset_fields = ['chart_date', 'position']
    search_fields = ['artist', 'title']
    ordering_fields = ['id', 'position', 'streams', 'total_streams']

class StreamPredictionView(APIView):
    """
    API para prever streams futuros de uma música
    """
    
    def post(self, request, format=None):
        serializer = StreamPredictionRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            title = serializer.validated_data['title']
            artist = serializer.validated_data['artist']
            days = serializer.validated_data.get('days', 7)
            
            # Verificar se a música existe
            if not SpotifyChart.objects.filter(title=title, artist=artist).exists():
                return Response(
                    {"error": f"Música '{title}' do artista '{artist}' não encontrada no banco de dados"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Obter previsões usando a nova estrutura
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            predictions = predictor.predict_future_streams(title, artist, days)
            
            return Response(predictions)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class SongTrendAnalysisView(APIView):
    """
    API para analisar tendências históricas de uma música
    """
    
    def post(self, request, format=None):
        serializer = StreamPredictionRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            title = serializer.validated_data['title']
            artist = serializer.validated_data['artist']
            
            # Verificar se a música existe
            if not SpotifyChart.objects.filter(title=title, artist=artist).exists():
                return Response(
                    {"error": f"Música '{title}' do artista '{artist}' não encontrada no banco de dados"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Obter análise de tendências usando a nova estrutura
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            analysis = predictor.analyze_song_trends(title, artist)
            
            return Response(analysis)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class SimpleReturnView(APIView):
    """
    API para retornar dados históricos simples de uma música
    """
    
    def post(self, request, format=None):
        serializer = SimpleReturnSerializer(data=request.data)
        
        if serializer.is_valid():
            title = serializer.validated_data['title']
            artist = serializer.validated_data['artist']
            
            # Verificar se a música existe
            if not SpotifyChart.objects.filter(title=title, artist=artist).exists():
                return Response(
                    {"error": f"Música '{title}' do artista '{artist}' não encontrada no banco de dados"}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Usar a nova estrutura para retornar dados simples
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            musicas = predictor.simple_return(title, artist)
            
            return Response(musicas)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ModelTrainingView(APIView):
    """
    API para treinar o modelo ML (nova funcionalidade)
    """
    
    def post(self, request, format=None):
        try:
            # Buscar todos os dados para treinamento
            all_data = SpotifyChart.objects.all()
            
            if all_data.count() == 0:
                return Response(
                    {"error": "Sem dados suficientes para treinamento"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Treinar o modelo usando a nova estrutura
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            training_result = predictor.train(all_data)
            
            return Response({
                "message": "Modelo treinado com sucesso!",
                "training_result": training_result
            })
            
        except Exception as e:
            return Response(
                {"error": f"Erro durante o treinamento: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ModelMetricsView(APIView):
    """
    API para obter métricas do modelo atual
    """
    
    def get(self, request, format=None):
        try:
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            metrics = predictor.model_manager.get_metrics_dict()
            
            return Response({
                "model_metrics": metrics,
                "model_status": "trained" if predictor.model_manager.model is not None else "not_trained"
            })
            
        except Exception as e:
            return Response(
                {"error": f"Erro ao obter métricas: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
# views.py
from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend

from .models import SpotifyChart
from .serializers import SpotifyChartSerializer, StreamPredictionRequestSerializer
from ML.ml_predictor import StreamsPredictor  # Caminho atualizado

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
            
            # Obter previsões
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
            
            # Obter análise de tendências
            predictor = StreamsPredictor(model_path='ML/spotify_streams_model.joblib')
            analysis = predictor.analyze_song_trends(title, artist)
            
            return Response(analysis)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
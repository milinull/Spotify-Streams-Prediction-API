# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api_charts.views import *

router = DefaultRouter()
router.register(r'charts', SpotifyChartViewSet)

urlpatterns = [
    # Endpoints básicos do DRF
    path('', include(router.urls)),
    
    # Endpoints de ML - Predição e Análise
    path('predict/', StreamPredictionView.as_view(), name='predict-streams'),
    path('analyze-trends/', SongTrendAnalysisView.as_view(), name='analyze-trends'),
    path('simple-return/', SimpleReturnView.as_view(), name='bd-return'),
    
    # Novos endpoints para gerenciamento do modelo
    #path('train-model/', ModelTrainingView.as_view(), name='train-model'),
    path('model-metrics/', ModelMetricsView.as_view(), name='model-metrics'),
]
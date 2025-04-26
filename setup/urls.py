# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api_charts.views import SpotifyChartViewSet, StreamPredictionView, SongTrendAnalysisView

router = DefaultRouter()
router.register(r'charts', SpotifyChartViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('predict/', StreamPredictionView.as_view(), name='predict-streams'),
    path('analyze-trends/', SongTrendAnalysisView.as_view(), name='analyze-trends'),
]
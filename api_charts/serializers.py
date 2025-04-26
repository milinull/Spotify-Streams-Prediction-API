# serializers.py
from rest_framework import serializers
from .models import SpotifyChart

class SpotifyChartSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpotifyChart
        fields = '__all__'

class StreamPredictionRequestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    artist = serializers.CharField(max_length=255)
    days = serializers.IntegerField(min_value=1, max_value=30, default=7)
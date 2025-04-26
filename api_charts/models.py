from django.db import models

class SpotifyChart(models.Model):
    """
    Modelo para armazenar dados dos charts do Spotify
    """
    position = models.IntegerField()
    change = models.CharField(max_length=10)
    artist = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    feat_artist = models.CharField(max_length=255, null=True, blank=True)
    days = models.IntegerField()
    peak = models.IntegerField()
    multiplier = models.IntegerField(null=True, blank=True)
    streams = models.BigIntegerField()
    streams_change = models.IntegerField(null=True, blank=True)
    week_streams = models.BigIntegerField()
    week_streams_change = models.IntegerField(null=True, blank=True)
    total_streams = models.BigIntegerField()
    chart_date = models.DateField()

    def __str__(self):
        if self.feat_artist:
            return f"{self.position}. {self.artist} - {self.title} (w/ {self.feat_artist})"
        return f"{self.position}. {self.artist} - {self.title}"

    class Meta:
        ordering = ['position']
        unique_together = ['position', 'chart_date']
# Generated by Django 5.1.7 on 2025-04-25 20:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SpotifyChart',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('position', models.IntegerField()),
                ('change', models.CharField(max_length=10)),
                ('artist', models.CharField(max_length=255)),
                ('title', models.CharField(max_length=255)),
                ('feat_artist', models.CharField(blank=True, max_length=255, null=True)),
                ('days', models.IntegerField()),
                ('peak', models.IntegerField()),
                ('multiplier', models.IntegerField(blank=True, null=True)),
                ('streams', models.BigIntegerField()),
                ('streams_change', models.IntegerField(blank=True, null=True)),
                ('week_streams', models.BigIntegerField()),
                ('week_streams_change', models.IntegerField(blank=True, null=True)),
                ('total_streams', models.BigIntegerField()),
                ('chart_date', models.DateField()),
            ],
            options={
                'ordering': ['position'],
                'unique_together': {('position', 'chart_date')},
            },
        ),
    ]

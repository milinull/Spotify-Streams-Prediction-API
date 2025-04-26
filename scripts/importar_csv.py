import os
import sys
import django
import csv
from datetime import datetime

# Garante que o Django encontre o projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Ajuste para o caminho do seu projeto
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "setup.settings")  # Nome correto do projeto
django.setup()

from api_charts.models import SpotifyChart

# Caminho do CSV
CSV_FILE_PATH = "../data_csv/processed/spotify_charts_2025-04-24_processed.csv"

# Definir uma data padrão para os dados importados (já que não existe no CSV)
CHART_DATE = datetime.strptime("2025-04-24", "%Y-%m-%d").date()

def limpar_numero(valor):
    """ Remove vírgulas, sinais e converte para inteiro, tratando valores vazios """
    if not valor or not valor.strip():
        return None
    
    valor_limpo = valor.replace(',', '').replace('"', '').replace('+', '')
    
    try:
        return int(valor_limpo)
    except ValueError:
        return None

def extrair_multiplier(valor):
    """ Converte diretamente para inteiro, tratando valores vazios """
    if not valor or not valor.strip():
        return None
    try:
        return int(valor)  # Converte diretamente para inteiro
    except ValueError:
        return None


def importar_csv():
    with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            try:
                multiplier_value = extrair_multiplier(row.get('Multiplier'))
                print(f"Multiplier processado: {row.get('Multiplier')} -> {multiplier_value}")  # Debug
                
                SpotifyChart.objects.update_or_create(
                    position=int(row['Position']),
                    chart_date=CHART_DATE,  # Definindo a data manualmente
                    defaults={
                        'change': row['Change'],
                        'artist': row['Artist'],
                        'title': row['Title'],
                        'feat_artist': row['Feat Artist'] if row['Feat Artist'] and row['Feat Artist'].strip() else None,
                        'days': int(row['Days']),
                        'peak': int(row['Peak']),
                        'multiplier': multiplier_value,
                        'streams': limpar_numero(row['Streams']),
                        'streams_change': limpar_numero(row['Streams Change']),
                        'week_streams': limpar_numero(row['7-Day Streams']),
                        'week_streams_change': limpar_numero(row['7-Day Change']),
                        'total_streams': limpar_numero(row['Total Streams']),
                    }
                )
                count += 1
            except Exception as e:
                print(f"Erro ao importar linha {row}: {e}")
    
    print(f"Importação concluída! {count} registros adicionados ou atualizados.")

if __name__ == "__main__":
    importar_csv()

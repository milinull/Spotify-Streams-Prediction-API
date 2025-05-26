import os
import sys
import django
import csv
import re
import requests
import html
from datetime import datetime
from bs4 import BeautifulSoup

# Configuração do Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "setup.settings")
django.setup()

from api_charts.models import SpotifyChart

def limpar_numero(valor):
    """
    Limpa uma string numérica removendo vírgulas, aspas e sinais de adição,
    convertendo-a em um inteiro. Retorna None se o valor estiver vazio ou 
    não for um número válido.

    Parâmetros:
        valor (str): String representando um número com possíveis símbolos.

    Retorna:
        int ou None: Valor convertido em inteiro ou None se inválido.
    """

    if not valor or not valor.strip():
        return None
    valor_limpo = valor.replace(',', '').replace('"', '').replace('+', '')
    try:
        return int(valor_limpo)
    except ValueError:
        return None

def extrair_multiplier(valor):
    """
    Converte uma string para inteiro, usada para extrair valores de multiplicadores.
    Retorna None se o valor estiver vazio ou não puder ser convertido.

    Parâmetros:
        valor (str): String representando um número inteiro.

    Retorna:
        int ou None: Valor inteiro ou None em caso de erro.
    """

    if not valor or not valor.strip():
        return None
    try:
        return int(valor)
    except ValueError:
        return None

def importar_csv(csv_path):
    """
    Importa dados de um arquivo CSV processado e insere ou atualiza os registros 
    correspondentes em uma tabela do banco de dados (SpotifyChart).

    O nome do arquivo deve conter a data no formato 'YYYY-MM-DD', usada como 
    referência para o campo 'chart_date'.

    Parâmetros:
        csv_path (str): Caminho do arquivo CSV a ser importado.

    Retorna:
        None
    """

    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado!")
        return
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            try:
                multiplier_value = extrair_multiplier(row.get('Multiplier'))

                # Extrai a data do nome do arquivo
                match = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path)
                chart_date = datetime.strptime(match.group(1), "%Y-%m-%d").date() if match else None

                
                SpotifyChart.objects.update_or_create(
                    position=int(row['Position']),
                    chart_date=chart_date,
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

def get_spotify_charts():
    """
    Faz o scraping da tabela diária de rankings globais do Spotify a partir do site 
    'kworb.net', salva o CSV original com os dados brutos e gera uma versão processada 
    com os campos organizados e limpos. Em seguida, importa os dados para o banco.

    Retorna:
        None
    """

    url = "https://kworb.net/spotify/country/global_daily.html"
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = 'utf-8'  # Força o encoding para UTF-8
    soup = BeautifulSoup(response.text, "html.parser", from_encoding="utf-8")
    
    title_text = soup.find("span", class_="pagetitle").text
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", title_text)
    date_str = date_match.group(1) if date_match else "unknown_date"
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_csv_dir = os.path.join(base_dir, "data_csv")
    original_dir = os.path.join(data_csv_dir, "original")
    processed_dir = os.path.join(data_csv_dir, "processed")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    original_filepath = os.path.join(original_dir, f"spotify_charts_{date_str.replace('/', '-')}.csv")
    processed_filepath = os.path.join(processed_dir, f"spotify_charts_{date_str.replace('/', '-')}_processed.csv")
    
    table_rows = soup.find_all("tr")[1:]
    data = []
    
    for row in table_rows:
        columns = row.find_all("td")
        if len(columns) < 10:
            continue
        
        pos = columns[0].text.strip()
        change = columns[1].text.strip()
        artist_title = html.unescape(columns[2].text.strip())
        days = columns[3].text.strip()
        peak = columns[4].text.strip()
        multiplier = columns[5].text.strip()
        streams = columns[6].text.strip().replace(',', '')
        streams_change = columns[7].text.strip()
        week_streams = columns[8].text.strip().replace(',', '')
        week_streams_change = columns[9].text.strip()
        total_streams = columns[10].text.strip().replace(',', '')
        
        data.append([pos, change, artist_title, days, peak, multiplier, 
                     streams, streams_change, week_streams, 
                     week_streams_change, total_streams])
    
    with open(original_filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Position", "Change", "Artist and Title", "Days", 
                         "Peak", "Multiplier", "Streams", "Streams Change", 
                         "7-Day Streams", "7-Day Change", "Total Streams"])
        writer.writerows(data)
    
    process_csv(original_filepath, processed_filepath)
    print(f"Iniciando importação do arquivo processado: {processed_filepath}")
    importar_csv(processed_filepath)

def process_csv(input_filepath, output_filepath):
    """
    Lê um arquivo CSV original com dados brutos dos charts do Spotify,
    processa os campos separando artista, título e artistas convidados,
    extrai o multiplicador corretamente e salva um novo arquivo CSV formatado.

    Parâmetros:
        input_filepath (str): Caminho do arquivo CSV original.
        output_filepath (str): Caminho do arquivo onde o CSV processado será salvo.

    Retorna:
        None
    """

    with open(input_filepath, mode="r", encoding="utf-8") as infile, \
         open(output_filepath, mode="w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        
        new_header = ["Position", "Change", "Artist", "Title", "Feat Artist", "Days",
                      "Peak", "Multiplier", "Streams", "Streams Change", "7-Day Streams",
                      "7-Day Change", "Total Streams"]
        writer.writerow(new_header)
        
        for row in reader:
            if len(row) < 3:
                continue  # Pula linhas incompletas
                
            artist_title = row[2]
            
            # Divide entre artista e título
            parts = artist_title.split(' - ', 1)
            if len(parts) < 2:
                artist, title = parts[0], ""
            else:
                artist, title = parts[0], parts[1]
            
            # Extrai artistas participantes (feat)
            feat_artist = None
            
            # Padrão (w/ Artista)
            w_pattern = re.search(r'\(w/\s*(.*?)\)', title)
            if w_pattern:
                feat_artist = w_pattern.group(1)
                title = re.sub(r'\s*\(w/\s*.*?\)', '', title).strip()
            
            # Padrão (feat. Artista)
            feat_pattern = re.search(r'\(feat\.\s*(.*?)\)', title)
            if not feat_artist and feat_pattern:
                feat_artist = feat_pattern.group(1)
                title = re.sub(r'\s*\(feat\.\s*.*?\)', '', title).strip()
            
            # Padrão (with Artista)
            with_pattern = re.search(r'\(with\s*(.*?)\)', title)
            if not feat_artist and with_pattern:
                feat_artist = with_pattern.group(1)
                title = re.sub(r'\s*\(with\s*.*?\)', '', title).strip()
            
            # Extrai o multiplicador corretamente
            multiplier = row[5]
            if '(x' in multiplier:
                multiplier_match = re.search(r'\(x(\d+)\)', multiplier)
                if multiplier_match:
                    multiplier = multiplier_match.group(1)
            
            # Criar nova linha com as colunas ajustadas
            new_row = [
                row[0],  # Position
                row[1],  # Change
                artist,  # Artist
                title,   # Title
                feat_artist,  # Feat Artist
                row[3],  # Days
                row[4],  # Peak
                multiplier,  # Multiplier
                row[6],  # Streams
                row[7],  # Streams Change
                row[8],  # 7-Day Streams
                row[9],  # 7-Day Change
                row[10]  # Total Streams
            ]
            
            writer.writerow(new_row)
    
    print(f"Arquivo processado salvo em: {output_filepath}")

if __name__ == "__main__":
    get_spotify_charts()
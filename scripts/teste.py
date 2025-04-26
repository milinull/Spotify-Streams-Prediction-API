import requests # type: ignore
import csv
import os
from bs4 import BeautifulSoup # type: ignore
import re
import html

def fix_encoding(text):
    """
    Corrige problemas de codificação, convertendo texto de Latin-1 para UTF-8.
    Útil para lidar com caracteres especiais em títulos de músicas.
    """
    try:
        return text.encode('latin1').decode('utf-8')
    except UnicodeEncodeError:
        return text

def split_artist_title(artist_title):
    """
    Separa o artista principal, o título da música e os artistas do feat.
    """
    match = re.match(r"^(.*?)\s*-\s*(.*?)\s*(\(w/ (.+)\))?$", artist_title)
    
    if match:
        artist = match.group(1).strip()
        title = match.group(2).strip()
        feat_artists = match.group(4).strip() if match.group(4) else None
        return artist, title, feat_artists
    else:
        return artist_title, None, None  # Caso não corresponda ao padrão

def get_spotify_charts():
    """
    Extrai os charts do Spotify, salva o CSV original e gera um CSV processado.
    """
    url = "https://kworb.net/spotify/country/global_daily.html"
    
    # Faz a requisição e processa o HTML
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extrai a data do chart
    title_text = soup.find("span", class_="pagetitle").text
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", title_text)
    date_str = date_match.group(1) if date_match else "unknown_date"
    
    # Define diretórios e nomes de arquivos
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Volta um nível para sair de "scripts/"
    data_csv_dir = os.path.join(base_dir, "data_csv")
    
    original_dir = os.path.join(data_csv_dir, "original")
    processed_dir = os.path.join(data_csv_dir, "processed")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    original_filepath = os.path.join(original_dir, f"spotify_charts_{date_str.replace('/', '-')}.csv")
    processed_filepath = os.path.join(processed_dir, f"spotify_charts_{date_str.replace('/', '-')}_processed.csv")
    
    # Processa dados da tabela de charts
    table_rows = soup.find_all("tr")[1:]  # Pula cabeçalho
    data = []
    
    for row in table_rows:
        columns = row.find_all("td")
        if len(columns) < 10:
            continue  # Pula linhas incompletas
        
        # Extrai e limpa dados de cada coluna
        pos = columns[0].text.strip()
        change = columns[1].text.strip()
        artist_title = fix_encoding(html.unescape(columns[2].text.strip()))
        days = columns[3].text.strip()
        peak = columns[4].text.strip()
        multiplier = columns[5].text.strip()
        streams = columns[6].text.strip().replace(',', '')
        streams_change = columns[7].text.strip()
        week_streams = columns[8].text.strip().replace(',', '')
        week_streams_change = columns[9].text.strip()
        total_streams = columns[10].text.strip().replace(',', '')
        
        # Adiciona linha de dados
        data.append([pos, change, artist_title, days, peak, multiplier, 
                     streams, streams_change, week_streams, 
                     week_streams_change, total_streams])
    
    # Salva dados em arquivo CSV original
    with open(original_filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Position", "Change", "Artist and Title", "Days", 
                         "Peak", "Multiplier", "Streams", "Streams Change", 
                         "7-Day Streams", "7-Day Change", "Total Streams"])
        writer.writerows(data)
    
    print(f"Arquivo original salvo em: {original_filepath}")

    # Processa e salva o CSV formatado
    process_csv(original_filepath, processed_filepath)

def process_csv(input_filepath, output_filepath):
    """
    Lê o arquivo CSV original, separa artista e título, e salva um novo CSV formatado.
    Converte as colunas Multiplier, Streams Change e 7-Day Change para valores numéricos.
    """
    with open(input_filepath, mode="r", encoding="utf-8") as infile, \
         open(output_filepath, mode="w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Lendo cabeçalho original
        header = next(reader)
        
        # Criando novo cabeçalho
        new_header = ["Position", "Change", "Artist", "Title", "Feat Artist", "Days",
                      "Peak", "Multiplier", "Streams", "Streams Change", "7-Day Streams",
                      "7-Day Change", "Total Streams"]
        writer.writerow(new_header)
        
        # Processando cada linha
        for row in reader:
            artist_title = row[2]  # Coluna "Artist and Title"
            artist, title, feat_artist = split_artist_title(artist_title)
            
            # Criar nova linha base com colunas separadas
            new_row = row[:2] + [artist, title, feat_artist] + row[3:]
            
            # Processar o Multiplier (coluna 7 na nova linha)
            if '(x' in new_row[7]:
                multiplier_match = re.search(r'\(x(\d+)\)', new_row[7])
                if multiplier_match:
                    new_row[7] = multiplier_match.group(1)
            
            # Processar Streams Change (coluna 9 na nova linha)
            if new_row[9]:
                new_row[9] = new_row[9].replace('+', '').replace(',', '').replace('"', '')
            
            # Processar 7-Day Change (coluna 11 na nova linha)
            if new_row[11]:
                new_row[11] = new_row[11].replace('+', '').replace(',', '').replace('"', '')
            
            writer.writerow(new_row)
    
    print(f"Arquivo processado salvo em: {output_filepath}")

# Executa script apenas quando rodado diretamente
if __name__ == "__main__":
    get_spotify_charts()

import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import re
import requests

def processar_dados(row):
    id = row['id']
    url = f'https://loft.com.br/imovel/{id}'
    response = requests.get(url)

    if response is not None and response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        encontrar_banheiro = re.compile(r"\d+\s*banheiro[s]?", re.IGNORECASE)
        elemento_banheiro = soup.find(text=encontrar_banheiro)

        num_banheiros = re.search(r"\d+", elemento_banheiro).group()
        print(num_banheiros)

        row['banheiros'] = num_banheiros
    else:
        print(f"Erro: {response.text}")

    return row

# Lê o arquivo CSV com pandas
df = pd.read_csv('imoveis_apartment.csv', encoding='latin-1')

# Cria um pool de threads para executar as requisições em paralelo
with ThreadPoolExecutor() as executor:
    # Mapeia a função processar_dados em cada linha do DataFrame e armazena os resultados em uma lista
    results = executor.map(processar_dados, df.to_dict(orient='records'))

df_result = pd.DataFrame(results)

df_result.to_csv('novo_arquivo.csv', index=False)

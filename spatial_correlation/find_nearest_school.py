import pandas as pd
from geopy.distance import geodesic

locais = pd.read_csv('imoveis_ordenado_lat_lng_atualizado.csv')
escolas = pd.read_csv('ranking_escolas.csv')

locais['escola_proxima'] = ''
locais['ranking_escola'] = ''
locais['distancia_escola'] = ''
locais['renda_media'] = ''

# https://www.zbs.com.br/enem-anteriores
# https://www.sejalguem.com/enem-escolas
# https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem

for index, row in locais.iterrows():
    menor_distancia = float('inf') # float infinito
    escola_mais_proxima = ''
    ranking_escola = 999
    renda_media = 0

    for index_escola, row_escola in escolas.iterrows():
        distancia = geodesic((row['lat'], row['lng']),
                             (row_escola['lat'], row_escola['lng'])).km

        if distancia < menor_distancia:
            menor_distancia = distancia
            escola_mais_proxima = row_escola['Escola']
            ranking_escola = row_escola['Rank Munic. Geral']
            renda_media = row_escola['Renda Média']

    # salva no data frame local
    locais.at[index, 'escola_proxima'] = escola_mais_proxima
    locais.at[index, 'ranking_escola'] = ranking_escola
    locais.at[index, 'distancia_escola'] = round(menor_distancia, 3)
    locais.at[index, 'renda_media'] = renda_media

locais.to_csv('imoveis_escolas_proximas.csv', index=False)

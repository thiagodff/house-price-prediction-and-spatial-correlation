import pandas as pd

# Ler o arquivo CSV
dados_imoveis = pd.read_csv('dataset_imoveis_new.csv')

# Remover as linhas em branco no campo "andar"
# dados_imoveis = dados_imoveis[dados_imoveis['andar'].notna()]
# dados_imoveis.to_csv('novo_arquivo_ord_lat_lng_atualizado.csv', index=False)


# Remove colunas que não serão utilizadas no treinamento
dados_imoveis.drop('lat', inplace=True, axis=1)
dados_imoveis.drop('lng', inplace=True, axis=1)
dados_imoveis.drop('distancia_metro', inplace=True, axis=1)
dados_imoveis.drop('ranking_escola', inplace=True, axis=1)
dados_imoveis.drop('distancia_escola', inplace=True, axis=1)
dados_imoveis.drop('renda_media', inplace=True, axis=1)
dados_imoveis.drop('qtd_crimes_violentos', inplace=True, axis=1)
dados_imoveis.drop('farmacias_proximas', inplace=True, axis=1)
dados_imoveis.drop('supermercados_proximos', inplace=True, axis=1)
dados_imoveis.drop('parques_proximos', inplace=True, axis=1)

dados_imoveis.to_csv('dataset_imoveis_sem_correlacao_espacial_new.csv', index=False)

print(dados_imoveis)

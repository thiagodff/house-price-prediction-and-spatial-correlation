import pandas as pd

# Ler o arquivo CSV
dados_imoveis = pd.read_csv('imoveis_escolas_violencia_farmacia_mercado_shopping_parque.csv')

# Remover as linhas em branco no campo "andar"
# dados_imoveis = dados_imoveis[dados_imoveis['andar'].notna()]
# dados_imoveis.to_csv('novo_arquivo_ord_lat_lng_atualizado.csv', index=False)

# Remove colunas que não serão utilizadas no treinamento
dados_imoveis.drop('id', inplace=True, axis=1)
dados_imoveis.drop('tipo_imovel', inplace=True, axis=1)
dados_imoveis.drop('rua', inplace=True, axis=1)
dados_imoveis.drop('bairro', inplace=True, axis=1)
dados_imoveis.drop('cidade', inplace=True, axis=1)
dados_imoveis.drop('estado', inplace=True, axis=1)
dados_imoveis.drop('criado_em', inplace=True, axis=1)
dados_imoveis.drop('escola_proxima', inplace=True, axis=1)

dados_imoveis.to_csv('dataset_imoveis.csv', index=False)

print(dados_imoveis)

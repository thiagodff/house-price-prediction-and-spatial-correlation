import pandas as pd

# Carregar o arquivo CSV
data = pd.read_csv('dataset_imoveis.csv')

# Remover imoveis com condominio menor que 10 reais e preco maior que 6 milhoes
data = data[(data['condominio'] > 10) & (data['preco'] <= 6000000)]

# Salvar o resultado em um novo arquivo CSV
data.to_csv('dataset_imoveis_new.csv', index=False)

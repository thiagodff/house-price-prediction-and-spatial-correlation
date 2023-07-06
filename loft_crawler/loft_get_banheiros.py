import re
import requests
from bs4 import BeautifulSoup
import csv

input_filename = 'imoveis_apartment.csv'
output_filename = 'imoveis_apartment_.csv'
with open(input_filename, mode='r') as input_file, open(output_filename, mode='w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        id = row[0]

        url = f'https://loft.com.br/imovel/{id}'
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            encontrar_banheiro = re.compile(r"\d+\s*banheiro[s]?", re.IGNORECASE)
            elemento_banheiro = soup.find(text=encontrar_banheiro)

            num_banheiros = re.search(r"\d+", elemento_banheiro).group()
            print(num_banheiros)

            row[6] = num_banheiros

            writer.writerow(row)

        else:
            print(f"Erro: {response.text}")

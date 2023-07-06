import requests
import csv

url = "https://landscape-api.loft.com.br/listing/v2/search"
num_paginas = 2

campos_necessarios = ["id", "price", "complexFee", "area", "floor", "bedrooms", "suits", "parkingSpots", "subwayShortestDistance", "condominiumDoorkeeperAvailability", "homeType", "propertyTax", "restrooms", "address", "amenities", "createdAt"]

itens = []

for pagina in range(1, num_paginas + 1):

    params = {"page": pagina, "cities[]": "belo horizonte, mg", "orderBy[]": "rankB", "hitsPerPage": 20}
    headers = {"Origin": "https://loft.com.br"}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        # Cria um novo array contendo apenas os campos_necessarios
        listings = response.json()['listings']
        itens_pagina = [{campo: item[campo] for campo in campos_necessarios} for item in listings]
        itens += itens_pagina
    else:
        print(f"Erro na página {pagina}: {response.text}")

print(itens)

filename = 'imoveis_teste.csv'

# Abre o arquivo CSV em modo de escrita e escreve os dados
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'preco', 'area', 'salas', 'quartos', 'suites', 'banheiros', 'andar', 'garagens', 'iptu', 'condominio', 'portaria', 'distancia_metro', 'espaco_social', 'piscina', 'academia', 'tipo_imovel', 'rua', 'bairro', 'cidade', 'estado', 'lat', 'lng', 'criado_em'])

    for item in itens:
        tipo_imovel = item['homeType']
        iptu = item['propertyTax']

        if tipo_imovel == "house" or iptu < 20:
            continue

        id = item['id']
        preco = item['price']
        area = item['area']
        salas = item['restrooms']
        quartos = item['bedrooms']
        suites = item['suits']
        andar = item['floor']
        garagens = item['parkingSpots']
        condominio = item['complexFee']
        portaria = 0 if item['condominiumDoorkeeperAvailability'] == "NOT_APPLICABLE" else 1
        distancia_metro = item['subwayShortestDistance']
        amenidades = item['amenities'] if item['amenities'] is not None else []
        piscina = 1 if amenidades.count("pool") > 0 else 0
        academia = 1 if amenidades.count("gym") > 0 else 0
        espaco_social = 1 if (amenidades.count("gourmet") + amenidades.count("party_room") + amenidades.count("play_game") + amenidades.count("grill") + amenidades.count("sports_court")) > 0 else 0

        rua = item['address']['streetFullName']
        bairro = item['address']['neighborhood']
        cidade = item['address']['city']
        estado = item['address']['state']
        lat = item['address']['lat']
        lng = item['address']['lng']
        criado_em = item['createdAt']

        writer.writerow([id, preco, area, salas, quartos, suites, None, andar, garagens, iptu, condominio, portaria, distancia_metro, piscina, espaco_social, academia, tipo_imovel, rua, bairro, cidade, estado, lat, lng, criado_em])

# Unir espaço gourmet, salão de festa, salão de jogos, churrasqueira
# Piscina
# Academia
# remover casas
# remover dados com iptu (ou criar o dado), verificar quantos são no total
# verificar quantos ap tem alguma amenidade e quantos não tem


#{
#    gourmet: "Espaço Gourmet",
#    party_room: "Salão de Festa",
#    play_game: "Salão de Jogos",
#    grill: "Churrasqueira",
#    sports_court: "Quadra Esportiva",
#
#    gym: "Academia",
#
#    heated_pool: "Piscina Aquecida",
#    pool: "Piscina",
#
#    green_area: "",
#    kids: "Brinquedoteca",
#    playground: "Playground",
#}
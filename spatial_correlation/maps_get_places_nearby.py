import pandas as pd
import googlemaps

gmaps = googlemaps.Client(key='API_KEY')

df = pd.read_csv('imoveis_escolas_violencia_farmacia_mercado_shopping.csv')
df['parques_proximos'] = ''

lat_anterior = None
lng_anterior = None
qtd_lugares_anterior = None
qtd_fetch=0

for i, row in df.iterrows():
    lat = row['lat']
    lng = row['lng']

    if lat == lat_anterior and lng == lng_anterior:
        qtd_lugares = qtd_lugares_anterior
    else:
        resultados = gmaps.places_nearby(
            location=(lat, lng),
            radius=500,
            keyword='parque', # farmacia, supermercado, restaurante, shopping, parque
        )
        print(f"fetch api: {i}")
        qtd_fetch = qtd_fetch + 1

        # Obter o número de farmácias encontradas
        qtd_lugares = len(resultados['results'])

        lat_anterior = lat
        lng_anterior = lng
        qtd_lugares_anterior = qtd_lugares

    df.at[i, 'parques_proximos'] = qtd_lugares
    print(f"Endereço: {row['id']} - Número de parques próximos: {qtd_lugares}")

print(f"Total de requisições para api: {qtd_fetch}")
df.to_csv('imoveis_escolas_violencia_farmacia_mercado_shopping_parque.csv', index=False)

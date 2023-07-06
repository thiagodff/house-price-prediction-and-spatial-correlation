import pandas as pd

# Read the property CSV file
property_df = pd.read_csv("imoveis_escolas_proximas.csv")

# Read the schools CSV file
crimes_violentos_por_bairro_df = pd.read_csv("crimes_violentos_bairro_jan_jul_2022.csv")

# Merge the two DataFrames on the "neighborhood" column
merged_df = property_df.merge(crimes_violentos_por_bairro_df, on="bairro")
# merged_df = merged_df.fillna('')

# Add a new column to the merged DataFrame that contains the number of schools in each neighborhood
merged_df["qtd_crimes_violentos"] = merged_df["qtd_crimes_violentos"].astype(int)

# Write the merged DataFrame to a new CSV file
merged_df.to_csv("imoveis_bairros_violentos.csv", index=False)

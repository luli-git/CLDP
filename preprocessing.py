import pandas as pd

description_csv_path = "/home/luli/drugBank/drugbank.csv"
molecule_csv_path = "/home/luli/drugBank/output_smiles.csv"
description_df = pd.read_csv(description_csv_path)
molecule_df = pd.read_csv(molecule_csv_path)
description_df = description_df.dropna(subset=["description"])
# Merge dataframes on drug name
merged_df = description_df.merge(molecule_df, on="drugbank-id")

# save the merged dataframe as csv
merged_df.to_csv("merged_description_smiles.csv", index=False)

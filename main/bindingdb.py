import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent.parent / "data" / "bindingdb_data" / "BindingDB_All.tsv"
df = pd.read_csv(file_path, sep="\t")
df = df[['Target Name', 'Ligand SMILES', 'Ki (nM)', 'Kd (nM)', 'IC50 (nM)', 'Reference PMID', 'UniProt ID']]

print(df.head())
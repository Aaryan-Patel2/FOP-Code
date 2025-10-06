from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Read your SMILES file
with open('your_smiles.txt', 'r') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

# Convert each SMILES to 3D
for i, smiles in enumerate(smiles_list):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Save as PDB for docking
        Chem.MolToPDBFile(mol, f'ligand_{i}.pdb')
        print(f"✓ Converted {smiles[:20]}... to ligand_{i}.pdb")
    except:
        print(f"✗ Failed to convert {smiles}")

print(f"Successfully converted {len(smiles_list)} molecules")
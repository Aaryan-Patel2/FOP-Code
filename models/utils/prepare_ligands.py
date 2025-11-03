from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdDistGeom
import os

# Read your SMILES file

def smiles3D():
    # Set up input and output directories
    base_dir = os.path.dirname(__file__)
    smiles_file = os.path.join(base_dir, 'data', 'initial_SMILES', 'SMILES_strings.txt')
    output_dir = os.path.join(base_dir, 'data', 'ligands_3d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    # Convert each SMILES to 3D
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol, randomSeed=42)
            rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
            
            # Save as PDB for docking in the output directory
            output_path = os.path.join(output_dir, f'ligand_{i}.pdb')
            Chem.MolToPDBFile(mol, output_path)
            print(f"✓ Converted {smiles[:20]}... to {output_path}")
        except:
            print(f"✗ Failed to convert {smiles}")

    print(f"Successfully converted {len(smiles_list)} molecules")


if __name__ == "__main__":
    smiles3D()
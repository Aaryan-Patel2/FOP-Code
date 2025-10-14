import pandas as pd
import os
import glob
from models.utils.scoring import *
from utils.parse_vina import * 

# Set up paths
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to FOP-Code directory
ligands_dir = os.path.join(base_dir, 'data', 'converted_ligands')
receptors_dir = os.path.join(base_dir, 'data', 'converted_structures')

# Get all ligand files (pdbqt and other formats)
ligand_files = glob.glob(os.path.join(ligands_dir, 'ligand_*.*'))
ligand_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

# Define receptor files
mutant_receptor = os.path.join(receptors_dir, 'acvr1_mutant_3mtf.pdbqt')
wt_receptor = os.path.join(receptors_dir, 'acvr1_wt_3mtf.pdbqt')

print(f"Found {len(ligand_files)} ligand files")
print(f"Mutant receptor: {mutant_receptor}")
print(f"Wildtype receptor: {wt_receptor}")

# Your docking results
results = []
for ligand_file in ligand_files:
    # Extract ligand number from filename
    ligand_name = os.path.basename(ligand_file)
    ligand_num = ligand_name.split('_')[1].split('.')[0]
    
    # For now, we're parsing pre-docked results
    # Assuming docked files are named: mutant_docked_{i}.pdbqt and wildtype_docked_{i}.pdbqt
    mutant_docked = os.path.join(base_dir, f"mutant_docked_{ligand_num}.pdbqt")
    wt_docked = os.path.join(base_dir, f"wildtype_docked_{ligand_num}.pdbqt")
    
    mutant_score = parse_vina_score(mutant_docked)
    wt_score = parse_vina_score(wt_docked)
    
    # USE YOUR CORRECTED SCORING FUNCTION:
    total_score = calculate_temporary_inhibition_score(mutant_score, wt_score)
    
    results.append({
        'molecule': f'ligand_{ligand_num}',
        'ligand_file': ligand_name,
        'mutant_score': mutant_score,
        'wt_score': wt_score, 
        'specificity': wt_score - mutant_score,
        'total_score': total_score
    })

# Find best candidates
df = pd.DataFrame(results)
best_molecules = df.nlargest(5, 'total_score')
print("TOP 5 CANDIDATES:")
print(best_molecules)
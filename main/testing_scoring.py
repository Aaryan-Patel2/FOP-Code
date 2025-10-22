import pandas as pd
import os
import sys
import glob


# Mutant score is trash, specificity is pretty reasonable (obv. since it comes from Vina), affinity is okay
# GOAL: Use affinity-designed models to then rearrange and get K_d


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.utils.scoring import *
from utils.parse_vina import * 

# Set up paths
base_dir = os.path.dirname(os.path.dirname(__file__))
ligands_dir = os.path.join(base_dir, 'data', 'converted_ligands')
docking_results_dir = os.path.join(base_dir, 'docking_results')

# Get ACTUAL docking result files (not expected ones)
mutant_files = glob.glob(os.path.join(docking_results_dir, "mutant_docked_*.pdbqt"))
wt_files = glob.glob(os.path.join(docking_results_dir, "wildtype_docked_*.pdbqt"))

print(f"Found {len(mutant_files)} mutant docking results")
print(f"Found {len(wt_files)} wildtype docking results")

# Process only the ligands that actually docked successfully
results = []
for mutant_file in mutant_files:
    # Extract ligand number from filename
    ligand_num = os.path.basename(mutant_file).split('_')[2].split('.')[0]
    wt_file = os.path.join(docking_results_dir, f"wildtype_docked_{ligand_num}.pdbqt")
    
    # Check if wildtype result also exists
    if os.path.exists(wt_file):
        mutant_score = parse_vina_score(mutant_file)
        wt_score = parse_vina_score(wt_file)
        
        # SANITY CHECK: Skip impossible scores
        if -15 < mutant_score < 0 and -15 < wt_score < 0:
            total_score = calculate_temporary_inhibition_score(mutant_score, wt_score)
            
            results.append({
                'molecule': f'ligand_{ligand_num}',
                'mutant_score': mutant_score,
                'wt_score': wt_score, 
                'specificity': wt_score - mutant_score,
                'total_score': total_score
            })
        else:
            print(f"SKIPPING ligand_{ligand_num}: Impossible scores (mutant: {mutant_score}, wt: {wt_score})")

# Find best candidates
if results:
    df = pd.DataFrame(results)
    
    # Filter out garbage scores before ranking
    reasonable_df = df[(df['mutant_score'] > -15) & (df['mutant_score'] < 0) & 
                       (df['wt_score'] > -15) & (df['wt_score'] < 0)]
    
    if len(reasonable_df) > 0:
        best_molecules = reasonable_df.nlargest(5, 'total_score')
        print("\n" + "="*50)
        print("TOP 5 REASONABLE CANDIDATES:")
        print("="*50)
        print(best_molecules)
        
        # Save full results
        output_csv = os.path.join(base_dir, 'docking_scores_clean.csv')
        reasonable_df.to_csv(output_csv, index=False)
        print(f"\nClean results saved to: {output_csv}")
        
        # Show molecules with good temporary inhibition potential
        good_candidates = reasonable_df[(reasonable_df['specificity'] > 0.5) & 
                                      (reasonable_df['mutant_score'] > -9.0) & 
                                      (reasonable_df['mutant_score'] < -6.0)]
        print(f"\nMolecules with good temporary inhibition potential: {len(good_candidates)}")
        if len(good_candidates) > 0:
            print(good_candidates[['molecule', 'mutant_score', 'specificity', 'total_score']])
    else:
        print("No reasonable docking scores found!")
else:
    print("No successful docking results to analyze!")
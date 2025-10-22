import os
import subprocess
import glob

# Set up paths
base_dir = os.path.dirname(os.path.dirname(__file__))
ligands_dir = os.path.join(base_dir, 'data', 'converted_ligands')
receptors_dir = os.path.join(base_dir, 'data', 'converted_structures')
output_dir = os.path.join(base_dir, 'docking_results')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get receptor files
mutant_receptor = os.path.join(receptors_dir, 'acvr1_mutant_3mtf.pdbqt')
wt_receptor = os.path.join(receptors_dir, 'acvr1_wt_3mtf.pdbqt')

# Get ligand files
ligand_files = glob.glob(os.path.join(ligands_dir, 'ligand_*.pdbqt'))
ligand_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

print(f"Found {len(ligand_files)} ligand files")

# via PyMol
center_x, center_y, center_z = -17.7, -13.7, 38.4 
size_x, size_y, size_z = 12, 16, 10

# In run_docking.py, add better error checking:
def run_vina_docking(receptor, ligand, output):
    cmd = [
        'vina',
        '--receptor', receptor,
        '--ligand', ligand,
        '--out', output,
        '--center_x', '-17.7',
        '--center_y', '-13.7', 
        '--center_z', '38.4',
        '--size_x', '12',
        '--size_y', '16',
        '--size_z', '10',
        '--exhaustiveness', '8'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5min timeout
        if result.returncode == 0:
            # Verify the output file was created and has reasonable size
            if os.path.exists(output) and os.path.getsize(output) > 1000:
                print(f"✓ Successfully docked {os.path.basename(ligand)}")
                return True
            else:
                print(f"✗ Output file empty/missing for {os.path.basename(ligand)}")
                return False
        else:
            print(f"✗ Vina failed for {os.path.basename(ligand)}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout for {os.path.basename(ligand)}")
        return False
    except Exception as e:
        print(f"✗ Error running Vina for {os.path.basename(ligand)}: {e}")
        return False

# Run docking for all ligands
for i, ligand_file in enumerate(ligand_files):
    print(f"\nDocking {i+1}/{len(ligand_files)}: {os.path.basename(ligand_file)}")
    
    # Dock to mutant
    mutant_output = os.path.join(output_dir, f"mutant_docked_{i}.pdbqt")
    run_vina_docking(mutant_receptor, ligand_file, mutant_output)
    
    # Dock to wildtype  
    wt_output = os.path.join(output_dir, f"wildtype_docked_{i}.pdbqt")
    run_vina_docking(wt_receptor, ligand_file, wt_output)

print(f"\nDocking complete! Results saved to {output_dir}")
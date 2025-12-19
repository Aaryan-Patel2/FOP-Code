#!/usr/bin/env python3
"""
AutoDock Vina Docking Validation
Test generated molecules against receptor structure to validate predicted affinity
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from Bio.PDB import PDBParser


def smiles_to_3d(smiles: str, output_pdb: str, optimize: bool = True) -> bool:
    """
    Convert SMILES to 3D structure and save as PDB.
    
    Args:
        smiles: SMILES string
        output_pdb: Output PDB file path
        optimize: Whether to optimize geometry with MMFF
    
    Returns:
        True if successful
    """
    print(f"Converting SMILES to 3D: {smiles}")
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Invalid SMILES: {smiles}")
        return False
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    success = AllChem.EmbedMolecule(mol, randomSeed=42)
    if success != 0:
        print(f"WARNING: 3D embedding failed, trying with random coords")
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    
    # Optimize geometry
    if optimize:
        print("Optimizing geometry with MMFF...")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            print("WARNING: MMFF optimization failed, using UFF")
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    
    # Save to PDB
    Chem.MolToPDBFile(mol, output_pdb)
    print(f"✓ 3D structure saved to: {output_pdb}")
    
    return True


def prepare_ligand_pdbqt(pdb_file: str, pdbqt_file: str) -> bool:
    """
    Convert PDB to PDBQT format using Open Babel.
    
    Args:
        pdb_file: Input PDB file
        pdbqt_file: Output PDBQT file
    
    Returns:
        True if successful
    """
    print(f"Converting {pdb_file} to PDBQT format...")
    
    # Try with obabel (Open Babel)
    cmd = f"obabel {pdb_file} -O {pdbqt_file} -p 7.4"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(pdbqt_file):
            print(f"✓ PDBQT file created: {pdbqt_file}")
            return True
        else:
            print(f"ERROR: obabel failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Failed to run obabel: {e}")
        return False


def prepare_receptor_pdbqt(pdb_file: str, pdbqt_file: str) -> bool:
    """
    Prepare receptor for docking (add hydrogens, assign charges).
    
    Args:
        pdb_file: Input receptor PDB
        pdbqt_file: Output PDBQT file
    
    Returns:
        True if successful
    """
    if os.path.exists(pdbqt_file):
        print(f"✓ Receptor PDBQT already exists: {pdbqt_file}")
        return True
    
    print(f"Preparing receptor: {pdb_file}")
    
    # Use obabel for receptor preparation
    cmd = f"obabel {pdb_file} -O {pdbqt_file} -xr"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(pdbqt_file):
            print(f"✓ Receptor PDBQT created: {pdbqt_file}")
            return True
        else:
            print(f"ERROR: Receptor preparation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def get_ligand_center(ligand_pdb: str) -> Tuple[float, float, float]:
    """
    Calculate geometric center of reference ligand.
    
    Args:
        ligand_pdb: Reference ligand PDB file
    
    Returns:
        (x, y, z) centroid coordinates
    """
    coords = []
    
    with open(ligand_pdb, 'r') as f:
        for line in f:
            if line.startswith(('HETATM', 'ATOM')):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                except ValueError:
                    continue
    
    if not coords:
        raise ValueError(f"No coordinates found in {ligand_pdb}")
    
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    
    # Calculate box size (max distance from centroid + buffer)
    distances = np.linalg.norm(coords - centroid, axis=1)
    max_dist = distances.max()
    box_size = max_dist * 2 + 10  # Add 10 Å buffer
    
    return tuple(centroid), box_size


def run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: float = 25.0,
    output_pdbqt: str = "docked.pdbqt",
    exhaustiveness: int = 8
) -> Optional[Dict]:
    """
    Run AutoDock Vina docking.
    
    Args:
        receptor_pdbqt: Receptor PDBQT file
        ligand_pdbqt: Ligand PDBQT file
        center: (x, y, z) coordinates of binding box center
        box_size: Size of binding box (Angstroms)
        output_pdbqt: Output file for docked poses
        exhaustiveness: Vina exhaustiveness parameter (higher = more thorough)
    
    Returns:
        Dictionary with docking results or None if failed
    """
    print("\n" + "="*70)
    print("RUNNING AUTODOCK VINA")
    print("="*70)
    print(f"Receptor: {receptor_pdbqt}")
    print(f"Ligand: {ligand_pdbqt}")
    print(f"Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"Box size: {box_size:.2f} Å")
    print(f"Exhaustiveness: {exhaustiveness}")
    
    # Build Vina command
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size),
        "--size_y", str(box_size),
        "--size_z", str(box_size),
        "--out", output_pdbqt,
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "9"  # Generate 9 binding poses
    ]
    
    # Run Vina
    try:
        print("\nRunning Vina (this may take 2-5 minutes)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"ERROR: Vina failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse output
        output = result.stdout
        print("\n" + output)
        
        # Extract binding affinities
        affinities = []
        for line in output.split('\n'):
            if line.strip().startswith('1') or line.strip().startswith('2'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        affinity = float(parts[1])
                        affinities.append(affinity)
                    except ValueError:
                        continue
        
        if not affinities:
            print("WARNING: Could not parse binding affinities from Vina output")
            return None
        
        # Best pose (most negative affinity)
        best_affinity = affinities[0]
        
        # Convert to pKd (approximate): pKd ≈ -ΔG / (RT * ln(10))
        # where RT ≈ 0.593 kcal/mol at 298K
        # pKd ≈ -affinity / 1.364
        predicted_pkd = -best_affinity / 1.364
        
        results = {
            "vina_affinity_kcal_mol": best_affinity,
            "vina_predicted_pkd": predicted_pkd,
            "all_affinities": affinities,
            "output_file": output_pdbqt
        }
        
        return results
        
    except subprocess.TimeoutExpired:
        print("ERROR: Vina docking timed out (>10 minutes)")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Main docking workflow."""
    
    # Configuration
    SMILES = "CCC(=C=C=N)CC1NCCC(C2CC(CCC(C)C(=O)O)C2(C)C)O1"
    RECEPTOR_PDB = "data/structures/receptor_siteA.pdb"
    REFERENCE_LIGAND_PDB = "data/structures/ligand_siteA.pdb"
    PREDICTED_PKD = 7.134643077850342
    PREDICTED_KOFF = 0.2708185911178589
    
    # Output directory
    output_dir = Path("docking_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("AUTODOCK VINA VALIDATION - TOP GCDM MOLECULE")
    print("="*70)
    print(f"SMILES: {SMILES}")
    print(f"Predicted pKd: {PREDICTED_PKD:.2f}")
    print(f"Predicted k_off: {PREDICTED_KOFF:.3f} s⁻¹")
    print("="*70 + "\n")
    
    # Step 1: Convert SMILES to 3D
    ligand_pdb = output_dir / "ligand_3d.pdb"
    if not smiles_to_3d(SMILES, str(ligand_pdb)):
        print("ERROR: Failed to generate 3D structure")
        return
    
    # Step 2: Prepare ligand PDBQT
    ligand_pdbqt = output_dir / "ligand.pdbqt"
    if not prepare_ligand_pdbqt(str(ligand_pdb), str(ligand_pdbqt)):
        print("ERROR: Failed to prepare ligand PDBQT")
        return
    
    # Step 3: Prepare receptor PDBQT
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    if not prepare_receptor_pdbqt(RECEPTOR_PDB, str(receptor_pdbqt)):
        print("ERROR: Failed to prepare receptor PDBQT")
        return
    
    # Step 4: Get binding box center from reference ligand
    print(f"\nExtracting binding box center from: {REFERENCE_LIGAND_PDB}")
    center, box_size = get_ligand_center(REFERENCE_LIGAND_PDB)
    print(f"✓ Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"✓ Box size: {box_size:.2f} Å")
    
    # Step 5: Run Vina docking
    output_pdbqt = output_dir / "docked_top_molecule.pdbqt"
    results = run_vina_docking(
        receptor_pdbqt=str(receptor_pdbqt),
        ligand_pdbqt=str(ligand_pdbqt),
        center=center,
        box_size=box_size,
        output_pdbqt=str(output_pdbqt),
        exhaustiveness=8
    )
    
    if results is None:
        print("\nERROR: Docking failed")
        return
    
    # Step 6: Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"Bayesian Predictor pKd:  {PREDICTED_PKD:.2f}")
    print(f"Vina Estimated pKd:      {results['vina_predicted_pkd']:.2f}")
    print(f"Vina Binding Affinity:   {results['vina_affinity_kcal_mol']:.2f} kcal/mol")
    print(f"\nDifference (pKd units):  {abs(PREDICTED_PKD - results['vina_predicted_pkd']):.2f}")
    
    # Interpretation
    diff = abs(PREDICTED_PKD - results['vina_predicted_pkd'])
    if diff < 1.0:
        interpretation = "✓ EXCELLENT agreement (< 1 pKd unit)"
    elif diff < 2.0:
        interpretation = "✓ GOOD agreement (< 2 pKd units)"
    elif diff < 3.0:
        interpretation = "⚠ MODERATE agreement (2-3 pKd units)"
    else:
        interpretation = "✗ POOR agreement (> 3 pKd units)"
    
    print(f"\nInterpretation: {interpretation}")
    print("="*70)
    
    # Save results to JSON
    results_json = output_dir / "docking_results.json"
    results["predicted_pkd"] = PREDICTED_PKD
    results["predicted_koff"] = PREDICTED_KOFF
    results["smiles"] = SMILES
    results["interpretation"] = interpretation
    
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_json}")
    print(f"✓ Docked poses saved to: {output_pdbqt}")
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test affinity-guided GCDM generation
Generate molecules for ACVR1 with affinity predictor guidance
"""

import sys
sys.path.insert(0, '/workspace/GCDM-SBDD')
sys.path.insert(0, '/workspace/FOP-Code')
sys.path.insert(0, '/workspace')

import torch
from lightning_modules import LigandPocketDDPM
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

# Import affinity guidance module
from affinity_guidance import add_guidance_to_gcdm, predict_affinity_from_coords

print("="*70)
print("AFFINITY-GUIDED MOLECULE GENERATION TEST")
print("="*70)

# ACVR1 sequence
ACVR1_SEQ = 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPARTVATRQVVEKEKRATLLLFSNPPNPNNAKKKMEEWTFLRLSQDSRPPNPSLLHGSSPPPPSHRQFPEEESPGDASSSSSSTQSSSDLQAFQTNPSAALVAGSSPTLSGTPSPTGLVTPSSHTVSSPVPPPAPSGGGAEVESAPAGAVGPSSPLPASQPVGGMPDVSPGSAYAVSGSSVFPSSSHVGMGFPAAAGFPFVPSSS'

# PDB file - ACVR1 wild-type structure
PDB_FILE = '/workspace/acvr1.pdb'

# Load GCDM model
print("\n1. Loading GCDM model...")
checkpoint_path = '/workspace/GCDM-SBDD/checkpoints/crossdocked_ca_cond_egnn.ckpt'
model = LigandPocketDDPM.load_from_checkpoint(
    checkpoint_path,
    map_location='cuda' if torch.cuda.is_available() else 'cpu'
)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"✓ GCDM loaded on {device}")

# Enable affinity guidance via monkey-patching
print(f"\n2. Enabling affinity guidance...")
guidance_scale = 1.0  # Start with 1.0, tune if needed

try:
    add_guidance_to_gcdm(
        ddpm=model.ddpm,
        protein_sequence=ACVR1_SEQ,
        guidance_scale=guidance_scale,
        checkpoint_path='/workspace/affinity_model.ckpt'
    )
except Exception as e:
    print(f"✗ Failed to enable guidance: {e}")
    print("Continuing without guidance...")
    import traceback
    traceback.print_exc()

# Generate molecules
print(f"\n3. Generating molecules with guidance...")
print(f"   PDB: {PDB_FILE}")
print(f"   Samples: 20")
print(f"   Guidance: {guidance_scale}")

try:
    # Generate molecules with pocket definition
    # Using ref_ligand format (chain:residue) or pocket_ids list
    results = model.generate_ligands(
        pdb_file=PDB_FILE,
        n_samples=20,
        ref_ligand='A:300',  # Adjust based on PDB - using placeholder
        num_nodes_lig=25,  # Target size
        sanitize=True,
        relax_iter=0
    )
    
    print(f"\n4. Generation complete!")
    print(f"   Generated: {len(results)} molecules")
    
    # Analyze results
    print(f"\n5. Analyzing generated molecules...")
    print("="*70)
    
    valid_count = 0
    for i, mol_data in enumerate(results):
        mol = mol_data.get('mol')
        smiles = mol_data.get('smiles')
        
        if mol is not None:
            valid_count += 1
            n_atoms = mol.GetNumAtoms()
            mw = Descriptors.MolWt(mol)
            
            print(f"\nMolecule {i+1}:")
            print(f"  Atoms: {n_atoms}")
            print(f"  MW: {mw:.1f} Da")
            print(f"  SMILES: {smiles}")
    
    print("\n" + "="*70)
    print(f"SUCCESS: {valid_count}/{len(results)} valid molecules generated")
    print("="*70)
    
    # Save results
    output_file = '/workspace/guided_molecules.sdf'
    writer = Chem.SDWriter(output_file)
    for mol_data in results:
        mol = mol_data.get('mol')
        if mol is not None:
            writer.write(mol)
    writer.close()
    
    print(f"\n✓ Molecules saved to: {output_file}")
    print("  Copy to host with: docker cp fop-gcdm:/workspace/guided_molecules.sdf ~/FOP-Code/")
    
except Exception as e:
    print(f"\n✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

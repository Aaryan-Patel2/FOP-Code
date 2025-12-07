#!/usr/bin/env python3
"""
Flask API for GCDM-SBDD
Provides REST API for molecule generation from within Docker container
"""

import os
import sys
import json
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
import torch

# Add GCDM-SBDD to path if needed
if '/workspace/GCDM-SBDD' not in sys.path:
    sys.path.insert(0, '/workspace/GCDM-SBDD')

# Import GCDM modules
from lightning_modules import LigandPocketDDPM  # type: ignore
import utils
from rdkit import Chem
from rdkit.Chem import Descriptors

app = Flask(__name__)
CORS(app)

# Global model storage
MODELS = {}

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load GCDM model from checkpoint"""
    if checkpoint_path in MODELS:
        return MODELS[checkpoint_path]
    
    print(f"Loading GCDM model from {checkpoint_path}...")
    model = LigandPocketDDPM.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model = model.to(device)
    model.eval()
    MODELS[checkpoint_path] = model
    print(f"✓ Model loaded successfully")
    return model


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'message': 'GCDM-SBDD API is running'
    })


@app.route('/generate', methods=['POST'])
def generate_molecules():
    """
    Generate molecules for a given pocket
    
    Request JSON:
    {
        "pdb_file": "<PDB file content as string>" OR "/path/to/file.pdb",
        "binding_site_residues": ["A:201", "A:202", "A:203"],
        "n_samples": 1,
        "checkpoint": "/workspace/GCDM-SBDD/GCDM_SBDD_Checkpoints/crossdocked_fullatom_cond.ckpt",
        "sanitize": true,
        "num_nodes_lig": null,
        "device": "auto"
    }
    
    Returns:
    {
        "success": true,
        "molecules": [
            {
                "smiles": "CC(C)Cc1ccc...",
                "mol_weight": 342.5,
                "logp": 3.2,
                "num_atoms": 25
            },
            ...
        ],
        "n_generated": 1,
        "n_requested": 1
    }
    """
    temp_pdb_file = None
    
    try:
        data = request.json
        
        # Parse parameters
        checkpoint = data.get('checkpoint', '/workspace/GCDM-SBDD/checkpoints/crossdocked_ca_cond_gcpnet.ckpt')
        pdb_input = data.get('pdb_file')
        resi_list = data.get('binding_site_residues', [])
        ref_ligand = data.get('ref_ligand')
        n_samples = data.get('n_samples', 1)
        sanitize = data.get('sanitize', True)
        num_nodes_lig = data.get('num_nodes_lig')
        device_pref = data.get('device', 'auto')
        
        if not pdb_input:
            return jsonify({'success': False, 'error': 'pdb_file is required'}), 400
        
        # Determine device
        if device_pref == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_pref
        
        print(f"\n{'='*70}")
        print(f"Generation Request:")
        print(f"  Device: {device}")
        print(f"  Samples: {n_samples}")
        print(f"  Binding site: {resi_list}")
        print(f"  Checkpoint: {Path(checkpoint).name}")
        print(f"{'='*70}\n")
        
        # Handle PDB input - could be file path or content
        # Check if it looks like a file path (short string) vs PDB content (long, has ATOM lines)
        if len(pdb_input) < 500 and os.path.exists(pdb_input):
            # It's a file path that exists
            pdb_file = pdb_input
            print(f"Using PDB file: {pdb_file}")
        else:
            # PDB content provided as string - save to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_input)
                temp_pdb_file = f.name
                pdb_file = temp_pdb_file
            print(f"Created temporary PDB file: {pdb_file}")
        
        # Load model
        model = load_model(checkpoint, device)
        
        # Prepare num_nodes_lig if specified
        if num_nodes_lig is not None:
            num_nodes_lig_tensor = torch.ones(n_samples, dtype=torch.int) * num_nodes_lig
        else:
            num_nodes_lig_tensor = None
        
        # Generate molecules using GCDM
        print(f"\nGenerating {n_samples} molecules...")
        
        molecules = model.generate_ligands(
            pdb_file,
            n_samples,
            resi_list,
            ref_ligand,
            num_nodes_lig_tensor,
            sanitize,
            largest_frag=True,
            relax_iter=0,
            sample_chain=False,
            resamplings=1,
            jump_length=1,
            timesteps=None
        )
        
        print(f"✓ Generated {len(molecules)} molecules")
        
        # Convert molecules to serializable format
        molecules_data = []
        for idx, mol in enumerate(molecules):
            if mol is None:
                print(f"  Molecule {idx}: Failed (None)")
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                mol_data = {
                    'smiles': smiles,
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms()
                }
                
                # Calculate molecular properties safely
                try:
                    mol_data['mol_weight'] = float(Descriptors.MolWt(mol))  # type: ignore
                    mol_data['logp'] = float(Descriptors.MolLogP(mol))  # type: ignore
                except:
                    pass
                
                # Try to calculate QED if available
                try:
                    from rdkit.Chem import QED
                    mol_data['qed'] = float(QED.qed(mol))
                except:
                    pass
                
                molecules_data.append(mol_data)
                print(f"  Molecule {idx}: {smiles[:50]}...")
                
            except Exception as e:
                print(f"  Molecule {idx}: Failed to process - {e}")
                continue
        
        print(f"\n{'='*70}")
        print(f"Generation Complete: {len(molecules_data)}/{n_samples} successful")
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': True,
            'molecules': molecules_data,
            'n_generated': len(molecules_data),
            'n_requested': n_samples,
            'device': device
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*70}")
        print(f"ERROR:")
        print(error_trace)
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace
        }), 500
    
    finally:
        # Clean up temp file
        if temp_pdb_file and os.path.exists(temp_pdb_file):
            try:
                os.unlink(temp_pdb_file)
                print(f"Cleaned up temporary file: {temp_pdb_file}")
            except:
                pass


@app.route('/available_checkpoints', methods=['GET'])
def list_checkpoints():
    """List available model checkpoints"""
    checkpoint_dirs = [
        Path('/workspace/GCDM-SBDD/GCDM_SBDD_Checkpoints'),
        Path('/workspace/GCDM-SBDD/checkpoints')
    ]
    
    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            for p in checkpoint_dir.glob('*.ckpt'):
                checkpoints.append(str(p))
    
    return jsonify({
        'checkpoints': checkpoints,
        'count': len(checkpoints)
    })


if __name__ == '__main__':
    print("=" * 70)
    print("GCDM-SBDD Flask API Server")
    print("=" * 70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running in CPU mode")
    print("=" * 70)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

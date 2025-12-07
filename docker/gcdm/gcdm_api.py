#!/usr/bin/env python3
"""
Flask API for GCDM-SBDD
Provides REST API for molecule generation from within Docker container
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, request, jsonify  # type: ignore
from flask_cors import CORS  # type: ignore
import torch
import numpy as np

# Import GCDM modules
from lightning_modules import LigandPocketDDPM  # type: ignore
from analysis.molecule_builder import build_molecule  # type: ignore
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

app = Flask(__name__)
CORS(app)

# Global model storage
MODELS = {}

def load_model(checkpoint_path: str, device: str = 'cuda') -> LigandPocketDDPM:
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
        'models_loaded': len(MODELS)
    })


@app.route('/generate', methods=['POST'])
def generate_molecules():
    """
    Generate molecules for a given pocket
    
    Request JSON:
    {
        "checkpoint": "checkpoints/bindingmoad_ca_cond_gcpnet.ckpt",
        "pdb_file": "/workspace/fop-data/structures/protein.pdb",
        "resi_list": ["A:1", "A:2", "A:3"],
        "n_samples": 20,
        "sanitize": true,
        "fix_n_nodes": false,
        "device": "cuda"
    }
    
    Returns:
    {
        "success": true,
        "molecules": [
            {
                "smiles": "CC(C)Cc1ccc...",
                "mol_weight": 342.5,
                "logp": 3.2,
                "qed": 0.65,
                "num_atoms": 25
            },
            ...
        ],
        "n_generated": 18,
        "n_requested": 20
    }
    """
    try:
        data = request.json
        
        # Parse parameters
        checkpoint = data.get('checkpoint', 'checkpoints/bindingmoad_ca_cond_gcpnet.ckpt')
        pdb_file = data.get('pdb_file')
        resi_list = data.get('resi_list', [])
        ref_ligand = data.get('ref_ligand')
        n_samples = data.get('n_samples', 20)
        sanitize = data.get('sanitize', True)
        fix_n_nodes = data.get('fix_n_nodes', False)
        device = data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if not pdb_file:
            return jsonify({'success': False, 'error': 'pdb_file is required'}), 400
        
        if not Path(pdb_file).exists():
            return jsonify({'success': False, 'error': f'PDB file not found: {pdb_file}'}), 400
        
        # Load model
        model = load_model(checkpoint, device)
        
        # Generate molecules using GCDM
        # NOTE: This is a simplified version - you'll need to adapt based on GCDM's actual API
        # The actual implementation should call model.generate() or similar
        
        generated_mols = []
        
        print(f"Generating {n_samples} molecules for pocket in {pdb_file}...")
        
        # This is a placeholder - replace with actual GCDM generation logic
        # You'll need to:
        # 1. Load PDB and extract pocket
        # 2. Call model's generation method
        # 3. Build molecules from generated coordinates
        
        for i in range(n_samples):
            try:
                # Placeholder for actual generation
                # mol = generate_single_molecule(model, pdb_file, resi_list, device)
                
                # For now, return structure indicating what should be here
                pass
            except Exception as e:
                print(f"Failed to generate molecule {i}: {e}")
                continue
        
        # Convert molecules to serializable format
        molecules_data = []
        for mol in generated_mols:
            if mol is None:
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                molecules_data.append({
                    'smiles': smiles,
                    'mol_weight': float(Descriptors.MolWt(mol)),  # type: ignore
                    'logp': float(Descriptors.MolLogP(mol)),  # type: ignore
                    'qed': float(Descriptors.qed(mol)),  # type: ignore
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms()
                })
            except Exception as e:
                print(f"Failed to process molecule: {e}")
                continue
        
        return jsonify({
            'success': True,
            'molecules': molecules_data,
            'n_generated': len(molecules_data),
            'n_requested': n_samples
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/available_checkpoints', methods=['GET'])
def list_checkpoints():
    """List available model checkpoints"""
    checkpoint_dir = Path('/workspace/GCDM-SBDD/checkpoints')
    if not checkpoint_dir.exists():
        return jsonify({'checkpoints': []})
    
    checkpoints = [str(p.relative_to('/workspace/GCDM-SBDD')) 
                   for p in checkpoint_dir.glob('*.ckpt')]
    
    return jsonify({'checkpoints': checkpoints})


if __name__ == '__main__':
    print("=" * 70)
    print("GCDM-SBDD Flask API Server")
    print("=" * 70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

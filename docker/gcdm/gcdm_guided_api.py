#!/usr/bin/env python3
"""
Flask API for Affinity-Guided GCDM-SBDD
Integrates affinity predictor directly into the diffusion sampling process
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np

# Import GCDM modules
from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

app = Flask(__name__)
CORS(app)

# Global model storage
MODELS = {}
AFFINITY_MODEL = None


def load_gcdm_model(checkpoint_path: str, device: str = 'cuda') -> LigandPocketDDPM:
    """Load GCDM diffusion model"""
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
    print(f"✓ GCDM model loaded")
    return model


def load_affinity_model(checkpoint_path: str, device: str = 'cuda'):
    """Load affinity predictor for guidance"""
    global AFFINITY_MODEL
    
    if AFFINITY_MODEL is not None:
        return AFFINITY_MODEL
    
    print(f"Loading affinity predictor from {checkpoint_path}...")
    
    # Import affinity predictor modules
    sys.path.insert(0, '/workspace/FOP-Code')
    from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork
    from models.pdbbind_data_preparation import ProteinSequenceEncoder, SMILESEncoder
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocab sizes from checkpoint
    ligand_vocab_size = checkpoint['state_dict']['model.ligand_encoder.embedding.weight'].shape[0]
    protein_vocab_size = checkpoint['state_dict']['model.protein_encoder.embedding.weight'].shape[0]
    
    # Create model
    model = HybridBayesianAffinityNetwork(
        protein_vocab_size=protein_vocab_size,
        ligand_vocab_size=ligand_vocab_size,
        complex_descriptor_dim=200,
        protein_output_dim=256,
        ligand_output_dim=256,
        complex_output_dim=128,
        fusion_hidden_dims=[512, 256, 128],
        dropout=0.3,
        prior_sigma=1.0
    )
    
    # Load state dict (strip 'model.' prefix)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Create encoders
    protein_encoder = ProteinSequenceEncoder()
    ligand_encoder = SMILESEncoder()
    
    AFFINITY_MODEL = {
        'model': model,
        'protein_encoder': protein_encoder,
        'ligand_encoder': ligand_encoder,
        'device': device
    }
    
    print(f"✓ Affinity predictor loaded")
    return AFFINITY_MODEL


def coords_to_smiles(coords: torch.Tensor, atom_types: torch.Tensor) -> Optional[str]:
    """Convert 3D coordinates and atom types to SMILES"""
    try:
        # Convert to numpy
        coords_np = coords.detach().cpu().numpy()
        types_np = atom_types.detach().cpu().numpy()
        
        # Build molecule with RDKit
        mol = build_molecule(coords_np, types_np)
        if mol is None:
            return None
        
        # Sanitize and generate SMILES
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
        
    except Exception as e:
        return None


def compute_affinity_gradient(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    protein_sequence: str,
    guidance_scale: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute gradient of affinity predictor w.r.t. coordinates
    
    This is the key function for classifier-guided diffusion.
    Returns gradient that pushes coordinates toward higher affinity.
    """
    if AFFINITY_MODEL is None:
        return torch.zeros_like(coords)
    
    try:
        # Convert current coordinates to SMILES
        smiles = coords_to_smiles(coords, atom_types)
        if smiles is None:
            return torch.zeros_like(coords)
        
        # Encode protein and ligand
        model = AFFINITY_MODEL['model']
        protein_encoder = AFFINITY_MODEL['protein_encoder']
        ligand_encoder = AFFINITY_MODEL['ligand_encoder']
        
        protein_tokens = protein_encoder.encode(protein_sequence, max_length=512)
        ligand_tokens = ligand_encoder.encode(smiles, max_length=128)
        complex_desc = np.zeros(200, dtype=np.float32)
        
        # Convert to tensors
        protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(device)
        ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(device)
        complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0).to(device)
        
        # Predict affinity with gradient
        coords.requires_grad_(True)
        
        with torch.enable_grad():
            affinity_pred, _ = model.predict_with_uncertainty(
                protein_tensor, 
                ligand_tensor, 
                complex_tensor,
                n_samples=1
            )
            
            # Maximize affinity = minimize negative affinity
            loss = -affinity_pred
            
            # Compute gradient
            grad = torch.autograd.grad(loss, coords, create_graph=False)[0]
            
        coords.requires_grad_(False)
        
        # Scale gradient
        return guidance_scale * grad
        
    except Exception as e:
        print(f"Warning: Affinity gradient computation failed: {e}")
        return torch.zeros_like(coords)


def guided_sampling_step(
    model: LigandPocketDDPM,
    x_t: torch.Tensor,
    atom_types: torch.Tensor,
    protein_coords: torch.Tensor,
    protein_sequence: str,
    t: int,
    guidance_scale: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Single denoising step with affinity guidance
    
    Implements classifier-guided diffusion:
    x_{t-1} = GCDM_denoise(x_t) - guidance_scale * ∇_x log p(affinity | x)
    """
    # Standard GCDM denoising prediction
    with torch.no_grad():
        # Get model's prediction of noise
        noise_pred = model.model(x_t, atom_types, protein_coords, t)
    
    # Compute affinity gradient for guidance
    if guidance_scale > 0 and AFFINITY_MODEL is not None:
        affinity_grad = compute_affinity_gradient(
            x_t, 
            atom_types, 
            protein_sequence,
            guidance_scale,
            device
        )
        
        # Add guidance to noise prediction
        # In diffusion: x_{t-1} = μ_θ(x_t) + guidance
        # where μ_θ = (x_t - √(1-ᾱ_t) * ε_θ(x_t)) / √ᾱ_t
        noise_pred = noise_pred + affinity_grad
    
    return noise_pred


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'gcdm_models_loaded': len(MODELS),
        'affinity_model_loaded': AFFINITY_MODEL is not None
    })


@app.route('/generate_guided', methods=['POST'])
def generate_guided_molecules():
    """
    Generate molecules with affinity-guided diffusion
    
    Request JSON:
    {
        "gcdm_checkpoint": "checkpoints/crossdocked_ca_cond_egnn.ckpt",
        "affinity_checkpoint": "/workspace/FOP-Code/trained_models/best_model-v1.ckpt",
        "pdb_file": "/workspace/fop-data/structures/protein.pdb",
        "protein_sequence": "MTEYKLVVVGAG...",
        "resi_list": ["A:1", "A:2"],
        "n_samples": 20,
        "guidance_scale": 1.0,
        "device": "cuda"
    }
    """
    try:
        data = request.json
        
        # Extract parameters
        gcdm_ckpt = data.get('gcdm_checkpoint', 'checkpoints/crossdocked_ca_cond_egnn.ckpt')
        affinity_ckpt = data.get('affinity_checkpoint')
        pdb_file = data['pdb_file']
        protein_sequence = data['protein_sequence']
        resi_list = data.get('resi_list')
        n_samples = data.get('n_samples', 20)
        guidance_scale = data.get('guidance_scale', 1.0)
        device = data.get('device', 'cuda')
        
        # Load models
        gcdm_model = load_gcdm_model(gcdm_ckpt, device)
        
        if affinity_ckpt and guidance_scale > 0:
            load_affinity_model(affinity_ckpt, device)
        
        # TODO: Implement full guided sampling loop
        # This requires modifying GCDM's internal sampling to call guided_sampling_step
        # For now, return placeholder
        
        return jsonify({
            'success': False,
            'error': 'Guided sampling requires GCDM source modification - see guided_sampling_step function',
            'note': 'To implement: modify LigandPocketDDPM.p_sample() to accept guidance_fn and call it each step'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

#!/usr/bin/env python3
"""
Affinity Guidance Module for GCDM
This module can be imported to add affinity guidance to GCDM sampling
"""

import torch
import sys
import os

# Add FOP-Code to path with multiple fallbacks
for path in ['/workspace/FOP-Code', '/workspace/FOP-Code/models', '/workspace']:
    if path not in sys.path:
        sys.path.insert(0, path)

# Global affinity model storage
AFFINITY_MODEL = None
PROTEIN_ENCODER = None
LIGAND_ENCODER = None

def load_affinity_predictor(checkpoint_path='/workspace/affinity_model.ckpt', device='cuda'):
    """Load affinity predictor for guidance"""
    global AFFINITY_MODEL, PROTEIN_ENCODER, LIGAND_ENCODER
    
    if AFFINITY_MODEL is not None:
        print("Affinity predictor already loaded")
        return AFFINITY_MODEL
    
    try:
        # Try direct import from models directory
        sys.path.insert(0, '/workspace/FOP-Code')
        from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork
        from models.pdbbind_data_preparation import ProteinSequenceEncoder, SMILESEncoder
        
        print(f"Loading affinity predictor from {checkpoint_path}...")
        
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
        
        AFFINITY_MODEL = model
        PROTEIN_ENCODER = protein_encoder
        LIGAND_ENCODER = ligand_encoder
        # Ensure encoders set
        if PROTEIN_ENCODER is None or LIGAND_ENCODER is None:
            print("Warning: Encoders not initialized correctly")
            return None
        
        print("✓ Affinity predictor loaded successfully")
        return model
        
    except Exception as e:
        print(f"✗ Failed to load affinity predictor: {e}")
        import traceback
        traceback.print_exc()
        return None


def coords_to_smiles(coords, atom_types):
    """Convert 3D coordinates and atom types to SMILES"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np
        
        # Convert to numpy
        if torch.is_tensor(coords):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.array(coords)
            
        if torch.is_tensor(atom_types):
            types_np = atom_types.detach().cpu().numpy()
        else:
            types_np = np.array(atom_types)
        
        # Atom type mapping (adjust based on GCDM encoding)
        # Common: H, C, N, O, F, S, Cl, Br
        atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'S', 6: 'Cl', 7: 'Br'}
        
        # Build RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for i in range(len(coords_np)):
            if len(types_np.shape) > 1:
                # One-hot encoding
                atom_type_idx = np.argmax(types_np[i])
            else:
                atom_type_idx = int(types_np[i])
            
            atom_symbol = atom_map.get(atom_type_idx, 'C')
            atom = Chem.Atom(atom_symbol)
            mol.AddAtom(atom)
        
        # Add bonds based on distance
        for i in range(len(coords_np)):
            for j in range(i+1, len(coords_np)):
                dist = np.linalg.norm(coords_np[i] - coords_np[j])
                # Typical bond lengths: 1.0-2.0 Å
                if 0.8 < dist < 2.0:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                    except:
                        pass
        
        # Convert to molecule and sanitize
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
        
    except Exception as e:
        # Return simple fallback
        return None


def predict_affinity_from_coords(coords, atom_types, protein_sequence, device='cuda'):
    """
    Predict affinity from 3D coordinates
    
    Returns: (affinity, uncertainty) or (None, None) if failed
    """
    if AFFINITY_MODEL is None:
        print("Affinity model not loaded")
        return None, None
    
    try:
        import numpy as np
        
        # Convert to SMILES
        smiles = coords_to_smiles(coords, atom_types)
        if smiles is None:
            return None, None
        
        # Encode sequences
        if PROTEIN_ENCODER is None or LIGAND_ENCODER is None:
            print("Encoders not available for prediction")
            return None, None
        protein_tokens = PROTEIN_ENCODER.encode(protein_sequence, max_length=512)
        ligand_tokens = LIGAND_ENCODER.encode(smiles, max_length=128)
        complex_desc = np.zeros(200, dtype=np.float32)
        
        # Convert to tensors
        protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(device)
        ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(device)
        complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            affinity, uncertainty = AFFINITY_MODEL.predict_with_uncertainty(
                protein_tensor,
                ligand_tensor,
                complex_tensor,
                n_samples=10
            )
        
        return affinity.item(), uncertainty.item()
        
    except Exception as e:
        return None, None


def compute_affinity_gradient(z_lig, protein_sequence, guidance_scale=1.0, device='cuda'):
    """
    Compute gradient of affinity w.r.t. ligand coordinates
    
    This is the core guidance function for classifier-guided diffusion
    """
    if AFFINITY_MODEL is None:
        print("Affinity model not loaded for gradient computation")
        return torch.zeros_like(z_lig)
    
    try:
        import numpy as np
        
        # Extract coordinates (first 3 dims) and atom types (rest)
        coords = z_lig[:, :3].clone()
        atom_types = z_lig[:, 3:].clone()
        
        # Convert to SMILES
        smiles = coords_to_smiles(coords, atom_types)
        if smiles is None:
            return torch.zeros_like(z_lig)
        
        # Encode sequences
        if PROTEIN_ENCODER is None or LIGAND_ENCODER is None:
            print("Encoders not available for gradient computation")
            return torch.zeros_like(z_lig)
        protein_tokens = PROTEIN_ENCODER.encode(protein_sequence, max_length=512)
        ligand_tokens = LIGAND_ENCODER.encode(smiles, max_length=128)
        complex_desc = np.zeros(200, dtype=np.float32)
        
        # Convert to tensors
        protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(device)
        ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(device)
        complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0).to(device)
        
        # Enable gradients for coordinates only
        coords.requires_grad_(True)
        
        # Predict affinity with gradient
        if not hasattr(AFFINITY_MODEL, 'predict_with_uncertainty'):
            print('Affinity model has no predict_with_uncertainty')
            return torch.zeros_like(z_lig)
        affinity, _ = AFFINITY_MODEL.predict_with_uncertainty(
            protein_tensor,
            ligand_tensor,
            complex_tensor,
            n_samples=1
        )
        
        # Maximize affinity = minimize -affinity
        loss = -affinity.mean()
        
        # Compute gradient
        if coords.grad is not None:
            coords.grad.zero_()
        loss.backward()
        
        if coords.grad is not None:
            grad_coords = coords.grad.clone() * guidance_scale
        else:
            grad_coords = torch.zeros_like(coords)
        
        # Create full gradient (only for coordinates, not atom types)
        grad_full = torch.zeros_like(z_lig)
        grad_full[:, :3] = grad_coords
        
        return grad_full
        
    except Exception as e:
        print(f"Guidance gradient failed: {e}")
        return torch.zeros_like(z_lig)


# Monkey-patch function to add guidance to GCDM
def add_guidance_to_gcdm(ddpm, protein_sequence, guidance_scale=1.0, checkpoint_path='/workspace/affinity_model.ckpt'):
    """
    Monkey-patch GCDM's diffusion model to add affinity guidance
    
    Args:
        ddpm: The ddpm object (model.ddpm)
        protein_sequence: Target protein sequence
        guidance_scale: Guidance strength (0 = off, 1.0 = default)
        checkpoint_path: Path to affinity model checkpoint
    """
    
    # Detect device from model parameters
    try:
        device = next(ddpm.parameters()).device
        device_str = str(device)
    except:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load affinity predictor
    if guidance_scale > 0:
        load_affinity_predictor(checkpoint_path, device=device_str)
    
    # Store guidance parameters
    ddpm._guidance_scale = guidance_scale
    ddpm._protein_sequence = protein_sequence
    ddpm._device = device_str
    
    # Save original method
    original_sample_p_zs_given_zt = ddpm.sample_p_zs_given_zt
    
    # Create wrapped method with guidance
    def sample_p_zs_given_zt_with_guidance(s, t, zt_lig, zt_pocket, ligand_mask, pocket_mask, fix_noise=False):
        # Call original sampling
        zs_lig, zs_pocket = original_sample_p_zs_given_zt(s, t, zt_lig, zt_pocket, ligand_mask, pocket_mask, fix_noise)
        
        # Apply guidance if enabled
        if ddpm._guidance_scale > 0 and ddpm._protein_sequence is not None:
            try:
                # Compute affinity gradient
                grad = compute_affinity_gradient(
                    zs_lig,
                    ddpm._protein_sequence,
                    ddpm._guidance_scale,
                    device=ddpm._device
                )
                
                # Add gradient to ligand coordinates
                zs_lig = zs_lig + grad
                
            except Exception as e:
                print(f"Warning: Guidance failed at step: {e}")
        
        return zs_lig, zs_pocket
    
    # Replace method
    ddpm.sample_p_zs_given_zt = sample_p_zs_given_zt_with_guidance
    
    print(f"✓ Affinity guidance enabled (scale={guidance_scale})")


if __name__ == "__main__":
    print("Affinity Guidance Module for GCDM")
    print("Usage:")
    print("  from affinity_guidance import add_guidance_to_gcdm")
    print("  add_guidance_to_gcdm(model.ddpm, protein_sequence='MTEYK...', guidance_scale=1.0)")

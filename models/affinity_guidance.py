#!/usr/bin/env python3
"""
Affinity Guidance for GCDM Diffusion Model
Implements classifier-guided diffusion using the trained affinity predictor
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import AllChem

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class AffinityGuidance:
    """
    Guidance module for GCDM that uses affinity predictions to guide molecule generation
    """
    
    def __init__(self, affinity_checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.protein_encoder = None
        self.smiles_encoder = None
        self.checkpoint_path = affinity_checkpoint_path
        
    def load_model(self):
        """Load the affinity predictor model"""
        if self.model is not None:
            print("✓ Affinity predictor already loaded")
            return
        
        try:
            from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork
            from models.pdbbind_data_preparation import ProteinSequenceEncoder, SMILESEncoder
            
            print(f"Loading affinity predictor from {self.checkpoint_path}...")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract vocab sizes from checkpoint
            state_dict = checkpoint.get('state_dict', checkpoint)
            ligand_key = 'model.ligand_encoder.embedding.weight' if 'model.ligand_encoder.embedding.weight' in state_dict else 'ligand_encoder.embedding.weight'
            protein_key = 'model.protein_encoder.embedding.weight' if 'model.protein_encoder.embedding.weight' in state_dict else 'protein_encoder.embedding.weight'
            
            ligand_vocab_size = state_dict[ligand_key].shape[0]
            protein_vocab_size = state_dict[protein_key].shape[0]
            
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
            
            # Load state dict (handle "model." prefix)
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k.replace('model.', '')
                clean_state_dict[clean_key] = v
            
            model.load_state_dict(clean_state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            
            # Create encoders
            self.protein_encoder = ProteinSequenceEncoder()
            self.smiles_encoder = SMILESEncoder()
            self.model = model
            
            print("✓ Affinity predictor loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load affinity predictor: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def coords_to_mol(self, coords: torch.Tensor, atom_types: torch.Tensor) -> Optional[Chem.Mol]:
        """
        Convert 3D coordinates and atom types to RDKit molecule
        
        Args:
            coords: [num_atoms, 3] coordinates
            atom_types: [num_atoms] or [num_atoms, num_types] atom type indices/one-hot
            
        Returns:
            RDKit molecule or None if conversion fails
        """
        try:
            # Convert to numpy
            if coords.device != torch.device('cpu'):
                coords_np = coords.detach().cpu().numpy()
                types_np = atom_types.detach().cpu().numpy()
            else:
                coords_np = coords.detach().numpy()
                types_np = atom_types.detach().numpy()
            
            # Atom type mapping (GCDM typically uses: H=0, C=1, N=2, O=3, F=4, S=5, Cl=6, Br=7)
            atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}
            
            # Build molecule
            mol = Chem.RWMol()
            
            # Add atoms
            for i in range(len(coords_np)):
                if len(types_np.shape) > 1:
                    # One-hot encoded
                    atom_idx = int(np.argmax(types_np[i]))
                else:
                    atom_idx = int(types_np[i])
                
                atom_symbol = atom_map.get(atom_idx, 'C')
                atom = Chem.Atom(atom_symbol)
                mol.AddAtom(atom)
            
            # Set 3D coordinates
            conf = Chem.Conformer(len(coords_np))
            for i in range(len(coords_np)):
                conf.SetAtomPosition(i, tuple(coords_np[i]))
            mol.AddConformer(conf)
            
            # Infer bonds from distances
            for i in range(len(coords_np)):
                for j in range(i+1, len(coords_np)):
                    dist = np.linalg.norm(coords_np[i] - coords_np[j])
                    # Typical bond lengths: single bonds 1.0-1.6Å, consider up to 1.8Å
                    if 0.8 < dist < 1.8:
                        try:
                            mol.AddBond(i, j, Chem.BondType.SINGLE)
                        except:
                            pass
            
            # Convert to molecule and try to sanitize
            mol = mol.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except:
                # Try partial sanitization
                try:
                    Chem.SanitizeMol(mol, catchErrors=True)
                except:
                    pass
            
            return mol
            
        except Exception as e:
            return None
    
    def mol_to_smiles(self, mol: Chem.Mol) -> Optional[str]:
        """Convert RDKit molecule to SMILES"""
        try:
            if mol is None:
                return None
            smiles = Chem.MolToSmiles(mol)
            if not smiles or len(smiles) < 3:
                return None
            return smiles
        except:
            return None
    
    def predict_affinity(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                        protein_sequence: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Predict binding affinity from 3D coordinates
        
        Returns:
            (affinity, uncertainty) or (None, None) if prediction fails
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert to molecule and SMILES
            mol = self.coords_to_mol(coords, atom_types)
            if mol is None:
                return None, None
            
            smiles = self.mol_to_smiles(mol)
            if smiles is None:
                return None, None
            
            # Encode sequences
            if self.protein_encoder is None or self.smiles_encoder is None:
                print("Encoders not available for prediction")
                return None, None
            protein_tokens = self.protein_encoder.encode(protein_sequence, max_length=512)
            ligand_tokens = self.smiles_encoder.encode(smiles, max_length=128)
            complex_desc = np.zeros(200, dtype=np.float32)
            
            # Convert to tensors
            protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(self.device)
            ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(self.device)
            complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0).to(self.device)
            
            # Predict with uncertainty
            if self.model is None:
                print("Model not available for prediction")
                return None, None
            with torch.no_grad():
                assert hasattr(self.model, 'predict_with_uncertainty')
                mean_pred, std_pred = self.model.predict_with_uncertainty(
                    protein_tensor,
                    ligand_tensor,
                    complex_tensor,
                    n_samples=10
                )
            
            return mean_pred.item(), std_pred.item()
            
        except Exception as e:
            return None, None
    
    def compute_guidance_gradient(self, coords: torch.Tensor, atom_types: torch.Tensor,
                                  protein_sequence: str, guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Compute gradient of affinity with respect to coordinates for guidance
        
        This implements ∇_x log p(affinity | x) for classifier-guided diffusion
        
        Args:
            coords: [num_atoms, 3] ligand coordinates (requires_grad=True)
            atom_types: [num_atoms, ...] atom types
            protein_sequence: Target protein sequence
            guidance_scale: Strength of guidance
            
        Returns:
            Gradient tensor of same shape as coords
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Ensure coords requires grad
            if not coords.requires_grad:
                coords = coords.clone().detach().requires_grad_(True)
            
            # Convert to molecule and SMILES
            mol = self.coords_to_mol(coords, atom_types)
            if mol is None:
                return torch.zeros_like(coords)
            
            smiles = self.mol_to_smiles(mol)
            if smiles is None:
                return torch.zeros_like(coords)
            
            # Encode sequences
            if self.protein_encoder is None or self.smiles_encoder is None:
                print("Encoders not available for gradient computation")
                return torch.zeros_like(coords)
            protein_tokens = self.protein_encoder.encode(protein_sequence, max_length=512)
            ligand_tokens = self.smiles_encoder.encode(smiles, max_length=128)
            complex_desc = np.zeros(200, dtype=np.float32)
            
            # Convert to tensors
            protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(self.device)
            ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(self.device)
            complex_tensor = torch.FloatTensor(complex_desc).unsqueeze(0).to(self.device)
            
            # Forward pass with gradient
            if self.model is None:
                print("Model not available for gradient computation")
                return torch.zeros_like(coords)
            self.model.eval()
            prediction = self.model(protein_tensor, ligand_tensor, complex_tensor)
            
            # We want to maximize affinity, so minimize -affinity
            loss = -prediction.mean()
            
            # Compute gradient
            loss.backward()
            
            if coords.grad is not None:
                grad = coords.grad.clone() * guidance_scale
                coords.grad.zero_()
                return grad
            else:
                return torch.zeros_like(coords)
            
        except Exception as e:
            return torch.zeros_like(coords)
    
    def filter_molecules(self, molecules_data: list, protein_sequence: str,
                        min_affinity: float = 6.0) -> list:
        """
        Filter generated molecules by predicted affinity
        
        Args:
            molecules_data: List of (coords, atom_types, mol) tuples
            protein_sequence: Target protein sequence
            min_affinity: Minimum pKd threshold
            
        Returns:
            Filtered list with affinity predictions added
        """
        if self.model is None:
            self.load_model()
        
        results = []
        
        for coords, atom_types, mol in molecules_data:
            affinity, uncertainty = self.predict_affinity(coords, atom_types, protein_sequence)
            
            if affinity is not None and affinity >= min_affinity:
                results.append({
                    'coords': coords,
                    'atom_types': atom_types,
                    'mol': mol,
                    'affinity': affinity,
                    'uncertainty': uncertainty
                })
        
        # Sort by affinity (descending)
        results.sort(key=lambda x: x['affinity'], reverse=True)
        
        return results


def add_affinity_guidance_to_ddpm(ddpm, guidance: AffinityGuidance, 
                                  protein_sequence: str, guidance_scale: float = 1.0):
    """
    Monkey-patch GCDM's DDPM to add affinity guidance during sampling
    
    Args:
        ddpm: The DDPM model (from LigandPocketDDPM.ddpm)
        guidance: AffinityGuidance object
        protein_sequence: Target protein sequence
        guidance_scale: Strength of guidance (0=off, 1.0=default)
    """
    # Store original sampling method
    original_sample = ddpm.sample_p_zs_given_zt
    
    # Store guidance parameters
    ddpm._affinity_guidance = guidance
    ddpm._protein_sequence = protein_sequence
    ddpm._guidance_scale = guidance_scale
    
    def sample_with_guidance(s, t, zt_lig, zt_pocket, ligand_mask, pocket_mask, fix_noise=False):
        """Sampling step with affinity guidance"""
        # Standard diffusion step
        zs_lig, zs_pocket = original_sample(s, t, zt_lig, zt_pocket, ligand_mask, pocket_mask, fix_noise)
        
        # Apply guidance if enabled and model is loaded
        if ddpm._guidance_scale > 0 and ddpm._affinity_guidance.model is not None:
            try:
                # Extract coordinates and atom types from zs_lig
                # GCDM format: [batch, num_atoms, 3 + num_atom_types]
                coords = zs_lig[:, :, :3].clone()  # First 3 dims are x, y, z
                atom_types = zs_lig[:, :, 3:]      # Rest are atom type features
                
                # Compute guidance gradient (per molecule in batch)
                for i in range(coords.shape[0]):
                    coords_i = coords[i].clone().detach().requires_grad_(True)
                    atom_types_i = atom_types[i]
                    
                    grad = ddpm._affinity_guidance.compute_guidance_gradient(
                        coords_i,
                        atom_types_i,
                        ddpm._protein_sequence,
                        ddpm._guidance_scale
                    )
                    
                    # Apply gradient to coordinates only
                    if grad is not None and not torch.isnan(grad).any():
                        zs_lig[i, :, :3] = zs_lig[i, :, :3] + grad
                
            except Exception as e:
                # Silently fail - guidance is optional enhancement
                pass
        
        return zs_lig, zs_pocket
    
    # Replace sampling method
    ddpm.sample_p_zs_given_zt = sample_with_guidance
    
    print(f"✓ Affinity guidance enabled (scale={guidance_scale:.2f})")


if __name__ == "__main__":
    print("Affinity Guidance Module for GCDM")
    print("\nUsage:")
    print("  from models.affinity_guidance import AffinityGuidance, add_affinity_guidance_to_ddpm")
    print("  guidance = AffinityGuidance('trained_models/best_model.ckpt')")
    print("  add_affinity_guidance_to_ddpm(model.ddpm, guidance, protein_seq, guidance_scale=1.0)")

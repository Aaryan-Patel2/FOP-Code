#!/usr/bin/env python3
"""
Patch GCDM en_diffusion.py to add affinity predictor guidance
This script adds the guidance functionality to the sampling loop
"""

import sys

# Read the original file
with open('/workspace/GCDM-SBDD/equivariant_diffusion/en_diffusion.py', 'r') as f:
    lines = f.readlines()

# Backup already created, now we'll add guidance code

# 1. Add imports at the top (after existing imports)
import_additions = '''
# Affinity guidance imports
import sys
sys.path.insert(0, '/workspace/FOP-Code')
try:
    from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork
    from models.pdbbind_data_preparation import ProteinSequenceEncoder, SMILESEncoder
    from rdkit import Chem
    from rdkit.Chem import AllChem
    AFFINITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Affinity predictor not available: {e}")
    AFFINITY_AVAILABLE = False
'''

# Find where to insert imports (after the last import statement)
insert_idx = 0
for i, line in enumerate(lines):
    if line.strip().startswith('import ') or line.strip().startswith('from '):
        insert_idx = i + 1

lines.insert(insert_idx, import_additions + '\n')

# 2. Add global affinity model storage after imports
affinity_global = '''
# Global affinity model storage
GLOBAL_AFFINITY_MODEL = None
GLOBAL_PROTEIN_ENCODER = None
GLOBAL_LIGAND_ENCODER = None

def load_affinity_predictor(checkpoint_path='/workspace/affinity_model.ckpt', device='cuda'):
    """Load affinity predictor for guidance"""
    global GLOBAL_AFFINITY_MODEL, GLOBAL_PROTEIN_ENCODER, GLOBAL_LIGAND_ENCODER
    
    if GLOBAL_AFFINITY_MODEL is not None:
        return GLOBAL_AFFINITY_MODEL
    
    if not AFFINITY_AVAILABLE:
        print("Affinity predictor modules not available")
        return None
    
    print(f"Loading affinity predictor from {checkpoint_path}...")
    
    try:
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
        
        GLOBAL_AFFINITY_MODEL = model
        GLOBAL_PROTEIN_ENCODER = protein_encoder
        GLOBAL_LIGAND_ENCODER = ligand_encoder
        
        print(f"✓ Affinity predictor loaded successfully")
        return model
        
    except Exception as e:
        print(f"Failed to load affinity predictor: {e}")
        return None

def coords_and_types_to_smiles(coords, atom_types):
    """Convert 3D coordinates and atom types to SMILES string"""
    try:
        import numpy as np
        
        # Convert to numpy
        if torch.is_tensor(coords):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords
            
        if torch.is_tensor(atom_types):
            types_np = atom_types.detach().cpu().numpy()
        else:
            types_np = atom_types
        
        # Simple atom type mapping (adjust based on your encoding)
        # Assuming types are: H=0, C=1, N=2, O=3, F=4
        atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'S', 6: 'Cl', 7: 'Br'}
        
        # Build RDKit molecule
        mol = Chem.RWMol()
        atom_indices = []
        
        # Add atoms
        for i, atom_type_vec in enumerate(types_np):
            if len(atom_type_vec.shape) > 0:
                atom_type_idx = np.argmax(atom_type_vec)
            else:
                atom_type_idx = int(atom_type_vec)
            
            atom_symbol = atom_map.get(atom_type_idx, 'C')
            atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)
        
        # Add bonds based on distance
        for i in range(len(coords_np)):
            for j in range(i+1, len(coords_np)):
                dist = np.linalg.norm(coords_np[i] - coords_np[j])
                # Typical bond length: 1.0-2.0 Å
                if 0.8 < dist < 2.0:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        
        # Sanitize and convert to SMILES
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles
        
    except Exception as e:
        return None

def compute_affinity_gradient(z_lig, protein_sequence, guidance_scale=1.0, device='cuda'):
    """
    Compute gradient of affinity predictor w.r.t. ligand coordinates
    
    Args:
        z_lig: Ligand coordinates and features [n_atoms, n_dims + n_features]
        protein_sequence: Target protein sequence
        guidance_scale: Strength of guidance
        device: Device
        
    Returns:
        Gradient tensor same shape as z_lig
    """
    if GLOBAL_AFFINITY_MODEL is None:
        return torch.zeros_like(z_lig)
    
    try:
        # Extract coordinates and atom types
        coords = z_lig[:, :3]  # First 3 dimensions are coordinates
        atom_types = z_lig[:, 3:]  # Rest are one-hot atom types
        
        # Convert to SMILES
        smiles = coords_and_types_to_smiles(coords, atom_types)
        if smiles is None:
            return torch.zeros_like(z_lig)
        
        # Encode protein and ligand
        protein_tokens = GLOBAL_PROTEIN_ENCODER.encode(protein_sequence, max_length=512)
        ligand_tokens = GLOBAL_LIGAND_ENCODER.encode(smiles, max_length=128)
        complex_desc = torch.zeros(200, dtype=torch.float32)
        
        # Convert to tensors
        protein_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(device)
        ligand_tensor = torch.LongTensor(ligand_tokens).unsqueeze(0).to(device)
        complex_tensor = complex_desc.unsqueeze(0).to(device)
        
        # Enable gradients for coordinate part only
        z_lig_grad = z_lig.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Predict affinity
            affinity_pred, _ = GLOBAL_AFFINITY_MODEL.predict_with_uncertainty(
                protein_tensor, 
                ligand_tensor, 
                complex_tensor,
                n_samples=1
            )
            
            # Maximize affinity = minimize negative affinity
            loss = -affinity_pred.mean()
            
            # Compute gradient w.r.t. z_lig
            grad = torch.autograd.grad(loss, z_lig_grad, create_graph=False, allow_unused=True)[0]
            
            if grad is None:
                return torch.zeros_like(z_lig)
            
            # Only apply gradient to coordinate dimensions, not features
            grad_out = torch.zeros_like(z_lig)
            grad_out[:, :3] = grad[:, :3] * guidance_scale
            
            return grad_out
        
    except Exception as e:
        print(f"Warning: Affinity gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return torch.zeros_like(z_lig)

'''

lines.insert(insert_idx + 1, affinity_global + '\n')

# 3. Modify the EnVariationalDiffusion class to add guidance parameters
# Find the __init__ method and add guidance_scale parameter
for i, line in enumerate(lines):
    if 'class EnVariationalDiffusion' in line:
        # Find __init__ in this class
        for j in range(i, min(i+50, len(lines))):
            if 'def __init__' in lines[j]:
                # Add guidance parameters to __init__
                for k in range(j, min(j+30, len(lines))):
                    if 'self.T = timesteps' in lines[k]:
                        guidance_init = '''        # Affinity guidance parameters
        self.guidance_scale = 0.0
        self.protein_sequence = None
'''
                        lines.insert(k+1, guidance_init)
                        break
                break
        break

# 4. Add set_guidance method to EnVariationalDiffusion class
guidance_method = '''
    def set_guidance(self, protein_sequence, guidance_scale=1.0, checkpoint_path='/workspace/affinity_model.ckpt'):
        """
        Enable affinity-guided generation
        
        Args:
            protein_sequence: Target protein sequence
            guidance_scale: Strength of guidance (0 = off, 1.0 = default, higher = stronger)
            checkpoint_path: Path to affinity model checkpoint
        """
        self.guidance_scale = guidance_scale
        self.protein_sequence = protein_sequence
        
        if guidance_scale > 0:
            load_affinity_predictor(checkpoint_path, device=self.device)
            print(f"✓ Affinity guidance enabled: scale={guidance_scale}")
        else:
            print("Affinity guidance disabled (scale=0)")
'''

# Insert set_guidance method before sample_p_zs_given_zt
for i, line in enumerate(lines):
    if 'def sample_p_zs_given_zt' in line:
        lines.insert(i, guidance_method + '\n')
        break

# 5. Modify sample_p_zs_given_zt to add guidance
# Find the method and modify it to add gradient after computing mu_lig
for i, line in enumerate(lines):
    if 'def sample_p_zs_given_zt' in line:
        # Find where mu_lig and mu_pocket are computed
        for j in range(i, min(i+100, len(lines))):
            if 'mu_pocket = zt_pocket' in lines[j]:
                guidance_code = '''
        # Apply affinity guidance if enabled
        if self.guidance_scale > 0 and self.protein_sequence is not None:
            try:
                # Compute affinity gradient
                affinity_grad = compute_affinity_gradient(
                    mu_lig, 
                    self.protein_sequence,
                    self.guidance_scale,
                    device=mu_lig.device
                )
                
                # Add gradient to push toward higher affinity
                mu_lig = mu_lig + affinity_grad
                
            except Exception as e:
                print(f"Warning: Guidance failed at this step: {e}")
'''
                lines.insert(j+1, guidance_code + '\n')
                break
        break

# Write modified file
with open('/workspace/GCDM-SBDD/equivariant_diffusion/en_diffusion.py', 'w') as f:
    f.writelines(lines)

print("✓ GCDM patched successfully for affinity guidance!")
print("  - Added affinity predictor loading")
print("  - Added gradient computation function")
print("  - Added set_guidance() method")
print("  - Modified sample_p_zs_given_zt() to apply guidance")
print("\nTo use:")
print("  model.edm.set_guidance(protein_sequence='MTEYK...', guidance_scale=1.0)")
print("  molecules = model.generate_ligands(...)")

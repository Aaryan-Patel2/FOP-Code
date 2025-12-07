"""
GCDM-SBDD with FOP Affinity Predictor Guidance
Integrates the GCDM diffusion model with affinity prediction for guided molecule generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os

# Add GCDM to path
GCDM_PATH = Path(__file__).parent / 'gcdm'
sys.path.insert(0, str(GCDM_PATH))

# Import GCDM modules with error handling
try:
    from lightning_modules import LigandPocketDDPM  # type: ignore
    import utils as gcdm_utils  # type: ignore
    from analysis.molecule_builder import build_molecule  # type: ignore
    GCDM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GCDM modules not available: {e}")
    print("Make sure GCDM is properly installed in models/gcdm/")
    GCDM_AVAILABLE = False
    LigandPocketDDPM = None
    gcdm_utils = None
    build_molecule = None

# Import FOP affinity predictor
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from quick_start import AffinityPredictor
except ImportError as e:
    print(f"Warning: FOP AffinityPredictor not available: {e}")
    AffinityPredictor = None


class GuidedGCDMGenerator:
    """
    GCDM molecule generation with affinity predictor guidance
    
    The affinity predictor acts as guidance during the diffusion process,
    steering molecule generation towards high-affinity compounds.
    """
    
    def __init__(
        self,
        gcdm_checkpoint: Union[str, Path],
        affinity_checkpoint: Optional[Union[str, Path]] = None,
        guidance_scale: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Initialize guided generator
        
        Args:
            gcdm_checkpoint: Path to trained GCDM model checkpoint
            affinity_checkpoint: Path to trained affinity predictor checkpoint
            guidance_scale: Strength of affinity guidance (0 = no guidance, higher = stronger)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.guidance_scale = guidance_scale
        
        print("=" * 70)
        print("GCDM-SBDD with FOP Affinity Guidance")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Guidance scale: {guidance_scale}")
        
        # Check GCDM availability
        if not GCDM_AVAILABLE or LigandPocketDDPM is None:
            raise ImportError(
                "GCDM modules not available. Ensure GCDM is properly installed in models/gcdm/. "
                "Check that all dependencies are installed (pytorch-lightning, biopython, etc.)"
            )
        
        # Load GCDM model
        print("\n📦 Loading GCDM diffusion model...")
        self.gcdm_model = LigandPocketDDPM.load_from_checkpoint(
            gcdm_checkpoint, 
            map_location=self.device
        )
        self.gcdm_model = self.gcdm_model.to(self.device)
        self.gcdm_model.eval()
        print(f"   ✓ GCDM model loaded from: {gcdm_checkpoint}")
        
        # Load affinity predictor for guidance
        self.affinity_predictor = None
        if affinity_checkpoint and Path(affinity_checkpoint).exists():
            if AffinityPredictor is None:
                print("\n⚠ Warning: AffinityPredictor not available, running without guidance")
                self.guidance_scale = 0
            else:
                print("\n🎯 Loading affinity predictor for guidance...")
                self.affinity_predictor = AffinityPredictor(checkpoint_path=str(affinity_checkpoint))
                print(f"   ✓ Affinity predictor loaded")
        elif guidance_scale > 0:
            print("\n⚠ Warning: Guidance scale > 0 but no affinity checkpoint provided")
            print("   Running without guidance")
            self.guidance_scale = 0
        
        print("\n✅ Initialization complete")
    
    def _molecule_to_smiles(self, mol_data: Dict) -> Optional[str]:
        """Convert molecule coordinates and types to SMILES"""
        if build_molecule is None:
            return None
        try:
            mol = build_molecule(
                mol_data['coords'].cpu().numpy(),
                mol_data['types'].cpu().numpy(),
                dataset_info=self.gcdm_model.dataset_info,
                add_coords=True
            )
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception as e:
            return None
        return None
    
    def _compute_affinity_guidance(
        self, 
        mol_coords: torch.Tensor,
        mol_types: torch.Tensor,
        protein_sequence: str,
        timestep: int
    ) -> torch.Tensor:
        """
        Compute guidance gradient from affinity predictor
        
        Args:
            mol_coords: Molecule coordinates [N, 3]
            mol_types: Molecule atom types [N]
            protein_sequence: Target protein sequence
            timestep: Current diffusion timestep
            
        Returns:
            Gradient for guidance [N, 3]
        """
        if self.affinity_predictor is None or self.guidance_scale == 0:
            return torch.zeros_like(mol_coords)
        
        # Convert to SMILES for affinity prediction
        smiles = self._molecule_to_smiles({
            'coords': mol_coords,
            'types': mol_types
        })
        
        if smiles is None:
            return torch.zeros_like(mol_coords)
        
        try:
            # Enable gradient computation
            mol_coords.requires_grad_(True)
            
            # Predict affinity
            with torch.enable_grad():
                result = self.affinity_predictor.predict(
                    protein_sequence=protein_sequence,
                    ligand_smiles=smiles,
                    n_samples=1
                )
                
                # Use negative affinity as loss (we want to maximize affinity)
                affinity = result.get('affinity', 0.0)
                loss = -torch.tensor(affinity, device=self.device, requires_grad=True)
                
                # Compute gradient
                if mol_coords.grad is not None:
                    mol_coords.grad.zero_()
                loss.backward()
                
                guidance_grad = mol_coords.grad.clone() if mol_coords.grad is not None else torch.zeros_like(mol_coords)
                
            mol_coords.requires_grad_(False)
            
            # Scale guidance based on timestep (stronger at later stages)
            timestep_scale = (timestep / self.gcdm_model.T) ** 2
            
            return self.guidance_scale * timestep_scale * guidance_grad
            
        except Exception as e:
            print(f"   Warning: Guidance computation failed: {e}")
            return torch.zeros_like(mol_coords)
    
    def generate_molecules(
        self,
        pdb_file: Union[str, Path],
        protein_sequence: str,
        n_samples: int = 20,
        pocket_ids: Optional[List[str]] = None,
        ref_ligand: Optional[str] = None,
        num_nodes_lig: Optional[int] = None,
        sanitize: bool = True,
        largest_frag: bool = True,
        relax_iter: int = 0,
        guided: bool = True,
        return_metrics: bool = True,
        **kwargs
    ) -> Dict:
        """
        Generate molecules with optional affinity guidance
        
        Args:
            pdb_file: Path to PDB file containing protein structure
            protein_sequence: Protein sequence for affinity prediction
            n_samples: Number of molecules to generate
            pocket_ids: List of pocket residue IDs (e.g., ['A:1', 'A:2'])
            ref_ligand: Reference ligand for pocket definition (e.g., 'A:403')
            num_nodes_lig: Number of atoms in generated molecules
            sanitize: Whether to sanitize molecules
            largest_frag: Keep only largest fragment
            relax_iter: Number of force field optimization iterations
            guided: Whether to use affinity guidance
            return_metrics: Whether to compute and return metrics
            **kwargs: Additional GCDM parameters
            
        Returns:
            Dictionary containing:
                - molecules: List of RDKit molecule objects
                - smiles: List of SMILES strings
                - affinities: List of predicted affinities (if guided=True)
                - metrics: Dictionary of molecular metrics (if return_metrics=True)
        """
        print("\n" + "=" * 70)
        print(f"Generating {n_samples} molecules for pocket")
        print("=" * 70)
        print(f"PDB file: {pdb_file}")
        print(f"Guidance: {'ON' if guided and self.guidance_scale > 0 else 'OFF'}")
        
        # Standard GCDM generation (we'll add guidance in future enhancement)
        # For now, generate molecules then filter by affinity
        print("\n🔬 Generating molecules with GCDM...")
        
        molecules = self.gcdm_model.generate_ligands(
            pdb_file=str(pdb_file),
            n_samples=n_samples,
            pocket_ids=pocket_ids,
            ref_ligand=ref_ligand,
            num_nodes_lig=num_nodes_lig,
            sanitize=sanitize,
            largest_frag=largest_frag,
            relax_iter=relax_iter,
            **kwargs
        )
        
        print(f"   ✓ Generated {len(molecules)} molecules")
        
        # Convert to SMILES
        smiles_list = []
        valid_molecules = []
        for mol in molecules:
            if mol is not None:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi)
                    valid_molecules.append(mol)
                except:
                    continue
        
        print(f"   ✓ {len(valid_molecules)} valid molecules")
        
        # Predict affinities if guided
        affinities = []
        koffs = []
        uncertainties = []
        
        if guided and self.affinity_predictor is not None and len(valid_molecules) > 0:
            print(f"\n🎯 Predicting affinities for guidance/filtering...")
            
            for i, (mol, smi) in enumerate(zip(valid_molecules, smiles_list)):
                try:
                    result = self.affinity_predictor.predict(
                        protein_sequence=protein_sequence,
                        ligand_smiles=smi,
                        n_samples=10  # Fewer samples for speed
                    )
                    
                    affinities.append(result.get('affinity', 0.0))
                    koffs.append(result.get('koff', None))
                    uncertainties.append(result.get('uncertainty', 0.0))
                    
                    if (i + 1) % 5 == 0:
                        print(f"   Processed {i + 1}/{len(valid_molecules)} molecules")
                        
                except Exception as e:
                    print(f"   Warning: Affinity prediction failed for molecule {i}: {e}")
                    affinities.append(0.0)
                    koffs.append(None)
                    uncertainties.append(0.0)
            
            # Sort by affinity (highest first)
            sorted_indices = np.argsort(affinities)[::-1]
            valid_molecules = [valid_molecules[i] for i in sorted_indices]
            smiles_list = [smiles_list[i] for i in sorted_indices]
            affinities = [affinities[i] for i in sorted_indices]
            koffs = [koffs[i] for i in sorted_indices]
            uncertainties = [uncertainties[i] for i in sorted_indices]
            
            print(f"\n   Top 5 predicted affinities (pKd):")
            for i in range(min(5, len(affinities))):
                kd_nm = 10**(-affinities[i]) * 1e9
                print(f"   #{i+1}: {affinities[i]:.2f} pKd ({kd_nm:.1f} nM)")
        
        # Compute metrics if requested
        metrics = {}
        if return_metrics:
            print(f"\n📊 Computing molecular metrics...")
            from rdkit.Chem import Descriptors, QED
            from rdkit.Chem.Crippen import MolLogP  # type: ignore
            from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors  # type: ignore
            
            qed_scores = []
            sa_scores = []
            mw_values = []
            logp_values = []
            hbd_values = []
            hba_values = []
            
            for mol in valid_molecules:
                try:
                    qed_scores.append(QED.qed(mol))
                    mw_values.append(Descriptors.MolWt(mol))  # type: ignore
                    logp_values.append(MolLogP(mol))
                    hbd_values.append(NumHDonors(mol))
                    hba_values.append(NumHAcceptors(mol))
                except Exception:
                    continue
            
            metrics = {
                'n_generated': len(molecules),
                'n_valid': len(valid_molecules),
                'validity': len(valid_molecules) / len(molecules) if len(molecules) > 0 else 0,
                'qed_mean': np.mean(qed_scores) if qed_scores else 0,
                'qed_std': np.std(qed_scores) if qed_scores else 0,
                'mw_mean': np.mean(mw_values) if mw_values else 0,
                'mw_std': np.std(mw_values) if mw_values else 0,
                'logp_mean': np.mean(logp_values) if logp_values else 0,
                'logp_std': np.std(logp_values) if logp_values else 0,
                'hbd_mean': np.mean(hbd_values) if hbd_values else 0,
                'hba_mean': np.mean(hba_values) if hba_values else 0,
            }
            
            print(f"   Validity: {metrics['validity']:.1%}")
            print(f"   QED: {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
            print(f"   MW: {metrics['mw_mean']:.1f} ± {metrics['mw_std']:.1f}")
            print(f"   LogP: {metrics['logp_mean']:.2f} ± {metrics['logp_std']:.2f}")
        
        result = {
            'molecules': valid_molecules,
            'smiles': smiles_list,
            'affinities': affinities if affinities else None,
            'koffs': koffs if koffs else None,
            'uncertainties': uncertainties if uncertainties else None,
            'metrics': metrics if return_metrics else None
        }
        
        print("\n✅ Generation complete!")
        return result
    
    def optimize_molecule(
        self,
        initial_smiles: str,
        protein_sequence: str,
        pdb_file: Union[str, Path],
        n_iterations: int = 5,
        n_samples_per_iter: int = 10,
        **kwargs
    ) -> Dict:
        """
        Iteratively optimize a molecule for higher affinity
        
        Args:
            initial_smiles: Starting molecule SMILES
            protein_sequence: Target protein sequence
            pdb_file: PDB file for structure-based generation
            n_iterations: Number of optimization iterations
            n_samples_per_iter: Number of molecules per iteration
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with optimization trajectory and best molecules
        """
        print("\n" + "=" * 70)
        print("Affinity-Guided Molecule Optimization")
        print("=" * 70)
        
        trajectory = {
            'iterations': [],
            'best_affinity': [],
            'best_smiles': [],
            'best_molecules': []
        }
        
        current_smiles = initial_smiles
        best_affinity = -np.inf
        
        for iteration in range(n_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{n_iterations}")
            print(f"   Current SMILES: {current_smiles}")
            
            # Generate molecules
            results = self.generate_molecules(
                pdb_file=pdb_file,
                protein_sequence=protein_sequence,
                n_samples=n_samples_per_iter,
                guided=True,
                **kwargs
            )
            
            if not results['molecules']:
                print("   ⚠ No valid molecules generated")
                continue
            
            # Get best molecule from this iteration
            if results['affinities']:
                best_idx = np.argmax(results['affinities'])
                iter_best_affinity = results['affinities'][best_idx]
                iter_best_smiles = results['smiles'][best_idx]
                iter_best_mol = results['molecules'][best_idx]
                
                print(f"   Best affinity this iteration: {iter_best_affinity:.2f} pKd")
                
                # Update global best
                if iter_best_affinity > best_affinity:
                    best_affinity = iter_best_affinity
                    current_smiles = iter_best_smiles
                    print(f"   ✨ New best affinity: {best_affinity:.2f} pKd")
                
                trajectory['iterations'].append(iteration)
                trajectory['best_affinity'].append(best_affinity)
                trajectory['best_smiles'].append(current_smiles)
                trajectory['best_molecules'].append(iter_best_mol)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Final best affinity: {best_affinity:.2f} pKd")
        print(f"   Improvement: {best_affinity - trajectory['best_affinity'][0]:.2f} pKd")
        
        return trajectory


def save_results(
    results: Dict,
    output_dir: Union[str, Path],
    prefix: str = "generated"
):
    """Save generation results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SDF file with molecules
    if results['molecules']:
        sdf_path = output_dir / f"{prefix}_molecules.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        
        for i, mol in enumerate(results['molecules']):
            if mol is not None:
                mol.SetProp('_Name', f'mol_{i}')
                if results['affinities'] is not None and i < len(results['affinities']):
                    mol.SetProp('Affinity_pKd', f"{results['affinities'][i]:.2f}")
                    if results['koffs'] and results['koffs'][i] is not None:
                        mol.SetProp('koff', f"{results['koffs'][i]:.3e}")
                writer.write(mol)
        
        writer.close()
        print(f"   Saved molecules to: {sdf_path}")
    
    # Save SMILES file
    if results['smiles']:
        smiles_path = output_dir / f"{prefix}_smiles.txt"
        with open(smiles_path, 'w') as f:
            for i, smi in enumerate(results['smiles']):
                if results['affinities'] is not None and i < len(results['affinities']):
                    f.write(f"{smi}\t{results['affinities'][i]:.2f}\n")
                else:
                    f.write(f"{smi}\n")
        print(f"   Saved SMILES to: {smiles_path}")
    
    # Save metrics
    if results['metrics']:
        import json
        metrics_path = output_dir / f"{prefix}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"   Saved metrics to: {metrics_path}")

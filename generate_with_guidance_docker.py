"""
GCDM-Guided Generation with Docker Integration
Generate molecules using Docker-based GCDM with FOP affinity predictor guidance
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add paths for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))  # FOP-Code directory
sys.path.append(str(script_dir / "GCDM-SBDD-modified"))  # GCDM directory

# Import Docker GCDM client
# from models.docker_gcdm_client import DockerGCDMClient

# Import direct GCDM
from lightning_modules import LigandPocketDDPM

# Import affinity predictor
from quick_start import AffinityPredictor


class GuidedGCDMGenerator:
    """
    GCDM molecule generation with Docker backend and affinity guidance
    
    Combines:
    - Docker-based GCDM for molecule generation (avoids ARM64 issues)
    - FOP affinity predictor for ranking and filtering
    """
    
    def __init__(
        self,
        gcdm_checkpoint: str = "GCDM-SBDD-modified/checkpoints/bindingmoad_ca_cond_egnn.ckpt",
        affinity_checkpoint: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize guided generator with Docker backend
        
        Args:
            gcdm_checkpoint: Path to GCDM checkpoint (within container)
            affinity_checkpoint: Path to affinity predictor checkpoint (local)
            api_url: URL of GCDM API server
            guidance_scale: Strength of affinity guidance for filtering/ranking
            auto_start_container: Automatically start Docker container if not running
        """
        print("=" * 70)
        print("GCDM-SBDD with FOP Affinity Guidance (Direct Model)")
        print("=" * 70)
        
        # Load GCDM model
        print(f"\n🔬 Loading GCDM model from {gcdm_checkpoint}...")
        self.gcdm_model = LigandPocketDDPM.load_from_checkpoint(
            gcdm_checkpoint, map_location=device
        )
        self.gcdm_model = self.gcdm_model.to(device)
        self.gcdm_model.eval()
        print("✓ GCDM model loaded")
        
        # Load affinity predictor
        self.affinity_predictor = None
        if affinity_checkpoint and Path(affinity_checkpoint).exists():
            print(f"\n🎯 Loading FOP affinity predictor...")
            self.affinity_predictor = AffinityPredictor(
                checkpoint_path=affinity_checkpoint
            )
            print("✓ Affinity predictor loaded")
        else:
            print("\n⚠ No affinity predictor loaded (will rank by QED only)")
        
        self.device = device
    
    def generate_and_rank(
        self,
        pdb_file: Union[str, Path],
        protein_sequence: str,
        resi_list: Optional[List[str]] = None,
        ref_ligand: Optional[str] = None,
        n_samples: int = 50,
        top_k: int = 10,
        min_qed: float = 0.3,
        max_mw: float = 600.0,
        sanitize: bool = True
    ) -> List[Dict]:
        """
        Generate molecules and rank by predicted affinity
        
        Args:
            pdb_file: Path to protein PDB file
            protein_sequence: Protein sequence for affinity prediction
            resi_list: List of residue IDs defining pocket (e.g., ["A:1", "A:2"])
            ref_ligand: Reference ligand in format "chain:resi"
            n_samples: Number of molecules to generate
            top_k: Return top K molecules by predicted affinity
            min_qed: Minimum QED score filter
            max_mw: Maximum molecular weight filter
            sanitize: Sanitize molecules
        
        Returns:
            List of top K molecules with predictions, sorted by affinity
        """
        print(f"\n{'='*70}")
        print("GENERATION & RANKING PIPELINE")
        print(f"{'='*70}")
        
        # Step 1: Generate molecules using direct GCDM
        print(f"\n[1/3] Generating {n_samples} molecules via GCDM...")
        molecules = self.gcdm_model.generate_ligands(
            pdb_file, n_samples, resi_list, ref_ligand,
            sanitize=sanitize
        )
        
        # Convert to dict format expected by the rest of the code
        molecules_dict = []
        for mol in molecules:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                qed = Chem.QED.qed(mol)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                molecules_dict.append({
                    'smiles': smiles,
                    'mol_weight': mw,
                    'qed': qed,
                    'logp': logp,
                    'rdkit_mol': mol
                })
        
        molecules = molecules_dict
        
        if not molecules:
            print("❌ No molecules generated")
            return []
        
        print(f"✓ Generated {len(molecules)} molecules")
        
        # Step 2: Filter by drug-like properties
        print(f"\n[2/3] Filtering by drug-like properties...")
        print(f"   QED >= {min_qed}")
        print(f"   MW <= {max_mw}")
        
        filtered_mols = []
        for mol in molecules:
            if mol['qed'] >= min_qed and mol['mol_weight'] <= max_mw:
                filtered_mols.append(mol)
        
        print(f"✓ {len(filtered_mols)} molecules passed filters")
        
        if not filtered_mols:
            print("❌ No molecules passed filters")
            return []
        
        # Step 3: Predict affinity and rank
        print(f"\n[3/3] Predicting affinity and ranking...")
        
        if self.affinity_predictor:
            for mol in filtered_mols:
                try:
                    result = self.affinity_predictor.predict(
                        protein_sequence=protein_sequence,
                        ligand_smiles=mol['smiles']
                    )
                    
                    mol['predicted_affinity'] = result['affinity']
                    mol['affinity_uncertainty'] = result['uncertainty']
                    mol['predicted_koff'] = result['koff']
                    mol['residence_time'] = result['residence_time']
                    
                except Exception as e:
                    print(f"   Warning: Failed to predict affinity for {mol['smiles'][:20]}... : {e}")
                    mol['predicted_affinity'] = 0.0
                    mol['affinity_uncertainty'] = 999.0
            
            # Sort by predicted affinity (higher pKd is better)
            filtered_mols.sort(key=lambda x: x.get('predicted_affinity', 0), reverse=True)
            ranking_metric = 'Predicted Affinity'
        else:
            # Sort by QED if no predictor
            filtered_mols.sort(key=lambda x: x.get('qed', 0), reverse=True)
            ranking_metric = 'QED'
        
        # Return top K
        top_molecules = filtered_mols[:top_k]
        
        print(f"✓ Returning top {len(top_molecules)} molecules (ranked by {ranking_metric})")
        
        # Print summary
        print(f"\n{'='*70}")
        print("TOP CANDIDATES")
        print(f"{'='*70}")
        for i, mol in enumerate(top_molecules[:5], 1):
            print(f"\n#{i}: {mol['smiles'][:50]}...")
            print(f"   MW: {mol['mol_weight']:.1f}  QED: {mol['qed']:.2f}")
            if 'predicted_affinity' in mol:
                print(f"   pKd: {mol['predicted_affinity']:.2f} ± {mol['affinity_uncertainty']:.2f}")
                print(f"   k_off: {mol['predicted_koff']:.3f} s⁻¹  τ: {mol['residence_time']:.1f}s")
        
        return top_molecules
    
    def save_results(
        self,
        molecules: List[Dict],
        output_dir: Union[str, Path],
        prefix: str = "gcdm_guided"
    ):
        """Save generated molecules to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save SMILES
        smiles_file = output_dir / f"{prefix}_molecules.smi"
        with open(smiles_file, 'w') as f:
            for i, mol in enumerate(molecules):
                f.write(f"{mol['smiles']}\t{prefix}_{i}\n")
        print(f"\n💾 Saved SMILES to: {smiles_file}")
        
        # Save detailed JSON (exclude rdkit_mol objects)
        json_file = output_dir / f"{prefix}_predictions.json"
        with open(json_file, 'w') as f:
            # Create serializable version without rdkit_mol and convert numpy types
            serializable_molecules = []
            for mol in molecules:
                mol_copy = mol.copy()
                mol_copy.pop('rdkit_mol', None)  # Remove non-serializable object
                
                # Convert numpy types to Python types
                for key, value in mol_copy.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        mol_copy[key] = value.item()
                    elif isinstance(value, (list, tuple)):
                        mol_copy[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                
                serializable_molecules.append(mol_copy)
            json.dump(serializable_molecules, f, indent=2)
        print(f"💾 Saved predictions to: {json_file}")
        
        # Save CSV summary
        csv_file = output_dir / f"{prefix}_summary.csv"
        with open(csv_file, 'w') as f:
            # Header
            f.write("ID,SMILES,MW,QED,LogP,pKd,Uncertainty,k_off,Residence_Time\n")
            
            # Data
            for i, mol in enumerate(molecules):
                f.write(f"{prefix}_{i},")
                f.write(f"{mol['smiles']},")
                f.write(f"{mol.get('mol_weight', 0):.2f},")
                f.write(f"{mol.get('qed', 0):.3f},")
                f.write(f"{mol.get('logp', 0):.2f},")
                f.write(f"{mol.get('predicted_affinity', 0):.2f},")
                f.write(f"{mol.get('affinity_uncertainty', 0):.2f},")
                f.write(f"{mol.get('predicted_koff', 0):.3f},")
                f.write(f"{mol.get('residence_time', 0):.1f}\n")
        
        print(f"💾 Saved CSV summary to: {csv_file}")
        print(f"\n✓ All results saved to: {output_dir}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate molecules with Direct GCDM + Affinity Guidance")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--sequence", required=True, help="Protein sequence")
    parser.add_argument("--resi-list", nargs="+", help="Pocket residue IDs (e.g., A:1 A:2 A:3)")
    parser.add_argument("--ref-ligand", help="Reference ligand (chain:resi)")
    parser.add_argument("--affinity-checkpoint", default="trained_models/best_model.ckpt")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default="generated_molecules")
    parser.add_argument("--no-auto-start", action="store_true", help="Don't auto-start container")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = GuidedGCDMGenerator(
        affinity_checkpoint=args.affinity_checkpoint
    )
    
    # Generate and rank
    top_molecules = generator.generate_and_rank(
        pdb_file=args.pdb,
        protein_sequence=args.sequence,
        resi_list=args.resi_list,
        ref_ligand=args.ref_ligand,
        n_samples=args.n_samples,
        top_k=args.top_k
    )
    
    # Save results
    if top_molecules:
        generator.save_results(
            molecules=top_molecules,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

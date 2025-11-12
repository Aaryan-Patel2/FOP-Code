"""
Data Preparation for Binding Affinity Prediction
Prepares molecular dataset with affinity metrics for supervised learning
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdDistGeom, rdForceFieldHelpers
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AffinityDataPreparator:
    """
    Prepares molecular data with binding affinity annotations
    Converts various affinity metrics (Ki, IC50, Kd, EC50) to standardized format
    """
    
    def __init__(self, bindingdb_path: Optional[str] = None):
        """
        Initialize data preparator
        
        Args:
            bindingdb_path: Path to BindingDB TSV file
        """
        self.bindingdb_path = bindingdb_path
        self.data = None
        self.processed_data = []
        
        # Thermodynamic constants
        self.RT = 0.593  # kcal/mol at 298K (25°C)
        self.R = 1.987e-3  # kcal/(mol·K)
        
    def load_bindingdb(self, target_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load and filter BindingDB data
        
        Args:
            target_name: Filter by target name (e.g., 'ACVR1', 'ALK2')
        """
        assert self.bindingdb_path is not None, "bindingdb_path must be provided"
        
        print(f"Loading BindingDB from {self.bindingdb_path}...")
        
        # Load with specific columns
        usecols = ['Ligand SMILES', 'Target Name', 'Ki (nM)', 'IC50 (nM)', 
                   'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)',
                   'pH', 'Temp (C)', 'PDB ID(s) for Ligand-Target Complex']
        
        self.data = pd.read_csv(self.bindingdb_path, sep='\t', 
                                usecols=usecols, low_memory=False)
        
        print(f"Loaded {len(self.data)} entries from BindingDB")
        
        # Filter by target if specified
        if target_name:
            self.data = self.data[self.data['Target Name'].str.contains(
                target_name, case=False, na=False)]
            print(f"Filtered to {len(self.data)} entries for target: {target_name}")
        
        # Remove entries without SMILES
        self.data = self.data.dropna(subset=['Ligand SMILES'])
        print(f"Retained {len(self.data)} entries with valid SMILES")
        
        return self.data
    
    def convert_to_free_energy(self, affinity_nm: float, temp_c: float = 25.0) -> float:
        """
        Convert binding affinity (Kd, Ki, IC50 in nM) to free energy (kcal/mol)
        
        ΔG = RT ln(Kd)
        
        Args:
            affinity_nm: Binding affinity in nM
            temp_c: Temperature in Celsius
        
        Returns:
            Free energy in kcal/mol (more negative = stronger binding)
        """
        if pd.isna(affinity_nm) or affinity_nm <= 0:
            return np.nan
        
        # Convert nM to M
        kd_M = affinity_nm * 1e-9
        
        # Calculate RT at given temperature
        temp_K = temp_c + 273.15
        RT = self.R * temp_K
        
        # ΔG = RT ln(Kd)
        delta_g = RT * np.log(kd_M)
        
        return delta_g
    
    def get_best_affinity(self, row: pd.Series) -> Tuple[float, str]:
        """
        Get the best available affinity metric from a row
        Priority: Kd > Ki > IC50 > EC50
        
        Returns:
            Tuple of (affinity_nM, metric_type)
        """
        if not pd.isna(row['Kd (nM)']):
            return row['Kd (nM)'], 'Kd'
        elif not pd.isna(row['Ki (nM)']):
            return row['Ki (nM)'], 'Ki'
        elif not pd.isna(row['IC50 (nM)']):
            return row['IC50 (nM)'], 'IC50'
        elif not pd.isna(row['EC50 (nM)']):
            return row['EC50 (nM)'], 'EC50'
        else:
            return np.nan, 'None'
    
    def compute_molecular_descriptors(self, smiles: str, 
                                       conf_3d: bool = False) -> Optional[Dict]:
        """
        Compute comprehensive molecular descriptors
        
        Args:
            smiles: SMILES string
            conf_3d: Whether to generate 3D conformation and compute 3D descriptors
        
        Returns:
            Dictionary of molecular descriptors
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            
            descriptors = {
                # Basic properties
                'mol_weight': Descriptors.MolWt(mol),  # type: ignore
                'logp': Descriptors.MolLogP(mol),  # type: ignore
                'num_h_donors': Descriptors.NumHDonors(mol),  # type: ignore
                'num_h_acceptors': Descriptors.NumHAcceptors(mol),  # type: ignore
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),  # type: ignore
                'tpsa': Descriptors.TPSA(mol),  # type: ignore
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),  # type: ignore
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                
                # Topological
                'bertz_ct': Descriptors.BertzCT(mol),  # type: ignore
                
                # Electronic
                'num_valence_electrons': Descriptors.NumValenceElectrons(mol),  # type: ignore
                
                # Drug-likeness
                'qed': Descriptors.qed(mol),  # type: ignore
            }
            
            # Generate 3D conformation if requested
            if conf_3d:
                try:
                    rdDistGeom.EmbedMolecule(mol, randomSeed=42)
                    rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
                    
                    # 3D descriptors
                    descriptors['pmi1'] = Descriptors3D.PMI1(mol)  # type: ignore
                    descriptors['pmi2'] = Descriptors3D.PMI2(mol)  # type: ignore
                    descriptors['pmi3'] = Descriptors3D.PMI3(mol)  # type: ignore
                    descriptors['asphericity'] = Descriptors3D.Asphericity(mol)  # type: ignore
                    descriptors['eccentricity'] = Descriptors3D.Eccentricity(mol)  # type: ignore
                    descriptors['inertial_shape_factor'] = Descriptors3D.InertialShapeFactor(mol)  # type: ignore
                    descriptors['radius_of_gyration'] = Descriptors3D.RadiusOfGyration(mol)  # type: ignore
                    
                except Exception as e:
                    print(f"Warning: Could not compute 3D descriptors: {e}")
                    descriptors.update({k: np.nan for k in ['pmi1', 'pmi2', 'pmi3', 
                                                              'asphericity', 'eccentricity',
                                                              'inertial_shape_factor', 
                                                              'radius_of_gyration']})
            
            return descriptors
            
        except Exception as e:
            print(f"Error computing descriptors for {smiles}: {e}")
            return None
    
    def get_morgan_fingerprint(self, smiles: str, radius: int = 2, 
                                nBits: int = 2048) -> Optional[np.ndarray]:
        """
        Generate Morgan (circular) fingerprint
        
        Args:
            smiles: SMILES string
            radius: Fingerprint radius
            nBits: Number of bits
        
        Returns:
            Binary fingerprint array
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Use new MorganGenerator API to avoid deprecation warning
            mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = mgen.GetFingerprint(mol)
            return np.array(fp)
        except:
            return None
    
    def prepare_dataset(self, output_dir: str = 'data/processed_affinity',
                        target_name: Optional[str] = None,
                        min_affinity: Optional[float] = None,
                        max_affinity: Optional[float] = None) -> pd.DataFrame:
        """
        Prepare complete dataset with affinity annotations and molecular features
        
        Args:
            output_dir: Directory to save processed data
            target_name: Filter by specific target
            min_affinity: Minimum affinity threshold in nM (more selective)
            max_affinity: Maximum affinity threshold in nM (less selective)
        
        Returns:
            Processed DataFrame
        """
        if self.data is None:
            self.load_bindingdb(target_name=target_name)
        
        assert self.data is not None, "Data must be loaded before processing"
        
        os.makedirs(output_dir, exist_ok=True)
        
        processed_records = []
        
        print("Processing molecules...")
        for idx, row in self.data.iterrows():
            idx_int = int(idx) if isinstance(idx, (int, np.integer)) else 0  # type: ignore
            if idx_int % 100 == 0:
                print(f"Processed {idx_int}/{len(self.data)} molecules...")
            
            smiles = row['Ligand SMILES']
            
            # Get best affinity metric
            affinity_nm, metric_type = self.get_best_affinity(row)
            
            if pd.isna(affinity_nm):
                continue
            
            # Apply affinity filters
            if min_affinity and affinity_nm < min_affinity:
                continue
            if max_affinity and affinity_nm > max_affinity:
                continue
            
            # Convert to free energy
            temp = row['Temp (C)'] if not pd.isna(row['Temp (C)']) else 25.0
            delta_g = self.convert_to_free_energy(affinity_nm, temp)
            
            if pd.isna(delta_g):
                continue
            
            # Compute molecular descriptors
            descriptors = self.compute_molecular_descriptors(smiles, conf_3d=True)
            if descriptors is None:
                continue
            
            # Get fingerprint
            fingerprint = self.get_morgan_fingerprint(smiles)
            if fingerprint is None:
                continue
            
            # Compile record
            record = {
                'smiles': smiles,
                'target': row['Target Name'],
                'affinity_nM': affinity_nm,
                'affinity_type': metric_type,
                'delta_g_kcal_mol': delta_g,
                'pKd': -np.log10(affinity_nm * 1e-9),  # -log10(Kd in M)
                'temperature_C': temp,
                'pH': row['pH'] if not pd.isna(row['pH']) else np.nan,
                'pdb_id': row['PDB ID(s) for Ligand-Target Complex'],
                **descriptors,
                'fingerprint': fingerprint.tolist(),
            }
            
            # Add kinetics if available
            if not pd.isna(row['kon (M-1-s-1)']):
                record['kon'] = row['kon (M-1-s-1)']
            if not pd.isna(row['koff (s-1)']):
                record['koff'] = row['koff (s-1)']
                record['residence_time_s'] = 1.0 / row['koff (s-1)']
            
            processed_records.append(record)
        
        print(f"\nSuccessfully processed {len(processed_records)} molecules")
        
        # Check if we have any data
        if len(processed_records) == 0:
            print("\n⚠ Warning: No molecules were successfully processed")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(processed_records)
        
        # Save to files
        df_save = df.drop(columns=['fingerprint'])  # Save fingerprints separately
        df_save.to_csv(os.path.join(output_dir, 'affinity_dataset.csv'), index=False)
        
        # Save fingerprints as numpy array
        fingerprints = np.array([r['fingerprint'] for r in processed_records])
        np.save(os.path.join(output_dir, 'morgan_fingerprints.npy'), fingerprints)
        
        # Save metadata
        metadata = {
            'num_samples': len(df),
            'affinity_range_nM': {
                'min': float(df['affinity_nM'].min()),
                'max': float(df['affinity_nM'].max()),
                'mean': float(df['affinity_nM'].mean()),
                'median': float(df['affinity_nM'].median()),
            },
            'delta_g_range_kcal_mol': {
                'min': float(df['delta_g_kcal_mol'].min()),
                'max': float(df['delta_g_kcal_mol'].max()),
                'mean': float(df['delta_g_kcal_mol'].mean()),
            },
            'affinity_types': df['affinity_type'].value_counts().to_dict(),
            'targets': df['target'].value_counts().head(10).to_dict(),
            'fingerprint_dim': fingerprints.shape[1],
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to {output_dir}")
        print(f"Affinity range: {metadata['affinity_range_nM']['min']:.2f} - "
              f"{metadata['affinity_range_nM']['max']:.2f} nM")
        print(f"ΔG range: {metadata['delta_g_range_kcal_mol']['min']:.2f} - "
              f"{metadata['delta_g_range_kcal_mol']['max']:.2f} kcal/mol")
        
        return df
    
    def prepare_from_smiles_list(self, smiles_file: str, 
                                  output_dir: str = 'data/processed_ligands') -> pd.DataFrame:
        """
        Prepare dataset from a list of SMILES (for ligands without known affinity)
        
        Args:
            smiles_file: Path to file with SMILES strings
            output_dir: Directory to save processed data
        
        Returns:
            DataFrame with molecular features
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Read SMILES
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(smiles_list)} SMILES...")
        
        processed_records = []
        for idx, smiles in enumerate(smiles_list):
            print(f"Processing ligand_{idx}...")
            
            # Compute descriptors
            descriptors = self.compute_molecular_descriptors(smiles, conf_3d=True)
            if descriptors is None:
                print(f"  Skipped: Could not process SMILES")
                continue
            
            # Get fingerprint
            fingerprint = self.get_morgan_fingerprint(smiles)
            if fingerprint is None:
                print(f"  Skipped: Could not generate fingerprint")
                continue
            
            record = {
                'ligand_id': f'ligand_{idx}',
                'smiles': smiles,
                **descriptors,
                'fingerprint': fingerprint.tolist(),
            }
            
            processed_records.append(record)
            print(f"  ✓ Processed successfully")
        
        # Create DataFrame
        if len(processed_records) == 0:
            print("\n⚠ Warning: No ligands were successfully processed")
            return pd.DataFrame()
        
        df = pd.DataFrame(processed_records)
        
        # Save
        df_save = df.drop(columns=['fingerprint'])
        df_save.to_csv(os.path.join(output_dir, 'ligand_features.csv'), index=False)
        
        fingerprints = np.array([r['fingerprint'] for r in processed_records])
        np.save(os.path.join(output_dir, 'ligand_fingerprints.npy'), fingerprints)
        
        print(f"\nProcessed {len(df)} ligands, saved to {output_dir}")
        
        return df


if __name__ == "__main__":
    # Example usage
    preparator = AffinityDataPreparator(
        bindingdb_path='data/bindingdb_data/BindingDB_All.tsv'
    )
    
    # Prepare dataset for ACVR1/ALK2 target
    df = preparator.prepare_dataset(
        output_dir='data/processed_affinity',
        target_name='ACVR1',
        min_affinity=0.1,  # >= 0.1 nM
        max_affinity=10000,  # <= 10 µM
    )
    
    # Also prepare our candidate ligands
    df_ligands = preparator.prepare_from_smiles_list(
        smiles_file='data/initial_SMILES/SMILES_strings.txt',
        output_dir='data/processed_ligands'
    )

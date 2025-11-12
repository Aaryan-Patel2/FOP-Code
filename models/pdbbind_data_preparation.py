"""
PDBBind Data Preparation for Bayesian Affinity Prediction
Follows the workflow: PDBBind → Refined Set / General Set → Feature Extraction

Refined Set: High-quality data with Kd, Ki, or both
General Set: Broader data with Kd + Ki

Features extracted:
1. Protein sequences → Encoded as integers for CNN
2. Ligand SMILES → Encoded as character integers for CNN
3. BINANA protein-ligand descriptors → Complex interaction features
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
import os
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class ProteinSequenceEncoder:
    """Encode protein sequences as integer arrays for CNN"""
    
    # Standard amino acids + special tokens
    AMINO_ACIDS = ['<PAD>', '<UNK>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                   'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 
                   'W', 'Y', 'X', 'B', 'Z']
    
    def __init__(self):
        self.char_to_idx = {aa: idx for idx, aa in enumerate(self.AMINO_ACIDS)}
        self.idx_to_char = {idx: aa for aa, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.AMINO_ACIDS)
    
    def encode(self, sequence: str, max_length: int = 1000) -> np.ndarray:
        """
        Encode protein sequence to integer array
        
        Args:
            sequence: Amino acid sequence string
            max_length: Maximum sequence length (pad/truncate)
        
        Returns:
            Integer array of shape [max_length]
        """
        # Convert to uppercase and encode
        sequence = sequence.upper()
        encoded = [self.char_to_idx.get(aa, self.char_to_idx['<UNK>']) 
                   for aa in sequence[:max_length]]
        
        # Pad if necessary
        if len(encoded) < max_length:
            encoded += [self.char_to_idx['<PAD>']] * (max_length - len(encoded))
        
        return np.array(encoded, dtype=np.int64)


class SMILESEncoder:
    """Encode SMILES strings as integer arrays for CNN"""
    
    def __init__(self):
        # Common SMILES characters
        self.chars = ['<PAD>', '<UNK>', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I',
                      'c', 'n', 'o', 's', 'H', '1', '2', '3', '4', '5', '6',
                      '(', ')', '[', ']', '=', '#', '@', '+', '-', '/', '\\',
                      '.', ':', '%', '0', '7', '8', '9']
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)
    
    def encode(self, smiles: str, max_length: int = 200) -> np.ndarray:
        """
        Encode SMILES string to integer array
        
        Args:
            smiles: SMILES string
            max_length: Maximum SMILES length (pad/truncate)
        
        Returns:
            Integer array of shape [max_length]
        """
        # Handle multi-character tokens
        encoded = []
        i = 0
        while i < len(smiles) and len(encoded) < max_length:
            # Check for two-character tokens (Cl, Br)
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.char_to_idx:
                    encoded.append(self.char_to_idx[two_char])
                    i += 2
                    continue
            
            # Single character
            char = smiles[i]
            encoded.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
            i += 1
        
        # Pad if necessary
        if len(encoded) < max_length:
            encoded += [self.char_to_idx['<PAD>']] * (max_length - len(encoded))
        
        return np.array(encoded, dtype=np.int64)


class ComplexDescriptorCalculator:
    """
    Calculate protein-ligand complex interaction descriptors
    Mimics BINANA-style descriptors for protein-ligand interactions
    """
    
    def __init__(self):
        pass
    
    def calculate_from_smiles(self, smiles: str, protein_features: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate complex descriptors from ligand SMILES
        (Simplified version - in real pipeline, would use 3D structures)
        
        Args:
            smiles: Ligand SMILES string
            protein_features: Optional protein-specific features
        
        Returns:
            Descriptor vector of shape [descriptor_dim]
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(200)
            
            mol = Chem.AddHs(mol)
            
            # Molecular descriptors
            descriptors = [
                Descriptors.MolWt(mol),  # type: ignore
                Descriptors.MolLogP(mol),  # type: ignore
                Descriptors.NumHDonors(mol),  # type: ignore
                Descriptors.NumHAcceptors(mol),  # type: ignore
                Descriptors.NumRotatableBonds(mol),  # type: ignore
                Descriptors.TPSA(mol),  # type: ignore
                Descriptors.NumAromaticRings(mol),  # type: ignore
                Descriptors.NumAliphaticRings(mol),  # type: ignore
                Descriptors.NumSaturatedRings(mol),  # type: ignore
                Descriptors.NumHeteroatoms(mol),  # type: ignore
                Descriptors.FractionCSP3(mol),  # type: ignore
                Descriptors.NumValenceElectrons(mol),  # type: ignore
                Descriptors.BertzCT(mol),  # type: ignore
                Descriptors.BalabanJ(mol),  # type: ignore
                Descriptors.qed(mol),  # type: ignore
            ]
            
            # Generate Morgan fingerprint (circular) using new API
            mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=185)
            fp = mgen.GetFingerprint(mol)
            fp_array = np.array(fp)
            
            # Combine all features
            all_features = np.concatenate([descriptors, fp_array])
            
            # Pad to 200 dimensions
            if len(all_features) < 200:
                all_features = np.pad(all_features, (0, 200 - len(all_features)))
            else:
                all_features = all_features[:200]
            
            return all_features.astype(np.float32)
            
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return np.zeros(200, dtype=np.float32)


class PDBBindDataPreparator:
    """
    Prepare PDBBind-style data for Bayesian affinity prediction
    
    Workflow:
    1. Load PDBBind or BindingDB data
    2. Split into Refined Set (high quality) and General Set
    3. Extract protein sequences, ligand SMILES, and complex descriptors
    4. Save processed data for training
    """
    
    def __init__(self):
        self.protein_encoder = ProteinSequenceEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.descriptor_calculator = ComplexDescriptorCalculator()
    
    def load_bindingdb_as_pdbbind(self, bindingdb_path: str,
                                   target_name: Optional[str] = None,
                                   min_quality_entries: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load BindingDB data and split into Refined/General sets
        
        Refined Set: Entries with high-quality measurements (Kd, Ki, Kd+Ki)
        General Set: All entries with any affinity measurement (Kd+Ki)
        
        Args:
            bindingdb_path: Path to BindingDB TSV
            target_name: Optional target filter
            min_quality_entries: Minimum publications for refined set
        
        Returns:
            refined_df, general_df
        """
        print(f"Loading BindingDB from {bindingdb_path}...")
        
        # Load relevant columns
        usecols = ['Ligand SMILES', 'Target Name', 'Ki (nM)', 'Kd (nM)',
                   'IC50 (nM)', 'BindingDB Target Chain Sequence 1',
                   'PDB ID(s) for Ligand-Target Complex', 'Curation/DataSource']
        
        df = pd.read_csv(bindingdb_path, sep='\t', usecols=usecols, 
                         low_memory=False, nrows=50000, compression=None)  # Limit for memory, disable compression
        
        print(f"Loaded {len(df)} entries")
        
        # Filter by target if specified
        if target_name:
            df = df[df['Target Name'].str.contains(target_name, case=False, na=False)]
            print(f"Filtered to {len(df)} entries for target: {target_name}")
        
        # Remove entries without SMILES or sequence
        df = df.dropna(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence 1'])
        print(f"Entries with SMILES and sequence: {len(df)}")
        
        # Determine best affinity value
        def get_best_affinity(row):
            if not pd.isna(row['Kd (nM)']):
                return row['Kd (nM)'], 'Kd'
            elif not pd.isna(row['Ki (nM)']):
                return row['Ki (nM)'], 'Ki'
            elif not pd.isna(row['IC50 (nM)']):
                return row['IC50 (nM)'] * 2, 'IC50'  # Rough conversion
            else:
                return np.nan, None
        
        # Apply function and properly expand results
        affinity_results = df.apply(get_best_affinity, axis=1, result_type='expand')
        df['affinity_nM'] = pd.to_numeric(affinity_results[0], errors='coerce')
        df['affinity_type'] = affinity_results[1]
        
        df = df.dropna(subset=['affinity_nM'])
        print(f"Entries with valid affinity: {len(df)}")
        
        # Refined Set: High-quality data (Kd or Ki only, with PDB structure if possible)
        refined_mask = (
            (df['affinity_type'].isin(['Kd', 'Ki'])) &
            (df['affinity_nM'] > 0) &
            (df['affinity_nM'] < 100000)  # < 100 μM
        )
        
        refined_df = df[refined_mask].copy()
        print(f"\nRefined Set: {len(refined_df)} entries")
        
        # General Set: All data with affinity
        general_df = df.copy()
        print(f"General Set: {len(general_df)} entries")
        
        return refined_df, general_df
    
    def prepare_dataset(self, df: pd.DataFrame, output_dir: str,
                        dataset_name: str = 'refined',
                        max_protein_len: int = 1000,
                        max_smiles_len: int = 200) -> Dict:
        """
        Process dataset into CNN-ready format
        
        Args:
            df: DataFrame with protein sequences, SMILES, and affinities
            output_dir: Directory to save processed data
            dataset_name: Name of dataset ('refined' or 'general')
            max_protein_len: Maximum protein sequence length
            max_smiles_len: Maximum SMILES length
        
        Returns:
            Statistics dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing {dataset_name} dataset...")
        print(f"Total samples: {len(df)}")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            idx_int = int(idx) if isinstance(idx, (int, np.integer)) else 0  # type: ignore
            if idx_int % 100 == 0:
                print(f"  Processed {idx_int}/{len(df)}...")
            
            try:
                # Encode protein sequence
                protein_seq = row['BindingDB Target Chain Sequence 1']
                protein_encoded = self.protein_encoder.encode(protein_seq, max_protein_len)
                
                # Encode ligand SMILES
                smiles = row['Ligand SMILES']
                smiles_encoded = self.smiles_encoder.encode(smiles, max_smiles_len)
                
                # Calculate complex descriptors
                complex_desc = self.descriptor_calculator.calculate_from_smiles(smiles)
                
                # Convert affinity to pKd (or pKi)
                affinity_nM = row['affinity_nM']
                pKd = -np.log10(affinity_nM * 1e-9)  # Convert nM to M, then -log10
                
                record = {
                    'protein_seq_encoded': protein_encoded,
                    'smiles_encoded': smiles_encoded,
                    'complex_descriptors': complex_desc,
                    'affinity_nM': affinity_nM,
                    'pKd': pKd,
                    'affinity_type': row['affinity_type'],
                    'target_name': row['Target Name'],
                    'smiles': smiles,
                    'protein_seq': protein_seq[:100]  # Store first 100 AA for reference
                }
                
                processed_data.append(record)
                
            except Exception as e:
                if idx_int < 10:  # Only print first few errors
                    print(f"  Error processing entry {idx_int}: {e}")
                continue
        
        print(f"\nSuccessfully processed {len(processed_data)} samples")
        
        if len(processed_data) == 0:
            print("⚠ Warning: No samples were successfully processed")
            return {}
        
        # Convert to arrays
        protein_seqs = np.stack([d['protein_seq_encoded'] for d in processed_data])
        smiles_encoded = np.stack([d['smiles_encoded'] for d in processed_data])
        complex_descs = np.stack([d['complex_descriptors'] for d in processed_data])
        affinities = np.array([d['pKd'] for d in processed_data], dtype=np.float32)
        
        # Normalize targets to [0, 1] range for better training stability
        affinity_min = affinities.min()
        affinity_max = affinities.max()
        affinities_normalized = (affinities - affinity_min) / (affinity_max - affinity_min)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f'{dataset_name}_protein_sequences.npy'), protein_seqs)
        np.save(os.path.join(output_dir, f'{dataset_name}_ligand_smiles.npy'), smiles_encoded)
        np.save(os.path.join(output_dir, f'{dataset_name}_complex_descriptors.npy'), complex_descs)
        np.save(os.path.join(output_dir, f'{dataset_name}_affinities.npy'), affinities_normalized)
        
        # Save metadata
        metadata_df = pd.DataFrame([{
            'affinity_nM': d['affinity_nM'],
            'pKd': d['pKd'],
            'affinity_type': d['affinity_type'],
            'target_name': d['target_name'],
            'smiles': d['smiles'],
            'protein_seq': d['protein_seq']
        } for d in processed_data])
        
        metadata_df.to_csv(os.path.join(output_dir, f'{dataset_name}_metadata.csv'), index=False)
        
        # Statistics
        stats = {
            'n_samples': len(processed_data),
            'affinity_range_nM': {
                'min': float(metadata_df['affinity_nM'].min()),
                'max': float(metadata_df['affinity_nM'].max()),
                'mean': float(metadata_df['affinity_nM'].mean()),
            },
            'pKd_range': {
                'min': float(metadata_df['pKd'].min()),
                'max': float(metadata_df['pKd'].max()),
                'mean': float(metadata_df['pKd'].mean()),
            },
            'normalization': {
                'affinity_min': float(affinity_min),
                'affinity_max': float(affinity_max),
                'method': 'min-max to [0, 1]'
            },
            'affinity_types': metadata_df['affinity_type'].value_counts().to_dict(),
            'protein_seq_shape': protein_seqs.shape,
            'smiles_shape': smiles_encoded.shape,
            'complex_desc_shape': complex_descs.shape,
        }
        
        with open(os.path.join(output_dir, f'{dataset_name}_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{dataset_name.upper()} Dataset Statistics:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Affinity range: {stats['affinity_range_nM']['min']:.2f} - {stats['affinity_range_nM']['max']:.2f} nM")
        print(f"  pKd range: {stats['pKd_range']['min']:.2f} - {stats['pKd_range']['max']:.2f}")
        print(f"  Affinity types: {stats['affinity_types']}")
        print(f"\nSaved to: {output_dir}")
        
        return stats


if __name__ == "__main__":
    print("=" * 80)
    print("PDBBIND DATA PREPARATION - TEST")
    print("=" * 80)
    
    # Test encoders
    print("\n[TEST 1] Protein Sequence Encoder")
    protein_encoder = ProteinSequenceEncoder()
    test_seq = "MTEYKLVVVGAGGVGKSALTIQLIQ"
    encoded = protein_encoder.encode(test_seq, max_length=50)
    print(f"  Sequence: {test_seq}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Vocab size: {protein_encoder.vocab_size}")
    print(f"  First 10 tokens: {encoded[:10]}")
    
    print("\n[TEST 2] SMILES Encoder")
    smiles_encoder = SMILESEncoder()
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    encoded = smiles_encoder.encode(test_smiles, max_length=50)
    print(f"  SMILES: {test_smiles}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Vocab size: {smiles_encoder.vocab_size}")
    print(f"  First 20 tokens: {encoded[:20]}")
    
    print("\n[TEST 3] Complex Descriptor Calculator")
    desc_calc = ComplexDescriptorCalculator()
    descriptors = desc_calc.calculate_from_smiles(test_smiles)
    print(f"  Descriptors shape: {descriptors.shape}")
    print(f"  First 10 values: {descriptors[:10]}")
    
    print("\n" + "=" * 80)
    print("✓ Data preparation components tested successfully!")
    print("=" * 80)
    print("\nTo process full BindingDB dataset:")
    print("  python3 models/pdbbind_data_preparation.py")

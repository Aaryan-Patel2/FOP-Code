"""
PyTorch Dataset for Affinity Prediction
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class AffinityDataset(Dataset):
    """
    PyTorch Dataset for protein-ligand affinity data
    
    Handles:
    - Protein sequences (encoded)
    - Ligand SMILES (encoded)
    - Complex descriptors (molecular features)
    - Affinity values (pKd)
    """
    
    def __init__(self, 
                 protein_seqs: np.ndarray, 
                 ligand_smiles: np.ndarray,
                 complex_descriptors: np.ndarray, 
                 affinities: np.ndarray):
        """
        Args:
            protein_seqs: Encoded protein sequences [n_samples, seq_len]
            ligand_smiles: Encoded ligand SMILES [n_samples, smiles_len]
            complex_descriptors: Molecular descriptors [n_samples, n_features]
            affinities: Binding affinities in pKd [n_samples]
        """
        self.protein_seqs = torch.from_numpy(protein_seqs).long()
        self.ligand_smiles = torch.from_numpy(ligand_smiles).long()
        self.complex_descriptors = torch.from_numpy(complex_descriptors).float()
        self.affinities = torch.from_numpy(affinities).float()
    
    def __len__(self) -> int:
        return len(self.affinities)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.protein_seqs[idx], 
            self.ligand_smiles[idx],
            self.complex_descriptors[idx], 
            self.affinities[idx]
        )

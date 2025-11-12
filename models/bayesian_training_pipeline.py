"""
Bayesian Affinity Prediction Training Pipeline - PyTorch Lightning
- ELBO loss computation
- Ensemble ML models (RF, GB, DTBoost)
- Training and evaluation infrastructure
- Model evaluation with PCC, RMSE, MAE
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
import matplotlib.pyplot as plt

from models.bayesian_affinity_predictor import (
    HybridBayesianAffinityNetwork,
    BayesianAffinityLoss,
    create_hnn_affinity_model
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.bayesian_affinity_predictor import (
    HybridBayesianAffinityNetwork,
    BayesianAffinityLoss,
    create_hnn_affinity_model
)


class AffinityDataset(Dataset):
    """PyTorch Dataset for protein-ligand affinity data"""
    
    def __init__(self, protein_seqs: np.ndarray, ligand_smiles: np.ndarray,
                 complex_descriptors: np.ndarray, affinities: np.ndarray):
        self.protein_seqs = torch.from_numpy(protein_seqs).long()
        self.ligand_smiles = torch.from_numpy(ligand_smiles).long()
        self.complex_descriptors = torch.from_numpy(complex_descriptors).float()
        self.affinities = torch.from_numpy(affinities).float()
    
    def __len__(self):
        return len(self.affinities)
    
    def __getitem__(self, idx):
        return (self.protein_seqs[idx], self.ligand_smiles[idx],
                self.complex_descriptors[idx], self.affinities[idx])


class EnsembleMLModels:
    """
    Ensemble of traditional ML models for complex descriptors
    - Random Forest (RF)
    - Gradient Boosting (GB)
    - DTBoost (implemented as another GB variant)
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
        
        self.dtboost = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=7,
            learning_rate=0.05,
            random_state=random_state + 1
        )
        
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all ensemble models"""
        print("Training Random Forest...")
        self.rf.fit(X, y)
        
        print("Training Gradient Boosting...")
        self.gb.fit(X, y)
        
        print("Training DTBoost...")
        self.dtboost.fit(X, y)
        
        self.is_trained = True
        print("✓ Ensemble models trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get consensus prediction from ensemble"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        pred_rf = self.rf.predict(X)
        pred_gb = self.gb.predict(X)
        pred_dtboost = self.dtboost.predict(X)
        
        # Average predictions (consensus)
        consensus = (pred_rf + pred_gb + pred_dtboost) / 3.0
        return consensus
    
    def predict_all(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual and consensus predictions"""
        return {
            'rf': self.rf.predict(X),
            'gb': self.gb.predict(X),
            'dtboost': self.dtboost.predict(X),
            'consensus': self.predict(X)
        }




class BayesianAffinityTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for Bayesian Hybrid Affinity Network
    Refactored from manual training loop for cleaner, more maintainable code
    """
    
    def __init__(self, model: HybridBayesianAffinityNetwork,
                 learning_rate: float = 1e-3,
                 kl_weight: float = 1.0,
                 dataset_size: int = 1000,
                 uncertainty_samples: int = 50):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.uncertainty_samples = uncertainty_samples
        
        # Loss function
        self.loss_fn = BayesianAffinityLoss(
            dataset_size=dataset_size,
            kl_weight=kl_weight
        )
        
        # Ensemble ML models (trained separately)
        self.ensemble_ml = EnsembleMLModels()
        
        # For collecting outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, protein_seq, ligand_smiles, complex_desc):
        """Forward pass"""
        return self.model(protein_seq, ligand_smiles, complex_desc)
    
    def training_step(self, batch, batch_idx):
        """Training step - called automatically by Lightning"""
        protein_seq, ligand_smiles, complex_desc, target = batch
        
        # Forward pass
        predictions = self(protein_seq, ligand_smiles, complex_desc)
        kl_divergence = self.model.kl_divergence()
        
        # Compute loss
        loss, metrics = self.loss_fn(predictions, target, kl_divergence)
        
        # Log metrics
        self.log('train_loss', metrics['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_nll', metrics['nll'], on_step=False, on_epoch=True)
        self.log('train_kl', metrics['kl_divergence'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - called automatically by Lightning"""
        protein_seq, ligand_smiles, complex_desc, target = batch
        
        # Forward pass
        predictions = self(protein_seq, ligand_smiles, complex_desc)
        kl_divergence = self.model.kl_divergence()
        
        # Compute loss
        loss, metrics = self.loss_fn(predictions, target, kl_divergence)
        
        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': metrics['total_loss'],
            'preds': predictions.detach().cpu(),
            'targets': target.detach().cpu()
        })
        
        self.log('val_loss', metrics['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch"""
        if not self.validation_step_outputs:
            return
        
        # Gather predictions
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy()
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).numpy()
        
        # Diagnostic: Check if model is predicting constant values
        pred_std = np.std(all_preds)
        pred_range = all_preds.max() - all_preds.min()
        target_std = np.std(all_targets)
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        pcc_tuple = pearsonr(all_targets, all_preds)
        pcc = float(pcc_tuple[0])  # type: ignore
        
        # Log metrics
        self.log('val_rmse', float(rmse), prog_bar=True)
        self.log('val_mae', float(mae), prog_bar=True)
        self.log('val_pcc', pcc, prog_bar=True)
        self.log('val_r2', float(pcc ** 2))
        
        # Log diagnostics (not in progress bar)
        self.log('val_pred_std', float(pred_std))
        self.log('val_pred_range', float(pred_range))
        self.log('val_target_std', float(target_std))
        
        # Warning if model is collapsing to constant predictions
        if pred_std < 0.01 * target_std:
            print(f"\n⚠ Warning: Model predictions have very low variance (std={pred_std:.6f})")
            print(f"   Target std: {target_std:.6f}, Pred range: {pred_range:.6f}")
            print(f"   This may indicate over-regularization or vanishing gradients")
        
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step - called automatically by Lightning"""
        protein_seq, ligand_smiles, complex_desc, target = batch
        
        # Forward pass
        predictions = self(protein_seq, ligand_smiles, complex_desc)
        kl_divergence = self.model.kl_divergence()
        
        # Compute loss
        loss, metrics = self.loss_fn(predictions, target, kl_divergence)
        
        # Store for epoch-end metrics
        self.test_step_outputs.append({
            'loss': metrics['total_loss'],
            'preds': predictions.detach().cpu(),
            'targets': target.detach().cpu()
        })
        
        self.log('test_loss', metrics['total_loss'])
        
        return loss
    
    def on_test_epoch_end(self):
        """Compute metrics at end of test epoch"""
        if not self.test_step_outputs:
            return
        
        # Gather predictions
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).numpy()
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        pcc_tuple = pearsonr(all_targets, all_preds)
        pcc = float(pcc_tuple[0])  # type: ignore
        
        # Log metrics
        self.log('test_rmse', float(rmse))
        self.log('test_mae', float(mae))
        self.log('test_pcc', pcc)
        self.log('test_r2', float(pcc ** 2))
        
        print(f"\n{'='*60}")
        print(f"Test Set Performance (HNN-Affinity):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  PCC:  {pcc:.4f}")
        print(f"  R²:   {pcc**2:.4f}")
        print(f"{'='*60}\n")
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):  # type: ignore
        """Configure optimizer and scheduler - balanced regularization"""
        # Reduced weight decay to prevent over-regularization
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # Reduced from 1e-4 to allow model to learn patterns
        )
        
        # Cosine annealing for smooth learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,  # Period of cosine cycle
            eta_min=1e-5  # Minimum learning rate (not too low)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def get_consensus_prediction(self, protein_seq: torch.Tensor,
                                  ligand_smiles: torch.Tensor,
                                  complex_desc: torch.Tensor,
                                  n_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Get consensus predictions from HNN and ensemble ML models
        
        Returns:
            Dictionary with predictions from all models and consensus
        """
        # HNN predictions with uncertainty
        mean_hnn, std_hnn = self.model.predict_with_uncertainty(
            protein_seq, ligand_smiles, complex_desc, n_samples=n_samples
        )
        
        # Ensemble ML predictions
        complex_desc_np = complex_desc.cpu().numpy()
        ml_preds = self.ensemble_ml.predict_all(complex_desc_np)
        
        # Final consensus (weighted average)
        final_consensus = (
            0.6 * mean_hnn.cpu().numpy() +
            0.4 * ml_preds['consensus']
        )
        
        return {
            'hnn_mean': mean_hnn.cpu().numpy(),
            'hnn_std': std_hnn.cpu().numpy(),
            'ml_rf': ml_preds['rf'],
            'ml_gb': ml_preds['gb'],
            'ml_dtboost': ml_preds['dtboost'],
            'ml_consensus': ml_preds['consensus'],
            'final_consensus': final_consensus
        }



    
    
    def train_ensemble_ml(self, complex_descriptors: np.ndarray, affinities: np.ndarray):
        """Train ensemble ML models on complex descriptors"""
        print("\nTraining Ensemble ML Models (RF, GB, DTBoost)...")
        print("=" * 80)
        self.ensemble_ml.fit(complex_descriptors, affinities)


def plot_results(trainer: BayesianAffinityTrainer, val_preds: np.ndarray,
                 val_targets: np.ndarray, save_dir: str):
    """
    Plot training curves and predictions
    NOTE: This function is deprecated for Lightning version.
    Use TensorBoard for visualizations instead.
    """
    
    # Training curves - disabled for Lightning version
    # Lightning uses built-in loggers (TensorBoard, CSV, etc.)
    print("⚠ plot_results is deprecated for Lightning version.")
    print("  Use TensorBoard to visualize training: tensorboard --logdir lightning_logs")
    return


if __name__ == "__main__":
    print("=" * 80)
    print("BAYESIAN AFFINITY PREDICTOR - TRAINING PIPELINE TEST")
    print("=" * 80)
    
    # Test with dummy data
    print("\nGenerating dummy data...")
    n_samples = 1000
    protein_seqs = np.random.randint(0, 25, (n_samples, 500))
    ligand_smiles = np.random.randint(0, 70, (n_samples, 100))
    complex_desc = np.random.randn(n_samples, 200).astype(np.float32)
    affinities = np.random.randn(n_samples).astype(np.float32) * 2 + 7  # pKd ~5-9
    
    # Create dataset
    dataset = AffinityDataset(protein_seqs, ligand_smiles, complex_desc, affinities)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    config = {
        'protein_vocab_size': 25,
        'ligand_vocab_size': 70,
        'complex_descriptor_dim': 200,
        'fusion_hidden_dims': [512, 256, 128],
    }
    
    model = create_hnn_affinity_model(config)
    lit_model = BayesianAffinityTrainer(model, learning_rate=1e-3, kl_weight=0.01, dataset_size=train_size)
    
    # Quick training test with Lightning
    print("\nRunning quick training test (5 epochs) with PyTorch Lightning...")
    import pytorch_lightning as pl
    trainer = pl.Trainer(max_epochs=5, accelerator='auto', devices=1)
    trainer.fit(lit_model, train_loader, val_loader)
    
    print("\n✓ Training pipeline test completed successfully!")

"""
Complete Ensemble Affinity Predictor

Combines all models:
- Bayesian Neural Network (uncertainty quantification)
- Random Forest (molecular descriptors)
- Gradient Boosting (molecular descriptors)
- DTBoost (molecular descriptors)

Provides unified interface for predictions with ensemble averaging.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork, create_hnn_affinity_model
from models.random_forest_model import RandomForestAffinityModel
from models.gradient_boosting_models import GradientBoostingAffinityModel, DTBoostAffinityModel


class EnsembleAffinityPredictor:
    """
    Complete ensemble combining Bayesian NN and traditional ML models
    
    Ensemble composition:
    - Bayesian Neural Network (BNN): 60% weight
    - Random Forest (RF): 15% weight
    - Gradient Boosting (GB): 15% weight
    - DTBoost: 10% weight
    """
    
    def __init__(self,
                 protein_vocab_size: int = 25,
                 ligand_vocab_size: int = 70,
                 complex_descriptor_dim: int = 256,
                 ensemble_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            protein_vocab_size: Size of protein amino acid vocabulary
            ligand_vocab_size: Size of ligand SMILES vocabulary
            complex_descriptor_dim: Dimension of molecular descriptors
            ensemble_weights: Custom weights for ensemble (default: BNN=0.6, RF=0.15, GB=0.15, DTBoost=0.1)
        """
        # Bayesian Neural Network
        bnn_config = {
            'protein_vocab_size': protein_vocab_size,
            'ligand_vocab_size': ligand_vocab_size,
            'complex_descriptor_dim': complex_descriptor_dim
        }
        self.bnn = create_hnn_affinity_model(bnn_config)
        
        # Traditional ML models
        self.rf_model = RandomForestAffinityModel(n_estimators=100, max_depth=15)
        self.gb_model = GradientBoostingAffinityModel(n_estimators=100, max_depth=5)
        self.dtboost_model = DTBoostAffinityModel(n_estimators=100, max_depth=7)
        
        # Ensemble weights
        if ensemble_weights is None:
            self.ensemble_weights = {
                'bnn': 0.6,
                'rf': 0.15,
                'gb': 0.15,
                'dtboost': 0.1
            }
        else:
            self.ensemble_weights = ensemble_weights
        
        # Training status
        self.bnn_trained = False
        self.ml_models_trained = False
    
    def train_bnn(self, 
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  num_epochs: int = 20,
                  learning_rate: float = 1e-3,
                  device: str = 'cpu') -> Dict[str, List[float]]:
        """
        Train the Bayesian Neural Network component
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on ('cpu' or 'cuda')
        
        Returns:
            Training history with losses and metrics
        """
        from models.utils.losses import BayesianAffinityLoss
        from models.utils.metrics import calculate_metrics
        
        self.bnn.to(device)
        self.bnn.train()
        
        optimizer = torch.optim.Adam(self.bnn.parameters(), lr=learning_rate)
        loss_fn = BayesianAffinityLoss(kl_weight=0.01)
        
        history = {'train_loss': [], 'val_loss': [], 'val_pcc': []}
        
        for epoch in range(num_epochs):
            # Training
            train_losses = []
            for batch in train_loader:
                protein_seqs, ligand_smiles, complex_desc, affinities = batch
                protein_seqs = protein_seqs.to(device)
                ligand_smiles = ligand_smiles.to(device)
                complex_desc = complex_desc.to(device)
                affinities = affinities.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = self.bnn(protein_seqs, ligand_smiles, complex_desc)
                kl_div = self.bnn.kl_divergence if hasattr(self.bnn, 'kl_divergence') else None
                
                loss, metrics = loss_fn(predictions, affinities, kl_div)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.bnn.eval()
            val_preds = []
            val_targets = []
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    protein_seqs, ligand_smiles, complex_desc, affinities = batch
                    protein_seqs = protein_seqs.to(device)
                    ligand_smiles = ligand_smiles.to(device)
                    complex_desc = complex_desc.to(device)
                    affinities = affinities.to(device).unsqueeze(1)
                    
                    predictions = self.bnn(protein_seqs, ligand_smiles, complex_desc)
                    loss, _ = loss_fn(predictions, affinities)
                    
                    val_preds.append(predictions.cpu().numpy())
                    val_targets.append(affinities.cpu().numpy())
                    val_losses.append(loss.item())
            
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_metrics = calculate_metrics(val_preds, val_targets)
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            history['val_pcc'].append(val_metrics['pcc'])
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, "
                  f"Val PCC: {history['val_pcc'][-1]:.4f}")
            
            self.bnn.train()
        
        self.bnn_trained = True
        return history
    
    def train_ml_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train traditional ML models on molecular descriptors
        
        Args:
            X_train: Training features (complex descriptors) [n_samples, n_features]
            y_train: Training targets (affinities) [n_samples]
        """
        print("Training Random Forest...")
        self.rf_model.train(X_train, y_train)
        
        print("Training Gradient Boosting...")
        self.gb_model.train(X_train, y_train)
        
        print("Training DTBoost...")
        self.dtboost_model.train(X_train, y_train)
        
        self.ml_models_trained = True
        print("✓ All ML models trained")
    
    def predict_ensemble(self,
                        protein_seqs: torch.Tensor,
                        ligand_smiles: torch.Tensor,
                        complex_descriptors: torch.Tensor,
                        n_samples: int = 100,
                        device: str = 'cpu') -> Dict[str, float]:
        """
        Make ensemble prediction with all models
        
        Args:
            protein_seqs: Protein sequences [batch_size, seq_len]
            ligand_smiles: Ligand SMILES [batch_size, smiles_len]
            complex_descriptors: Molecular descriptors [batch_size, n_features]
            n_samples: Number of Monte Carlo samples for BNN uncertainty
            device: Device for computation
        
        Returns:
            Dict with 'affinity', 'uncertainty', and individual model predictions
        """
        # BNN predictions with uncertainty
        if self.bnn_trained:
            self.bnn.to(device)
            self.bnn.eval()
            
            protein_seqs = protein_seqs.to(device)
            ligand_smiles = ligand_smiles.to(device)
            complex_descriptors_tensor = complex_descriptors.to(device)
            
            bnn_predictions = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.bnn(protein_seqs, ligand_smiles, complex_descriptors_tensor)
                    bnn_predictions.append(pred.cpu().numpy())
            
            bnn_pred_mean = np.mean(bnn_predictions, axis=0).flatten()
            bnn_pred_std = np.std(bnn_predictions, axis=0).flatten()
        else:
            bnn_pred_mean = np.zeros(len(protein_seqs))
            bnn_pred_std = np.zeros(len(protein_seqs))
        
        # ML model predictions
        complex_descriptors_np = complex_descriptors.cpu().numpy()
        
        if self.ml_models_trained:
            rf_pred = self.rf_model.predict(complex_descriptors_np)
            gb_pred = self.gb_model.predict(complex_descriptors_np)
            dtboost_pred = self.dtboost_model.predict(complex_descriptors_np)
        else:
            rf_pred = np.zeros(len(complex_descriptors_np))
            gb_pred = np.zeros(len(complex_descriptors_np))
            dtboost_pred = np.zeros(len(complex_descriptors_np))
        
        # Ensemble averaging
        ensemble_pred = (
            self.ensemble_weights['bnn'] * bnn_pred_mean +
            self.ensemble_weights['rf'] * rf_pred +
            self.ensemble_weights['gb'] * gb_pred +
            self.ensemble_weights['dtboost'] * dtboost_pred
        )
        
        # Ensemble uncertainty (from BNN + variance across models)
        model_variance = np.var([bnn_pred_mean, rf_pred, gb_pred, dtboost_pred], axis=0)
        total_uncertainty = np.sqrt(bnn_pred_std**2 + model_variance)
        
        return {
            'affinity': float(ensemble_pred[0]),
            'uncertainty': float(total_uncertainty[0]),
            'bnn_pred': float(bnn_pred_mean[0]),
            'bnn_std': float(bnn_pred_std[0]),
            'rf_pred': float(rf_pred[0]),
            'gb_pred': float(gb_pred[0]),
            'dtboost_pred': float(dtboost_pred[0])
        }
    
    def save_models(self, save_dir: str) -> None:
        """Save all models to directory"""
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save BNN
        if self.bnn_trained:
            torch.save(self.bnn.state_dict(), f"{save_dir}/bnn_model.pt")
        
        # Save ML models
        if self.ml_models_trained:
            with open(f"{save_dir}/rf_model.pkl", 'wb') as f:
                pickle.dump(self.rf_model, f)
            with open(f"{save_dir}/gb_model.pkl", 'wb') as f:
                pickle.dump(self.gb_model, f)
            with open(f"{save_dir}/dtboost_model.pkl", 'wb') as f:
                pickle.dump(self.dtboost_model, f)
        
        print(f"✓ Models saved to {save_dir}")
    
    def load_models(self, save_dir: str) -> None:
        """Load all models from directory"""
        import os
        import pickle
        
        # Load BNN
        bnn_path = f"{save_dir}/bnn_model.pt"
        if os.path.exists(bnn_path):
            self.bnn.load_state_dict(torch.load(bnn_path, map_location='cpu'))
            self.bnn_trained = True
        
        # Load ML models
        if os.path.exists(f"{save_dir}/rf_model.pkl"):
            with open(f"{save_dir}/rf_model.pkl", 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(f"{save_dir}/gb_model.pkl", 'rb') as f:
                self.gb_model = pickle.load(f)
            with open(f"{save_dir}/dtboost_model.pkl", 'rb') as f:
                self.dtboost_model = pickle.load(f)
            self.ml_models_trained = True
        
        print(f"✓ Models loaded from {save_dir}")

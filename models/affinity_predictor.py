"""
Binding Affinity Prediction Module
Deep learning model for predicting K_d / binding affinity from molecular features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os


class MolecularAffinityDataset(Dataset):
    """PyTorch Dataset for molecular affinity prediction"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 fingerprints: Optional[np.ndarray] = None):
        """
        Args:
            features: Molecular descriptors (N x D)
            targets: Binding affinity values (N,)
            fingerprints: Morgan fingerprints (N x FP_dim)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.fingerprints = torch.FloatTensor(fingerprints) if fingerprints is not None else None
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        if self.fingerprints is not None:
            return (self.features[idx], self.fingerprints[idx]), self.targets[idx]
        else:
            return self.features[idx], self.targets[idx]


class AffinityPredictor(nn.Module):
    """
    Neural network for binding affinity prediction
    
    Architecture:
    - Separate encoders for molecular descriptors and fingerprints
    - Fusion layer combining both representations
    - MLP regressor head
    """
    
    def __init__(self, 
                 descriptor_dim: int,
                 fingerprint_dim: int = 2048,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Args:
            descriptor_dim: Dimension of molecular descriptors
            fingerprint_dim: Dimension of fingerprints
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(AffinityPredictor, self).__init__()
        
        self.descriptor_dim = descriptor_dim
        self.fingerprint_dim = fingerprint_dim
        self.use_batch_norm = use_batch_norm
        
        # Descriptor encoder
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Fingerprint encoder
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Fusion and prediction layers
        layers = []
        input_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tuple of (descriptors, fingerprints) or just descriptors
        
        Returns:
            Predicted affinity
        """
        if isinstance(x, tuple):
            descriptors, fingerprints = x
            desc_encoded = self.descriptor_encoder(descriptors)
            fp_encoded = self.fingerprint_encoder(fingerprints)
            combined = torch.cat([desc_encoded, fp_encoded], dim=1)
        else:
            # Only descriptors provided
            combined = self.descriptor_encoder(x)
        
        output = self.predictor(combined)
        return output.squeeze()


class AffinityPredictionTrainer:
    """Training and evaluation for affinity predictor"""
    
    def __init__(self, 
                 model: AffinityPredictor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        """
        Args:
            model: AffinityPredictor model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def prepare_data(self, data_dir: str, 
                     target_col: str = 'delta_g_kcal_mol',
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data for training
        
        Args:
            data_dir: Directory containing processed data
            target_col: Column name for target variable
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Load data
        df = pd.read_csv(os.path.join(data_dir, 'affinity_dataset.csv'))
        fingerprints = np.load(os.path.join(data_dir, 'morgan_fingerprints.npy'))
        
        print(f"Loaded {len(df)} samples")
        
        # Select descriptor columns (exclude metadata)
        descriptor_cols = [col for col in df.columns if col not in [
            'smiles', 'target', 'affinity_nM', 'affinity_type', target_col,
            'pKd', 'temperature_C', 'pH', 'pdb_id', 'kon', 'koff', 'residence_time_s'
        ]]
        
        # Extract features and targets
        X_descriptors = df[descriptor_cols].fillna(0).values
        y = df[target_col].values
        
        print(f"Feature dimensions: descriptors={X_descriptors.shape[1]}, "
              f"fingerprints={fingerprints.shape[1]}")
        
        # Split data
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=42)
        
        # Normalize features
        X_descriptors = self.scaler_features.fit_transform(X_descriptors)
        y = np.asarray(y)  # Ensure y is a numpy array
        y = self.scaler_targets.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create datasets
        train_dataset = MolecularAffinityDataset(
            X_descriptors[train_idx], y[train_idx], fingerprints[train_idx]
        )
        val_dataset = MolecularAffinityDataset(
            X_descriptors[val_idx], y[val_idx], fingerprints[val_idx]
        )
        test_dataset = MolecularAffinityDataset(
            X_descriptors[test_idx], y[test_idx], fingerprints[test_idx]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            if isinstance(batch_x, tuple):
                batch_x = tuple(x.to(self.device) for x in batch_x)
            else:
                batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if isinstance(batch_x, tuple):
                    batch_x = tuple(x.to(self.device) for x in batch_x)
                else:
                    batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = F.mse_loss(predictions, batch_y)
                
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Inverse transform to original scale
        all_preds_orig = self.scaler_targets.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets_orig = self.scaler_targets.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        metrics = {
            'mse': mean_squared_error(all_targets_orig, all_preds_orig),
            'rmse': np.sqrt(mean_squared_error(all_targets_orig, all_preds_orig)),
            'mae': mean_absolute_error(all_targets_orig, all_preds_orig),
            'r2': r2_score(all_targets_orig, all_preds_orig),
        }
        
        return total_loss / len(val_loader), metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, save_dir: str = 'models/checkpoints'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nTraining on {self.device}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pt'))
                print(f"✓ Saved best model (epoch {epoch+1})")
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val RMSE: {val_metrics['rmse']:.4f} kcal/mol")
                print(f"  Val MAE: {val_metrics['mae']:.4f} kcal/mol")
                print(f"  Val R²: {val_metrics['r2']:.4f}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_features': self.scaler_features,
            'scaler_targets': self.scaler_targets,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler_features = checkpoint['scaler_features']
        self.scaler_targets = checkpoint['scaler_targets']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Model loaded from {path}")
    
    def predict(self, features: np.ndarray, fingerprints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features: Molecular descriptors
            fingerprints: Morgan fingerprints (optional)
        
        Returns:
            Predicted affinities
        """
        self.model.eval()
        
        # Normalize features
        features = self.scaler_features.transform(features)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        if fingerprints is not None:
            fingerprints_tensor = torch.FloatTensor(fingerprints).to(self.device)
            x = (features_tensor, fingerprints_tensor)
        else:
            x = features_tensor
        
        with torch.no_grad():
            predictions = self.model(x).cpu().numpy()
        
        # Inverse transform to original scale
        predictions = self.scaler_targets.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions


if __name__ == "__main__":
    # Example training pipeline
    print("Initializing Affinity Predictor...")
    
    # Create model
    model = AffinityPredictor(
        descriptor_dim=20,  # Will be set based on data
        fingerprint_dim=2048,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3
    )
    
    # Create trainer
    trainer = AffinityPredictionTrainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        data_dir='data/processed_affinity',
        target_col='delta_g_kcal_mol'
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        save_dir='models/checkpoints'
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"Test RMSE: {test_metrics['rmse']:.4f} kcal/mol")
    print(f"Test MAE: {test_metrics['mae']:.4f} kcal/mol")
    print(f"Test R²: {test_metrics['r2']:.4f}")

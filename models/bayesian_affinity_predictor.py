"""
Bayesian Neural Network for Binding Affinity Prediction
Following the HNN-Affinity architecture with Bayesian inference

Architecture:
- Protein sequence encoder (CNN)
- Ligand SMILES encoder (CNN)
- Protein-ligand descriptor encoders (ensemble: RF, GB, DTBoost)
- Hybrid Bayesian Neural Network (BNN) that merges all features
- Uncertainty quantification via Bayesian posterior

Reference: Feed-forward Bayesian Neural Networks with moderate depth (3 conv layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional, List, Dict
import math


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty
    Implements variational inference with mean-field Gaussian approximation
    """
    
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))
        
        # Prior distribution (fixed)
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize means with Xavier
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.bias_mu, 0.)
        
        # Initialize log-sigma to small values (tight initial uncertainty)
        nn.init.constant_(self.weight_logsigma, -5.)
        nn.init.constant_(self.bias_logsigma, -5.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling (reparameterization trick)
        """
        if self.training:
            # Sample weights: w = μ + σ * ε, where ε ~ N(0,1)
            weight_sigma = torch.exp(self.weight_logsigma)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            bias_sigma = torch.exp(self.bias_logsigma)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean weights for inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior
        KL(q(w)||p(w)) for Gaussian distributions
        """
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        
        # KL for weights
        kl_weight = (
            self.prior_log_sigma - self.weight_logsigma +
            (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma**2) - 0.5
        ).sum()
        
        # KL for bias
        kl_bias = (
            self.prior_log_sigma - self.bias_logsigma +
            (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma**2) - 0.5
        ).sum()
        
        return kl_weight + kl_bias


class ProteinCNN(nn.Module):
    """
    Convolutional Neural Network for protein sequence encoding
    3-layer CNN as recommended for Bayesian networks
    """
    
    def __init__(self, vocab_size: int = 25, embedding_dim: int = 128, 
                 hidden_dims: List[int] = [256, 256, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 256):
        super(ProteinCNN, self).__init__()
        
        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Three convolutional layers (as per Bayesian BNN recommendations)
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim
        
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # Reduced from 0.3 - less aggressive regularization
                nn.MaxPool1d(2)
            ))
            in_channels = hidden_dim
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Protein sequence indices [batch_size, seq_len]
        Returns:
            Encoded features [batch_size, output_dim]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        x = self.embedding(x)
        
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global pooling: [batch, hidden_dim, seq_len] -> [batch, hidden_dim, 1]
        x = self.global_pool(x).squeeze(-1)
        
        # Final projection
        x = self.fc(x)
        return x


class LigandCNN(nn.Module):
    """
    Convolutional Neural Network for ligand SMILES encoding
    3-layer CNN for character-level SMILES processing
    """
    
    def __init__(self, vocab_size: int = 70, embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 256, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 output_dim: int = 256):
        super(LigandCNN, self).__init__()
        
        # SMILES character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Three convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim
        
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # Reduced from 0.3 - less aggressive regularization
                nn.MaxPool1d(2)
            ))
            in_channels = hidden_dim
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: SMILES character indices [batch_size, smiles_len]
        Returns:
            Encoded features [batch_size, output_dim]
        """
        x = self.embedding(x)
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class ComplexDescriptorEncoder(nn.Module):
    """
    Encoder for BINANA protein-ligand complex descriptors
    Uses standard layers (will be ensembled with RF, GB, DTBoost externally)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128],
                 output_dim: int = 128, dropout: float = 0.3):
        super(ComplexDescriptorEncoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HybridBayesianAffinityNetwork(nn.Module):
    """
    Hybrid Bayesian Neural Network for Affinity Prediction (HNN-Affinity)
    
    Merges:
    - Protein sequence features (CNN)
    - Ligand SMILES features (CNN)
    - Protein-ligand complex descriptors
    
    Uses Bayesian fully-connected layers for uncertainty quantification
    """
    
    def __init__(self,
                 protein_vocab_size: int = 25,
                 ligand_vocab_size: int = 70,
                 complex_descriptor_dim: int = 200,
                 protein_output_dim: int = 256,
                 ligand_output_dim: int = 256,
                 complex_output_dim: int = 128,
                 fusion_hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3,
                 prior_sigma: float = 1.0):
        super(HybridBayesianAffinityNetwork, self).__init__()
        
        # Feature encoders
        self.protein_encoder = ProteinCNN(
            vocab_size=protein_vocab_size,
            output_dim=protein_output_dim
        )
        
        self.ligand_encoder = LigandCNN(
            vocab_size=ligand_vocab_size,
            output_dim=ligand_output_dim
        )
        
        self.complex_encoder = ComplexDescriptorEncoder(
            input_dim=complex_descriptor_dim,
            output_dim=complex_output_dim
        )
        
        # Fusion layers (Bayesian)
        fusion_input_dim = protein_output_dim + ligand_output_dim + complex_output_dim
        
        self.bayesian_layers = nn.ModuleList()
        in_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            self.bayesian_layers.append(BayesianLinear(in_dim, hidden_dim, prior_sigma))
            in_dim = hidden_dim
        
        # Output layer (also Bayesian)
        self.output_layer = BayesianLinear(in_dim, 1, prior_sigma)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, protein_seq: torch.Tensor, ligand_smiles: torch.Tensor,
                complex_descriptors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network
        
        Args:
            protein_seq: [batch_size, seq_len]
            ligand_smiles: [batch_size, smiles_len]
            complex_descriptors: [batch_size, descriptor_dim]
        
        Returns:
            Predicted affinity: [batch_size]
        """
        # Encode features
        protein_features = self.protein_encoder(protein_seq)
        ligand_features = self.ligand_encoder(ligand_smiles)
        complex_features = self.complex_encoder(complex_descriptors)
        
        # Merge features
        merged = torch.cat([protein_features, ligand_features, complex_features], dim=1)
        
        # Bayesian fully-connected layers
        x = merged
        for bayesian_layer in self.bayesian_layers:
            x = bayesian_layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output
        output = self.output_layer(x)
        return output.squeeze(-1)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers
        """
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.bayesian_layers:
            kl = kl + layer.kl_divergence()  # type: ignore
        kl = kl + self.output_layer.kl_divergence()  # type: ignore
        return kl
    
    def predict_with_uncertainty(self, protein_seq: torch.Tensor, 
                                  ligand_smiles: torch.Tensor,
                                  complex_descriptors: torch.Tensor,
                                  n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification
        
        Args:
            protein_seq, ligand_smiles, complex_descriptors: Input features
            n_samples: Number of Monte Carlo samples for uncertainty estimation
        
        Returns:
            mean_prediction: Mean predicted affinity
            uncertainty: Standard deviation (epistemic uncertainty)
        """
        # Keep model in eval but enable dropout for MC sampling
        was_training = self.training
        self.eval()
        
        # Enable dropout layers only
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(protein_seq, ligand_smiles, complex_descriptors)
                predictions.append(pred)
        
        # Restore original mode
        if was_training:
            self.train()
        else:
            self.eval()
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class BayesianAffinityLoss(nn.Module):
    """
    ELBO Loss for Bayesian Neural Network
    Loss = -log p(y|x,w) + β * KL(q(w)||p(w))
    
    The KL term regularizes the posterior to stay close to the prior
    """
    
    def __init__(self, dataset_size: int, kl_weight: float = 1.0):
        super(BayesianAffinityLoss, self).__init__()
        self.dataset_size = dataset_size
        self.kl_weight = kl_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                kl_divergence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ELBO loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth affinities
            kl_divergence: Sum of KL divergences from all Bayesian layers
        
        Returns:
            total_loss: ELBO loss
            metrics: Dictionary with loss components
        """
        # Negative log-likelihood (reconstruction loss)
        # Use 'mean' reduction to avoid scaling with batch size
        nll = F.mse_loss(predictions, targets, reduction='mean')
        
        # KL divergence averaged by batch size for stability
        # This makes the KL term comparable in scale to the NLL term
        batch_size = predictions.size(0)
        kl_scaled = kl_divergence / batch_size
        
        # Total ELBO loss
        total_loss = nll + self.kl_weight * kl_scaled
        
        metrics = {
            'total_loss': total_loss.item(),
            'nll': nll.item(),
            'kl_divergence': kl_divergence.item(),
            'kl_scaled': kl_scaled.item()
        }
        
        return total_loss, metrics


def create_hnn_affinity_model(config: Dict) -> HybridBayesianAffinityNetwork:
    """
    Factory function to create HNN-Affinity model
    
    Args:
        config: Configuration dictionary with model hyperparameters
    
    Returns:
        Instantiated HNN-Affinity model
    """
    return HybridBayesianAffinityNetwork(
        protein_vocab_size=config.get('protein_vocab_size', 25),
        ligand_vocab_size=config.get('ligand_vocab_size', 70),
        complex_descriptor_dim=config.get('complex_descriptor_dim', 200),
        protein_output_dim=config.get('protein_output_dim', 256),
        ligand_output_dim=config.get('ligand_output_dim', 256),
        complex_output_dim=config.get('complex_output_dim', 128),
        fusion_hidden_dims=config.get('fusion_hidden_dims', [512, 256, 128]),
        dropout=config.get('dropout', 0.3),
        prior_sigma=config.get('prior_sigma', 1.0)
    )


if __name__ == "__main__":
    # Test the model architecture
    print("=" * 80)
    print("BAYESIAN HYBRID NEURAL NETWORK - ARCHITECTURE TEST")
    print("=" * 80)
    
    # Create model
    config = {
        'protein_vocab_size': 25,
        'ligand_vocab_size': 70,
        'complex_descriptor_dim': 200,
        'protein_output_dim': 256,
        'ligand_output_dim': 256,
        'complex_output_dim': 128,
        'fusion_hidden_dims': [512, 256, 128],
        'dropout': 0.3,
        'prior_sigma': 1.0
    }
    
    model = create_hnn_affinity_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 8
    protein_seq = torch.randint(0, 25, (batch_size, 500))
    ligand_smiles = torch.randint(0, 70, (batch_size, 100))
    complex_desc = torch.randn(batch_size, 200)
    
    print(f"\nInput shapes:")
    print(f"  Protein sequence: {protein_seq.shape}")
    print(f"  Ligand SMILES: {ligand_smiles.shape}")
    print(f"  Complex descriptors: {complex_desc.shape}")
    
    # Forward pass
    model.train()
    output = model(protein_seq, ligand_smiles, complex_desc)
    kl = model.kl_divergence()
    
    print(f"\nOutput:")
    print(f"  Predictions shape: {output.shape}")
    print(f"  KL divergence: {kl.item():.4f}")
    
    # Test uncertainty quantification
    print(f"\nTesting uncertainty quantification...")
    mean_pred, std_pred = model.predict_with_uncertainty(
        protein_seq, ligand_smiles, complex_desc, n_samples=50
    )
    
    print(f"  Mean predictions shape: {mean_pred.shape}")
    print(f"  Std predictions shape: {std_pred.shape}")
    print(f"  Mean uncertainty: {std_pred.mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Architecture test completed successfully!")
    print("=" * 80)

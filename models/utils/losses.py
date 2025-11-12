"""
Loss Functions for Affinity Prediction Models

Contains custom loss functions for training:
- BayesianAffinityLoss: ELBO loss for Bayesian models
- Additional loss utilities
"""

import torch
import torch.nn as nn
from typing import Tuple


class BayesianAffinityLoss(nn.Module):
    """
    Bayesian loss with ELBO (Evidence Lower Bound)
    Combines reconstruction loss and KL divergence
    """
    
    def __init__(self, kl_weight: float = 0.01):
        super().__init__()
        self.kl_weight = kl_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                kl_div: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Ground truth affinities [batch_size, 1]
            kl_div: KL divergence term (optional)
        
        Returns:
            loss: Total loss value
            metrics: Dict with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = self.mse(predictions, targets)
        
        # KL divergence (regularization)
        if kl_div is not None:
            kl_loss = kl_div
            total_loss = recon_loss + self.kl_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=predictions.device)
            total_loss = recon_loss
        
        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, metrics


def create_loss_function(loss_type: str = 'bayesian', **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss ('bayesian', 'mse', etc.)
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'bayesian':
        return BayesianAffinityLoss(**kwargs)
    elif loss_type == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

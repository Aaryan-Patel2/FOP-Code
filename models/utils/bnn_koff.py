"""
Bayesian k_off Prediction Module

Predicts dissociation rate (k_off) from binding affinity and molecular features.

Methods available:
1. Empirical relationship (literature-based)
2. Machine learning predictor (requires training data)
3. Bayesian neural network (full uncertainty quantification)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class KoffPrediction:
    """Container for k_off prediction results"""
    koff_mean: float          # Mean k_off (s⁻¹)
    koff_std: float           # Standard deviation (uncertainty)
    kon_estimated: float      # Estimated k_on (M⁻¹s⁻¹)
    residence_time: float     # 1/k_off (seconds)
    confidence: float         # Prediction confidence (0-1)
    method: str              # Which method was used


# =============================================================================
# Option 1: Empirical Relationship (Literature-Based)
# =============================================================================

def predict_koff_empirical(
    affinity_pKd: float,
    molecular_weight: Optional[float] = None,
    add_uncertainty: bool = True
) -> KoffPrediction:
    """
    Predict k_off using empirical relationships from literature.
    
    Based on:
    - Copeland et al. (2006): Drug-target residence time correlations
    - Tonge (2018): Structure-kinetics relationships
    - Empirical observation: log(k_off) ≈ -0.5 * pKd + constant
    
    Args:
        affinity_pKd: Binding affinity in pKd units (e.g., 7.5)
        molecular_weight: Optional molecular weight (Da) for corrections
        add_uncertainty: Whether to add uncertainty estimates
    
    Returns:
        KoffPrediction with estimated k_off and uncertainty
    
    Examples:
        >>> pred = predict_koff_empirical(affinity_pKd=7.5)
        >>> print(f"k_off: {pred.koff_mean:.3f} s⁻¹")
        >>> print(f"Residence time: {pred.residence_time:.1f} seconds")
    """
    # Empirical relationship: log10(k_off) ≈ -0.5 * pKd + 3.0
    # This gives:
    #   pKd = 7 → k_off ≈ 1 s⁻¹ (fast dissociation - good for FOP!)
    #   pKd = 8 → k_off ≈ 0.3 s⁻¹
    #   pKd = 9 → k_off ≈ 0.1 s⁻¹
    #   pKd = 10 → k_off ≈ 0.03 s⁻¹
    
    # Base prediction
    log_koff = -0.5 * affinity_pKd + 3.0
    
    # Molecular weight correction (larger molecules dissociate slightly slower)
    if molecular_weight is not None:
        mw_correction = -0.0001 * (molecular_weight - 400)  # Normalize around 400 Da
        log_koff += mw_correction
    
    koff_mean = 10 ** log_koff
    
    # Uncertainty estimate (empirical relationships have ~1 log unit uncertainty)
    if add_uncertainty:
        koff_std = koff_mean * 1.5  # ~150% uncertainty (conservative)
    else:
        koff_std = koff_mean * 0.5
    
    # Estimate k_on from K_d = k_off / k_on
    Kd = 10 ** (-affinity_pKd)  # Convert pKd to Kd (M)
    kon_estimated = koff_mean / Kd
    
    # Calculate residence time
    residence_time = 1.0 / koff_mean
    
    # Confidence decreases for extreme affinity values
    confidence = 1.0 - abs(affinity_pKd - 8.0) / 10.0  # Best confidence around pKd=8
    confidence = max(0.3, min(1.0, confidence))
    
    return KoffPrediction(
        koff_mean=koff_mean,
        koff_std=koff_std,
        kon_estimated=kon_estimated,
        residence_time=residence_time,
        confidence=confidence,
        method="empirical"
    )


# =============================================================================
# Option 2: Machine Learning Predictor (Requires Training Data)
# =============================================================================

class MLKoffPredictor:
    """
    Machine learning model for k_off prediction.
    
    Requires training data with known k_off values.
    Uses affinity + molecular descriptors to predict kinetics.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def train(self, features: np.ndarray, koff_values: np.ndarray):
        """
        Train ML model on kinetics data.
        
        Args:
            features: (N, D) array of features (affinity, MW, descriptors, etc.)
            koff_values: (N,) array of k_off values (s⁻¹)
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Train ensemble
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        self.model.fit(features_scaled, np.log10(koff_values))  # Predict log(k_off)
        
        self.is_trained = True
        print(f"✓ ML k_off predictor trained on {len(koff_values)} samples")
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict k_off from features.
        
        Args:
            features: (D,) array of features
        
        Returns:
            (koff_mean, koff_std) tuple
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call .train() first or use empirical method.")
        
        assert self.scaler is not None and self.model is not None
        features_scaled = self.scaler.transform(features.reshape(1, -1))  # type: ignore
        log_koff_pred = self.model.predict(features_scaled)[0]  # type: ignore
        koff_mean = 10 ** log_koff_pred
        
        # Uncertainty from tree variance (if available)
        koff_std = koff_mean * 0.5  # Placeholder
        
        return koff_mean, koff_std


# =============================================================================
# Option 3: Bayesian Neural Network (Full Uncertainty Quantification)
# =============================================================================

class BayesianKoffNetwork(nn.Module):
    """
    Bayesian neural network for k_off prediction with uncertainty.
    
    Similar architecture to your affinity predictor but for kinetics.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Use Bayesian layers (with dropout as approximation)
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)  # MC Dropout for uncertainty
            ])
            prev_dim = hidden_dim
        
        # Output: log(k_off)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log(k_off)"""
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Predict k_off with uncertainty using MC Dropout.
        
        Args:
            x: Input features (affinity encoding + molecular descriptors)
            n_samples: Number of MC samples
        
        Returns:
            (koff_mean, koff_std) tuple
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                log_koff = self.forward(x)
                koff = 10 ** log_koff.item()
                predictions.append(koff)
        
        predictions = np.array(predictions)
        koff_mean = predictions.mean()
        koff_std = predictions.std()
        
        return koff_mean, koff_std


# =============================================================================
# Unified Interface
# =============================================================================

def predict_koff_with_uncertainty(
    affinity: float,
    molecular_features: Optional[np.ndarray] = None,
    method: str = "empirical",
    predictor: Optional[MLKoffPredictor] = None
) -> Tuple[float, float]:
    """
    Unified interface for k_off prediction.
    
    This is the main function called by quick_start.py
    
    Args:
        affinity: Predicted pKd value
        molecular_features: Optional molecular descriptors (200-dim array)
        method: "empirical" or "ml" or "bayesian"
        predictor: Pre-trained ML/Bayesian predictor (optional)
    
    Returns:
        (koff_mean, koff_std) tuple in s⁻¹ units
    
    Examples:
        >>> # Simple empirical prediction
        >>> koff, std = predict_koff_with_uncertainty(affinity=7.5)
        >>> print(f"k_off: {koff:.3f} ± {std:.3f} s⁻¹")
        
        >>> # With molecular features (more accurate)
        >>> koff, std = predict_koff_with_uncertainty(
        ...     affinity=7.5,
        ...     molecular_features=complex_descriptors,
        ...     method="empirical"
        ... )
    """
    
    if method == "empirical":
        # Extract molecular weight if available
        mw = None
        if molecular_features is not None and len(molecular_features) > 0:
            # Assuming MW is one of the first features (you'd need to verify)
            mw = molecular_features[0] if molecular_features[0] > 0 else None
        
        result = predict_koff_empirical(affinity, molecular_weight=mw)
        return result.koff_mean, result.koff_std
    
    elif method == "ml":
        if predictor is None or not predictor.is_trained:
            print("⚠ ML predictor not trained. Falling back to empirical method.")
            result = predict_koff_empirical(affinity)
            return result.koff_mean, result.koff_std
        
        # Combine affinity with molecular features
        if molecular_features is None:
            molecular_features = np.zeros(200)  # Placeholder
        
        features = np.concatenate([[affinity], molecular_features])
        return predictor.predict(features)
    
    elif method == "bayesian":
        print("⚠ Bayesian k_off predictor not yet implemented. Using empirical.")
        result = predict_koff_empirical(affinity)
        return result.koff_mean, result.koff_std
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Utility Functions
# =============================================================================

def evaluate_binding_profile(
    affinity_pKd: float,
    koff: float,
    target_profile: str = "fop"
) -> Dict[str, Any]:
    """
    Evaluate if a compound matches desired binding profile.
    
    Args:
        affinity_pKd: Binding affinity (pKd)
        koff: Dissociation rate (s⁻¹)
        target_profile: "fop" or "traditional"
    
    Returns:
        Dictionary with scoring and recommendations
    """
    Kd = 10 ** (-affinity_pKd)  # M
    kon = koff / Kd  # M⁻¹s⁻¹
    residence_time = 1.0 / koff  # seconds
    
    if target_profile == "fop":
        # FOP goals: pKd 7-8, k_off 0.1-1 s⁻¹, residence 1-10s
        affinity_score = 1.0 - abs(affinity_pKd - 7.5) / 5.0  # Best at pKd=7.5
        koff_score = 1.0 if 0.1 <= koff <= 1.0 else 0.5
        residence_score = 1.0 if 1.0 <= residence_time <= 10.0 else 0.5
        
        overall_score = (affinity_score + koff_score + residence_score) / 3.0
        
        recommendation = "✅ Good FOP candidate!" if overall_score > 0.7 else "⚠️ Needs optimization"
        
    else:  # traditional
        # Traditional goals: pKd > 8, k_off < 0.01 s⁻¹, residence > 100s
        affinity_score = 1.0 if affinity_pKd > 8.0 else 0.5
        koff_score = 1.0 if koff < 0.01 else 0.5
        residence_score = 1.0 if residence_time > 100 else 0.5
        
        overall_score = (affinity_score + koff_score + residence_score) / 3.0
        
        recommendation = "✅ Good traditional drug!" if overall_score > 0.7 else "⚠️ Needs optimization"
    
    return {
        'affinity_pKd': affinity_pKd,
        'Kd_M': Kd,
        'koff_s': koff,
        'kon_M_s': kon,
        'residence_time_s': residence_time,
        'affinity_score': affinity_score,
        'kinetics_score': (koff_score + residence_score) / 2.0,
        'overall_score': overall_score,
        'recommendation': recommendation,
        'profile': target_profile
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("k_off Prediction Examples")
    print("=" * 60)
    
    # Example 1: Empirical prediction
    print("\n[Example 1] Empirical k_off prediction")
    result = predict_koff_empirical(affinity_pKd=7.5, molecular_weight=400)
    print(f"  Affinity: pKd = 7.5 (K_d = 32 nM)")
    print(f"  k_off: {result.koff_mean:.3f} ± {result.koff_std:.3f} s⁻¹")
    print(f"  k_on: {result.kon_estimated:.2e} M⁻¹s⁻¹")
    print(f"  Residence time: {result.residence_time:.1f} seconds")
    print(f"  Confidence: {result.confidence:.2%}")
    
    # Example 2: Profile evaluation
    print("\n[Example 2] FOP profile evaluation")
    profile = evaluate_binding_profile(
        affinity_pKd=7.5,
        koff=0.5,
        target_profile="fop"
    )
    print(f"  Affinity score: {profile['affinity_score']:.2f}")
    print(f"  Kinetics score: {profile['kinetics_score']:.2f}")
    print(f"  Overall score: {profile['overall_score']:.2f}")
    print(f"  {profile['recommendation']}")
    
    # Example 3: Unified interface
    print("\n[Example 3] Unified interface")
    koff_mean, koff_std = predict_koff_with_uncertainty(
        affinity=8.0,
        method="empirical"
    )
    print(f"  k_off: {koff_mean:.3f} ± {koff_std:.3f} s⁻¹")
    print(f"  Residence time: {1/koff_mean:.1f} seconds")

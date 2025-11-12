"""
Random Forest Model for Affinity Prediction

Traditional machine learning model using molecular descriptors.
Part of the ensemble predictor.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional


class RandomForestAffinityModel:
    """
    Random Forest Regressor for binding affinity prediction
    
    Uses complex molecular descriptors as features.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 15,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target affinities [n_samples]
        """
        self.model.fit(X, y.ravel())
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix [n_samples, n_features]
        
        Returns:
            predictions: Predicted affinities [n_samples]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores"""
        if self.is_trained:
            return self.model.feature_importances_
        return None

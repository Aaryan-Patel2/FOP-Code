"""
Gradient Boosting Models for Affinity Prediction

Two variants of Gradient Boosting for ensemble diversity.
Part of the ensemble predictor.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingAffinityModel:
    """
    Standard Gradient Boosting Regressor for binding affinity
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 random_state: int = 42):
        """
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of individual trees
            learning_rate: Learning rate shrinks contribution of each tree
            random_state: Random seed
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X, y.ravel())
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)


class DTBoostAffinityModel:
    """
    Decision Tree Boosting variant (deeper trees, slower learning)
    Provides diversity in the ensemble
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 7,
                 learning_rate: float = 0.05,
                 random_state: int = 43):
        """
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth (deeper than standard GB)
            learning_rate: Learning rate (slower than standard GB)
            random_state: Random seed (different from standard GB)
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X, y.ravel())
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

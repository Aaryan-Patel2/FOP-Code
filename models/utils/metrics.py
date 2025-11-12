"""
Metrics for Model Evaluation

Evaluation metrics for affinity prediction:
- Pearson correlation coefficient (PCC)
- Root mean squared error (RMSE)
- Mean absolute error (MAE)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from typing import Dict, Tuple, Any


def _extract_pcc_value(result: Any) -> float:
    """Extract correlation value from pearsonr result (handles different scipy versions)"""
    if hasattr(result, 'statistic'):
        return float(result.statistic)  # type: ignore
    elif isinstance(result, tuple):
        return float(result[0])
    else:
        return float(result)


def _extract_pcc_pvalue(result: Any) -> float:
    """Extract p-value from pearsonr result (handles different scipy versions)"""
    if hasattr(result, 'pvalue'):
        return float(result.pvalue)  # type: ignore
    elif isinstance(result, tuple):
        return float(result[1])
    else:
        return 0.0


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Model predictions [n_samples]
        targets: Ground truth values [n_samples]
    
    Returns:
        Dict with 'pcc', 'rmse', 'mae'
    """
    # Pearson correlation coefficient
    pcc_result = pearsonr(predictions.flatten(), targets.flatten())
    pcc = _extract_pcc_value(pcc_result)
    
    # RMSE
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    
    # MAE
    mae = float(mean_absolute_error(targets, predictions))
    
    return {
        'pcc': pcc,
        'rmse': rmse,
        'mae': mae
    }


def calculate_pcc(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient
    
    Returns:
        (correlation, p_value)
    """
    result = pearsonr(predictions.flatten(), targets.flatten())
    corr = _extract_pcc_value(result)
    pval = _extract_pcc_pvalue(result)
    return (corr, pval)


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate root mean squared error"""
    return float(np.sqrt(mean_squared_error(targets, predictions)))


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate mean absolute error"""
    return float(mean_absolute_error(targets, predictions))

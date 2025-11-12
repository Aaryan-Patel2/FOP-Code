"""
FOP Affinity Predictor - Bayesian Neural Network for Binding Affinity Prediction

A reusable module for predicting protein-ligand binding affinity and dissociation kinetics.
Designed for integration with molecule generation pipelines (e.g., GCDM).

Main Classes:
    AffinityPredictor - Primary interface for training and prediction

Usage:
    from fop_affinity import AffinityPredictor
    
    # For predictions
    predictor = AffinityPredictor(checkpoint_path='models/pretrained/affinity_predictor.ckpt')
    result = predictor.predict(protein_seq, ligand_smiles)
    
    # For training
    predictor = AffinityPredictor()
    predictor.train(bindingdb_path='data/bindingdb_data/BindingDB_All.tsv', target_name='ACVR1')
"""

__version__ = '1.0.0'
__author__ = 'FOP Research Team'

from quick_start import AffinityPredictor

__all__ = ['AffinityPredictor']

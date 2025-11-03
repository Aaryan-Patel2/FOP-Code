"""
Quick Start Interface for Bayesian Affinity Prediction
Easy-to-use API for making predictions with minimal setup
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AffinityPredictor:
    """
    Simple interface for binding affinity prediction
    
    Usage:
        # Initialize predictor
        predictor = AffinityPredictor()
        
        # Make predictions
        results = predictor.predict(
            protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQ...",
            ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"
        )
        
        print(f"Predicted pKd: {results['affinity']:.2f}")
        print(f"Uncertainty: {results['uncertainty']:.2f}")
        print(f"Predicted k_off: {results['koff']:.2e} s^-1")
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to trained model checkpoint (optional)
                           If None, will use a pre-trained model or train a new one
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.encoders = None
        
        print("=" * 60)
        print("BAYESIAN AFFINITY PREDICTOR")
        print("=" * 60)
        print(f"Device: {self.device}")
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        else:
            print("⚠ No checkpoint provided. Use .train() to train a model first.")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model from checkpoint"""
        from models.bayesian_training_pipeline import BayesianAffinityTrainer
        from models.bayesian_affinity_predictor import create_hnn_affinity_model
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with saved config
        config = checkpoint.get('hyper_parameters', {}).get('model_config', {})
        model = create_hnn_affinity_model(config)
        
        # Load weights
        lit_model = BayesianAffinityTrainer.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self.device
        )
        
        self.model = lit_model.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def predict(self, 
                protein_sequence: str,
                ligand_smiles: str,
                n_samples: int = 100) -> Dict[str, Optional[float]]:
        """
        Predict binding affinity and dissociation rate
        
        Args:
            protein_sequence: Amino acid sequence (e.g., "MTEYKLVVVG...")
            ligand_smiles: SMILES string (e.g., "CC(C)Cc1ccc(cc1)C(C)C(O)=O")
            n_samples: Number of samples for uncertainty estimation
        
        Returns:
            Dictionary with:
                - affinity: Predicted pKd
                - uncertainty: Prediction uncertainty (standard deviation)
                - koff: Predicted dissociation rate (s^-1)
                - residence_time: Predicted residence time (seconds)
                - confidence: Confidence score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide checkpoint_path or train first.")
        
        # Encode inputs
        from models.pdbbind_data_preparation import (
            ProteinSequenceEncoder,
            SMILESEncoder,
            ComplexDescriptorCalculator
        )
        
        if self.encoders is None:
            self.encoders = {
                'protein': ProteinSequenceEncoder(),
                'smiles': SMILESEncoder(),
                'complex': ComplexDescriptorCalculator()
            }
        
        # Encode protein
        protein_encoded = self.encoders['protein'].encode(protein_sequence)
        protein_tensor = torch.from_numpy(protein_encoded).long().unsqueeze(0).to(self.device)
        
        # Encode SMILES
        smiles_encoded = self.encoders['smiles'].encode(ligand_smiles)
        smiles_tensor = torch.from_numpy(smiles_encoded).long().unsqueeze(0).to(self.device)
        
        # Calculate complex descriptors
        complex_desc = self.encoders['complex'].calculate_from_smiles(ligand_smiles)
        complex_tensor = torch.from_numpy(complex_desc).float().unsqueeze(0).to(self.device)
        
        # Make predictions with uncertainty
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(protein_tensor, smiles_tensor, complex_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions).squeeze()
        
        # Calculate statistics
        mean_affinity = predictions.mean()
        uncertainty = predictions.std()
        confidence = 1.0 / (1.0 + uncertainty)  # Higher confidence = lower uncertainty
        
        # Predict k_off (if BNN k_off model is available)
        # Note: k_off prediction module not yet implemented
        koff_mean = None
        residence_time = None
        
        # Uncomment when bnn_koff module is implemented:
        # try:
        #     from models.utils.bnn_koff import predict_koff_with_uncertainty
        #     koff_mean, koff_std = predict_koff_with_uncertainty(
        #         affinity=mean_affinity,
        #         molecular_features=complex_desc
        #     )
        #     residence_time = 1.0 / koff_mean if koff_mean > 0 else np.inf
        # except Exception as e:
        #     koff_mean = None
        #     residence_time = None
        #     print(f"⚠ k_off prediction unavailable: {e}")
        
        return {
            'affinity': float(mean_affinity),
            'uncertainty': float(uncertainty),
            'koff': koff_mean,
            'residence_time': residence_time,
            'confidence': float(confidence)
        }
    
    def predict_batch(self,
                     protein_sequences: List[str],
                     ligand_smiles: List[str],
                     n_samples: int = 100) -> List[Dict[str, Optional[float]]]:
        """
        Predict binding affinity for multiple protein-ligand pairs
        
        Args:
            protein_sequences: List of amino acid sequences
            ligand_smiles: List of SMILES strings
            n_samples: Number of samples for uncertainty estimation
        
        Returns:
            List of prediction dictionaries
        """
        if len(protein_sequences) != len(ligand_smiles):
            raise ValueError("Number of proteins and ligands must match")
        
        results = []
        print(f"Predicting {len(protein_sequences)} protein-ligand pairs...")
        
        for i, (protein, smiles) in enumerate(zip(protein_sequences, ligand_smiles)):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(protein_sequences)}")
            
            try:
                result = self.predict(protein, smiles, n_samples)
                results.append(result)
            except Exception as e:
                print(f"  Error on pair {i}: {e}")
                results.append({
                    'affinity': np.nan,
                    'uncertainty': np.nan,
                    'koff': None,
                    'residence_time': None,
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        print(f"✓ Completed {len(results)} predictions")
        return results
    
    def train(self,
             bindingdb_path: str,
             target_name: Optional[str] = None,
             num_epochs: int = 50,
             batch_size: int = 32,
             learning_rate: float = 1e-3,
             output_dir: str = 'trained_models'):
        """
        Train the model on BindingDB data
        
        Args:
            bindingdb_path: Path to BindingDB TSV file
            target_name: Optional target name filter (e.g., 'ACVR1')
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save trained model
        """
        print("\n" + "=" * 60)
        print("TRAINING BAYESIAN AFFINITY MODEL")
        print("=" * 60)
        
        # Import training components
        from models.pdbbind_data_preparation import PDBBindDataPreparator
        from models.bayesian_affinity_predictor import create_hnn_affinity_model
        from models.bayesian_training_pipeline import AffinityDataset, BayesianAffinityTrainer
        from torch.utils.data import DataLoader, random_split
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        # Prepare data
        preparator = PDBBindDataPreparator()
        refined_df, _ = preparator.load_bindingdb_as_pdbbind(
            bindingdb_path=bindingdb_path,
            target_name=target_name
        )
        
        output_data_dir = 'data/quick_start_processed'
        stats = preparator.prepare_dataset(
            refined_df,
            output_dir=output_data_dir,
            dataset_name='training'
        )
        
        # Load processed data
        import os
        protein_seqs = np.load(os.path.join(output_data_dir, 'training_protein_sequences.npy'))
        ligand_smiles = np.load(os.path.join(output_data_dir, 'training_ligand_smiles.npy'))
        complex_desc = np.load(os.path.join(output_data_dir, 'training_complex_descriptors.npy'))
        affinities = np.load(os.path.join(output_data_dir, 'training_affinities.npy'))
        
        # Create dataset
        dataset = AffinityDataset(protein_seqs, ligand_smiles, complex_desc, affinities)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
        
        # Create model
        config = {
            'protein_vocab_size': preparator.protein_encoder.vocab_size,
            'ligand_vocab_size': preparator.smiles_encoder.vocab_size,
            'complex_descriptor_dim': complex_desc.shape[1],
            'protein_output_dim': 256,
            'ligand_output_dim': 256,
            'complex_output_dim': 128,
            'fusion_hidden_dims': [512, 256, 128],
            'dropout': 0.3,
            'prior_sigma': 1.0
        }
        
        model = create_hnn_affinity_model(config)
        lit_model = BayesianAffinityTrainer(
            model=model,
            learning_rate=learning_rate,
            kl_weight=0.01,
            dataset_size=train_size
        )
        
        # Setup trainer
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss', patience=10)],
            accelerator='auto',
            devices=1
        )
        
        # Train
        trainer.fit(lit_model, train_loader, val_loader)
        
        # Save checkpoint path
        self.checkpoint_path = checkpoint_callback.best_model_path
        self.model = lit_model.model
        
        print(f"\n✓ Training completed!")
        print(f"  Best model saved to: {self.checkpoint_path}")
        
        return self.checkpoint_path


def quick_example():
    """Quick example of how to use the predictor"""
    print("=" * 60)
    print("QUICK START EXAMPLE")
    print("=" * 60)
    print("\nExample 1: Making a single prediction")
    print("-" * 60)
    print("```python")
    print("from quick_start import AffinityPredictor")
    print("")
    print("# Initialize predictor with trained model")
    print("predictor = AffinityPredictor(checkpoint_path='models/best_model.ckpt')")
    print("")
    print("# Make prediction")
    print("result = predictor.predict(")
    print("    protein_sequence='MTEYKLVVVGAGGVGKSALTIQLIQ...',")
    print("    ligand_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O'")
    print(")")
    print("")
    print("print(f'Predicted pKd: {result[\"affinity\"]:.2f}')")
    print("print(f'Uncertainty: {result[\"uncertainty\"]:.2f}')")
    print("print(f'k_off: {result[\"koff\"]:.2e} s^-1')")
    print("```")
    print("")
    print("\nExample 2: Training a new model")
    print("-" * 60)
    print("```python")
    print("predictor = AffinityPredictor()")
    print("predictor.train(")
    print("    bindingdb_path='data/bindingdb_data/BindingDB_All.tsv',")
    print("    target_name='ACVR1',")
    print("    num_epochs=50")
    print(")")
    print("```")
    print("")
    print("\nExample 3: Batch predictions")
    print("-" * 60)
    print("```python")
    print("proteins = ['MTEYK...', 'MAPKK...', 'GPCR...']")
    print("ligands = ['CC(C)Cc1...', 'CN1C=NC2...', 'c1ccc...']")
    print("")
    print("results = predictor.predict_batch(proteins, ligands)")
    print("for i, res in enumerate(results):")
    print("    print(f'Pair {i}: pKd = {res[\"affinity\"]:.2f}')")
    print("```")


if __name__ == "__main__":
    quick_example()

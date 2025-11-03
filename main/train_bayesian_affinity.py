"""
Main Training Script for Bayesian Hybrid Affinity Network - PyTorch Lightning
Complete workflow following the diagram architecture

Workflow:
1. Load PDBBind/BindingDB data → Split into Refined/General sets
2. Extract features: Protein sequences (CNN), Ligand SMILES (CNN), Complex descriptors
3. Train HNN-Affinity model (Bayesian Neural Network) with Lightning
4. Train Ensemble ML models (RF, GB, DTBoost)
5. Generate consensus predictions
6. Evaluate with PCC, RMSE, MAE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from pathlib import Path

from models.pdbbind_data_preparation import PDBBindDataPreparator
from models.bayesian_affinity_predictor import create_hnn_affinity_model
from models.bayesian_training_pipeline import (
    AffinityDataset,
    BayesianAffinityTrainer,
    plot_results
)


def main(args):
    print("=" * 80)
    print("BAYESIAN HYBRID AFFINITY NETWORK - FULL TRAINING PIPELINE")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n[STEP 1] Data Preparation")
    print("-" * 80)
    
    preparator = PDBBindDataPreparator()
    
    # Check if data already processed
    refined_data_dir = args.refined_output_dir
    general_data_dir = args.general_output_dir
    
    refined_protein_path = os.path.join(refined_data_dir, 'refined_protein_sequences.npy')
    
    if not os.path.exists(refined_protein_path) or args.force_reprocess:
        print("Processing BindingDB data...")
        
        # Load and split data
        refined_df, general_df = preparator.load_bindingdb_as_pdbbind(
            bindingdb_path=args.bindingdb_path,
            target_name=args.target_name
        )
        
        # Process Refined Set
        if len(refined_df) > 0:
            print(f"\nProcessing Refined Set ({len(refined_df)} samples)...")
            refined_stats = preparator.prepare_dataset(
                refined_df, 
                refined_data_dir,
                dataset_name='refined',
                max_protein_len=args.max_protein_len,
                max_smiles_len=args.max_smiles_len
            )
        else:
            print("⚠ No samples in Refined Set")
        
        # Process General Set (optional, usually larger)
        if args.use_general_set and len(general_df) > 0:
            print(f"\nProcessing General Set ({len(general_df)} samples)...")
            general_stats = preparator.prepare_dataset(
                general_df,
                general_data_dir,
                dataset_name='general',
                max_protein_len=args.max_protein_len,
                max_smiles_len=args.max_smiles_len
            )
    else:
        print(f"✓ Using preprocessed data from {refined_data_dir}")
    
    # Load processed data
    print(f"\nLoading processed data...")
    protein_seqs = np.load(os.path.join(refined_data_dir, 'refined_protein_sequences.npy'))
    ligand_smiles = np.load(os.path.join(refined_data_dir, 'refined_ligand_smiles.npy'))
    complex_desc = np.load(os.path.join(refined_data_dir, 'refined_complex_descriptors.npy'))
    affinities = np.load(os.path.join(refined_data_dir, 'refined_affinities.npy'))
    
    print(f"✓ Loaded {len(affinities)} samples")
    print(f"  Protein sequences shape: {protein_seqs.shape}")
    print(f"  Ligand SMILES shape: {ligand_smiles.shape}")
    print(f"  Complex descriptors shape: {complex_desc.shape}")
    print(f"  Affinities shape: {affinities.shape}")
    
    # ========================================================================
    # STEP 2: CREATE DATASETS
    # ========================================================================
    print("\n[STEP 2] Creating Train/Val/Test Splits")
    print("-" * 80)
    
    dataset = AffinityDataset(protein_seqs, ligand_smiles, complex_desc, affinities)
    
    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(args.train_split * total_size)
    val_size = int(args.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, 
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, 
                             shuffle=False, num_workers=0)
    
    print(f"✓ Dataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")
    
    # ========================================================================
    # STEP 3: CREATE AND TRAIN HNN-AFFINITY MODEL
    # ========================================================================
    print("\n[STEP 3] Training HNN-Affinity (Bayesian Neural Network)")
    print("-" * 80)
    
    config = {
        'protein_vocab_size': preparator.protein_encoder.vocab_size,
        'ligand_vocab_size': preparator.smiles_encoder.vocab_size,
        'complex_descriptor_dim': complex_desc.shape[1],
        'protein_output_dim': args.protein_dim,
        'ligand_output_dim': args.ligand_dim,
        'complex_output_dim': args.complex_dim,
        'fusion_hidden_dims': args.fusion_dims,
        'dropout': args.dropout,
        'prior_sigma': args.prior_sigma
    }
    
    print(f"Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    model = create_hnn_affinity_model(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create Lightning module
    lit_model = BayesianAffinityTrainer(
        model=model,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        dataset_size=len(train_dataset),
        uncertainty_samples=args.uncertainty_samples
    )
    
    # Setup Lightning Trainer with callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min'
        ) if args.early_stopping else None
    ]
    callbacks = [cb for cb in callbacks if cb is not None]
    
    logger = CSVLogger(save_dir=args.log_dir, name='bayesian_affinity')
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train model with Lightning
    print("\n" + "=" * 80)
    print("TRAINING WITH PYTORCH LIGHTNING")
    print("=" * 80)
    trainer.fit(lit_model, train_loader, val_loader)
    
    # Test model
    print("\n" + "=" * 80)
    print("TESTING MODEL")
    print("=" * 80)
    test_results = trainer.test(lit_model, test_loader)
    
    # ========================================================================
    # STEP 4: TRAIN ENSEMBLE ML MODELS
    # ========================================================================
    print("\n[STEP 4] Training Ensemble ML Models (RF, GB, DTBoost)")
    print("-" * 80)
    
    # Get training data for ensemble
    train_indices = train_dataset.indices
    train_complex_desc = complex_desc[train_indices]
    train_affinities = affinities[train_indices]
    
    lit_model.train_ensemble_ml(train_complex_desc, train_affinities)
    
    # ========================================================================
    # STEP 5: EVALUATE ON TEST SET WITH CONSENSUS
    # ========================================================================
    print("\n[STEP 5] Generating Consensus Predictions on Test Set")
    print("-" * 80)
    
    # Get test set data
    test_indices = test_dataset.indices
    test_protein = torch.from_numpy(protein_seqs[test_indices]).long()
    test_smiles = torch.from_numpy(ligand_smiles[test_indices]).long()
    test_complex = torch.from_numpy(complex_desc[test_indices]).float()
    test_affinities_np = affinities[test_indices]
    
    # Get consensus predictions
    consensus_results = lit_model.get_consensus_prediction(
        test_protein.to(lit_model.device),
        test_smiles.to(lit_model.device),
        test_complex.to(lit_model.device),
        n_samples=args.uncertainty_samples
    )
    
    # Evaluate consensus
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    final_pred = consensus_results['final_consensus']
    final_rmse = np.sqrt(mean_squared_error(test_affinities_np, final_pred))
    final_mae = mean_absolute_error(test_affinities_np, final_pred)
    final_pcc_result = pearsonr(test_affinities_np, final_pred)
    final_pcc = float(final_pcc_result[0])  # type: ignore
    
    print(f"\nTest Set Performance (Final Consensus):")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  MAE:  {final_mae:.4f}")
    print(f"  PCC:  {final_pcc:.4f}")
    print(f"  R²:   {final_pcc**2:.4f}")
    
    # Note: Plot results disabled for Lightning version
    # Use TensorBoard for visualization: tensorboard --logdir {args.log_dir}
    
    # ========================================================================
    # STEP 6: PREDICT ON CANDIDATE LIGANDS (if provided)
    # ========================================================================
    if args.candidate_ligands:
        print("\n[STEP 6] Predicting on Candidate Ligands")
        print("-" * 80)
        print(f"Loading candidates from: {args.candidate_ligands}")
        
        # This would load your 26 candidate ligands
        # For now, just a placeholder
        print("✓ Feature: Ready to predict on new ligands")
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY (PyTorch Lightning)")
    print("=" * 80)
    print(f"\nModel checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print(f"Test PCC (Consensus): {final_pcc:.4f}")
    print(f"\n💡 View training progress with:")
    print(f"   tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Bayesian Hybrid Affinity Network with PyTorch Lightning'
    )
    
    # Data arguments
    parser.add_argument('--bindingdb_path', type=str,
                        default='data/bindingdb_data/BindingDB_All.tsv',
                        help='Path to BindingDB TSV file')
    parser.add_argument('--target_name', type=str, default=None,
                        help='Target protein name to filter (e.g., ACVR1, kinase)')
    parser.add_argument('--refined_output_dir', type=str,
                        default='data/pdbbind_refined',
                        help='Output directory for refined set')
    parser.add_argument('--general_output_dir', type=str,
                        default='data/pdbbind_general',
                        help='Output directory for general set')
    parser.add_argument('--use_general_set', action='store_true',
                        help='Also process general set (larger dataset)')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of data')
    parser.add_argument('--max_protein_len', type=int, default=1000,
                        help='Maximum protein sequence length')
    parser.add_argument('--max_smiles_len', type=int, default=200,
                        help='Maximum SMILES length')
    
    # Model arguments
    parser.add_argument('--protein_dim', type=int, default=256,
                        help='Protein encoder output dimension')
    parser.add_argument('--ligand_dim', type=int, default=256,
                        help='Ligand encoder output dimension')
    parser.add_argument('--complex_dim', type=int, default=128,
                        help='Complex descriptor encoder output dimension')
    parser.add_argument('--fusion_dims', type=int, nargs='+',
                        default=[512, 256, 128],
                        help='Fusion layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--prior_sigma', type=float, default=1.0,
                        help='Prior standard deviation for Bayesian layers')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='Weight for KL divergence term')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training set fraction')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation set fraction')
    parser.add_argument('--uncertainty_samples', type=int, default=100,
                        help='Number of samples for uncertainty quantification')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str,
                        default='models/lightning_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str,
                        default='lightning_logs',
                        help='Directory to save logs')
    parser.add_argument('--candidate_ligands', type=str, default=None,
                        help='Path to candidate ligands for prediction')
    
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    main(args)

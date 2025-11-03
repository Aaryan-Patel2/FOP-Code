"""
Main Training Pipeline for Binding Affinity Prediction

This script:
1. Prepares molecular data with affinity annotations from BindingDB
2. Trains a deep learning model to predict binding affinities
3. Evaluates the model and makes predictions on candidate ligands
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_preparation import AffinityDataPreparator
from models.affinity_predictor import AffinityPredictor, AffinityPredictionTrainer


def plot_training_curves(trainer, save_path='models/training_curves.png'):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    ax.plot(epochs, trainer.train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, trainer.val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, save_path='models/prediction_scatter.png',
                     title='Predicted vs Actual Binding Affinity'):
    """Plot predicted vs actual values"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Plot diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    ax.set_xlabel('Actual ΔG (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted ΔG (kcal/mol)', fontsize=12)
    ax.set_title(f'{title}\nR² = {r2:.3f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction scatter plot saved to {save_path}")
    plt.close()


def main(args):
    """Main training pipeline"""
    
    print("=" * 80)
    print("BINDING AFFINITY PREDICTION TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Data Preparation
    print("\n[STEP 1] Preparing data...")
    print("-" * 80)
    
    preparator = AffinityDataPreparator(
        bindingdb_path=args.bindingdb_path
    )
    
    if not os.path.exists(os.path.join(args.output_dir, 'affinity_dataset.csv')):
        print("Processing BindingDB data...")
        df = preparator.prepare_dataset(
            output_dir=args.output_dir,
            target_name=args.target_name,
            min_affinity=args.min_affinity,
            max_affinity=args.max_affinity,
        )
    else:
        print(f"Using existing processed data from {args.output_dir}")
        df = pd.read_csv(os.path.join(args.output_dir, 'affinity_dataset.csv'))
    
    print(f"✓ Dataset prepared: {len(df)} samples")
    
    # Also prepare candidate ligands if available
    if args.ligands_smiles and os.path.exists(args.ligands_smiles):
        print(f"\nPreparing candidate ligands from {args.ligands_smiles}...")
        df_ligands = preparator.prepare_from_smiles_list(
            smiles_file=args.ligands_smiles,
            output_dir=args.ligands_output_dir
        )
        print(f"✓ Prepared {len(df_ligands)} candidate ligands")
    
    # Step 2: Model Initialization
    print("\n[STEP 2] Initializing model...")
    print("-" * 80)
    
    # Determine descriptor dimension from data
    descriptor_cols = [col for col in df.columns if col not in [
        'smiles', 'target', 'affinity_nM', 'affinity_type', 'delta_g_kcal_mol',
        'pKd', 'temperature_C', 'pH', 'pdb_id', 'kon', 'koff', 'residence_time_s'
    ]]
    descriptor_dim = len(descriptor_cols)
    
    print(f"Descriptor dimension: {descriptor_dim}")
    print(f"Fingerprint dimension: {args.fingerprint_dim}")
    
    model = AffinityPredictor(
        descriptor_dim=descriptor_dim,
        fingerprint_dim=args.fingerprint_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_batch_norm=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Model initialized")
    
    # Step 3: Training
    print("\n[STEP 3] Training model...")
    print("-" * 80)
    
    trainer = AffinityPredictionTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    train_loader, val_loader, test_loader = trainer.prepare_data(
        data_dir=args.output_dir,
        target_col=args.target_col,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir
    )
    
    print("✓ Training completed")
    
    # Step 4: Evaluation
    print("\n[STEP 4] Evaluating model...")
    print("-" * 80)
    
    # Load best model
    trainer.load_model(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.validate(test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_metrics['rmse']:.4f} kcal/mol")
    print(f"  MAE:  {test_metrics['mae']:.4f} kcal/mol")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    # Get predictions for plotting
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if isinstance(batch_x, tuple):
                batch_x = tuple(x.to(trainer.device) for x in batch_x)
            else:
                batch_x = batch_x.to(trainer.device)
            
            predictions = model(batch_x).cpu().numpy()
            all_preds.extend(predictions)
            all_targets.extend(batch_y.numpy())
    
    all_preds = trainer.scaler_targets.inverse_transform(
        np.array(all_preds).reshape(-1, 1)).flatten()
    all_targets = trainer.scaler_targets.inverse_transform(
        np.array(all_targets).reshape(-1, 1)).flatten()
    
    # Plot results
    plot_training_curves(trainer, save_path=os.path.join(args.checkpoint_dir, 'training_curves.png'))
    plot_predictions(all_targets, all_preds, 
                     save_path=os.path.join(args.checkpoint_dir, 'test_predictions.png'))
    
    print("✓ Evaluation completed")
    
    # Step 5: Predict on candidate ligands
    if args.ligands_smiles and os.path.exists(args.ligands_smiles):
        print("\n[STEP 5] Predicting affinities for candidate ligands...")
        print("-" * 80)
        
        # Load candidate ligands
        ligands_df = pd.read_csv(os.path.join(args.ligands_output_dir, 'ligand_features.csv'))
        ligands_fp = np.load(os.path.join(args.ligands_output_dir, 'ligand_fingerprints.npy'))
        
        # Extract features
        ligand_descriptor_cols = [col for col in ligands_df.columns 
                                   if col not in ['ligand_id', 'smiles']]
        ligand_features = ligands_df[ligand_descriptor_cols].fillna(0).values
        
        # Make predictions
        predicted_affinities = trainer.predict(ligand_features, ligands_fp)
        
        # Convert to Kd (nM)
        # ΔG = RT ln(Kd)  =>  Kd = exp(ΔG/RT)
        RT = 0.593  # kcal/mol at 298K
        predicted_kd_nM = np.exp(predicted_affinities / RT) * 1e9
        
        # Add to dataframe
        ligands_df['predicted_delta_g_kcal_mol'] = predicted_affinities
        ligands_df['predicted_Kd_nM'] = predicted_kd_nM
        ligands_df['predicted_pKd'] = -np.log10(predicted_kd_nM * 1e-9)
        
        # Save results
        output_file = os.path.join(args.ligands_output_dir, 'predicted_affinities.csv')
        ligands_df.to_csv(output_file, index=False)
        
        print(f"\nPredicted Affinities:")
        print(ligands_df[['ligand_id', 'predicted_delta_g_kcal_mol', 
                          'predicted_Kd_nM', 'predicted_pKd']].to_string())
        print(f"\n✓ Predictions saved to {output_file}")
        
        # Rank by affinity
        ranked = ligands_df.sort_values('predicted_delta_g_kcal_mol')
        print(f"\nTop 5 Candidates (strongest predicted binding):")
        print(ranked[['ligand_id', 'smiles', 'predicted_delta_g_kcal_mol', 
                     'predicted_Kd_nM']].head().to_string())
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train binding affinity predictor')
    
    # Data arguments
    parser.add_argument('--bindingdb_path', type=str, 
                        default='data/bindingdb_data/BindingDB_All.tsv',
                        help='Path to BindingDB TSV file')
    parser.add_argument('--output_dir', type=str, 
                        default='data/processed_affinity',
                        help='Output directory for processed data')
    parser.add_argument('--target_name', type=str, default='ACVR1',
                        help='Target protein name to filter')
    parser.add_argument('--min_affinity', type=float, default=0.1,
                        help='Minimum affinity in nM')
    parser.add_argument('--max_affinity', type=float, default=10000,
                        help='Maximum affinity in nM')
    parser.add_argument('--target_col', type=str, default='delta_g_kcal_mol',
                        help='Target column for prediction')
    
    # Candidate ligands
    parser.add_argument('--ligands_smiles', type=str,
                        default='data/initial_SMILES/SMILES_strings.txt',
                        help='Path to candidate ligands SMILES file')
    parser.add_argument('--ligands_output_dir', type=str,
                        default='data/processed_ligands',
                        help='Output directory for processed ligands')
    
    # Model arguments
    parser.add_argument('--fingerprint_dim', type=int, default=2048,
                        help='Fingerprint dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                        default=[512, 256, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set fraction')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='models/checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    main(args)

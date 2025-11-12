#!/usr/bin/env python3
"""
Transfer Learning Training Script for Affinity Prediction

Strategy:
1. Pre-train on diverse protein-ligand pairs (all targets)
2. Fine-tune on specific target (kinases)

This addresses the data scarcity problem by learning general
protein-ligand interaction patterns before specializing.
"""

import os
import sys
import argparse
from pathlib import Path

def train_pretrain_phase(args):
    """Phase 1: Pre-train on diverse protein-ligand data"""
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING ON DIVERSE PROTEIN-LIGAND DATA")
    print("="*80)
    print("\n📚 Loading broad dataset (no target filter)...")
    
    from quick_start import AffinityPredictor
    
    predictor = AffinityPredictor()
    
    # Pre-train on ALL available data (no target filter)
    checkpoint_path = predictor.train(
        bindingdb_path=args.data_path,
        target_name=None,  # No filter - use all targets
        num_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_lr,
        output_dir=args.pretrain_dir
    )
    
    print(f"\n✓ Pre-training complete!")
    print(f"  Checkpoint: {checkpoint_path}")
    
    return checkpoint_path


def train_finetune_phase(pretrain_checkpoint, args):
    """Phase 2: Fine-tune on target-specific data"""
    print("\n" + "="*80)
    print(f"PHASE 2: FINE-TUNING ON {args.target.upper()} DATA")
    print("="*80)
    print(f"\n🎯 Loading pre-trained model from: {pretrain_checkpoint}")
    
    from quick_start import AffinityPredictor
    
    # Load pre-trained model
    predictor = AffinityPredictor(checkpoint_path=pretrain_checkpoint)
    
    # Fine-tune on target-specific data with lower learning rate
    checkpoint_path = predictor.train(
        bindingdb_path=args.data_path,
        target_name=args.target,
        num_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.finetune_lr,  # Lower LR for fine-tuning
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Fine-tuning complete!")
    print(f"  Final model: {checkpoint_path}")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Transfer Learning Training for Affinity Prediction'
    )
    
    # Data arguments
    parser.add_argument('--target', type=str, default='kinase',
                        help='Target protein family for fine-tuning')
    parser.add_argument('--data-path', type=str, 
                        default='data/bindingdb_data/BindingDB_All.tsv',
                        help='Path to BindingDB data')
    
    # Pre-training arguments
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='Number of pre-training epochs')
    parser.add_argument('--pretrain-lr', type=float, default=5e-4,
                        help='Pre-training learning rate')
    parser.add_argument('--pretrain-dir', type=str, default='trained_models/pretrain',
                        help='Directory for pre-training checkpoints')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune-epochs', type=int, default=15,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate (lower than pre-training)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Directory to save final model')
    
    # Control arguments
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pre-training and use existing checkpoint')
    parser.add_argument('--pretrain-checkpoint', type=str,
                        help='Path to existing pre-trained checkpoint')
    
    args = parser.parse_args()
    
    print("\n🧬 Transfer Learning Pipeline for Affinity Prediction")
    print("="*80)
    print("\n📋 Configuration:")
    print(f"  Target for fine-tuning: {args.target}")
    print(f"  Pre-training: {args.pretrain_epochs} epochs @ LR={args.pretrain_lr}")
    print(f"  Fine-tuning: {args.finetune_epochs} epochs @ LR={args.finetune_lr}")
    print(f"  Batch size: {args.batch_size}")
    print("="*80)
    
    # Phase 1: Pre-training
    if args.skip_pretrain and args.pretrain_checkpoint:
        print(f"\n⏭  Skipping pre-training, using: {args.pretrain_checkpoint}")
        pretrain_checkpoint = args.pretrain_checkpoint
    else:
        pretrain_checkpoint = train_pretrain_phase(args)
    
    # Phase 2: Fine-tuning
    final_checkpoint = train_finetune_phase(pretrain_checkpoint, args)
    
    # Export for integration
    print("\n" + "="*80)
    print("EXPORTING MODEL")
    print("="*80)
    
    export_dir = Path('models/pretrained')
    export_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy(final_checkpoint, export_dir / 'affinity_predictor_transfer.ckpt')
    
    # Save metadata
    import json
    metadata = {
        'training_method': 'transfer_learning',
        'pretrain_epochs': args.pretrain_epochs,
        'finetune_epochs': args.finetune_epochs,
        'target': args.target,
        'pretrain_lr': args.pretrain_lr,
        'finetune_lr': args.finetune_lr,
        'final_checkpoint': str(final_checkpoint)
    }
    
    with open(export_dir / 'transfer_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Model exported to: {export_dir / 'affinity_predictor_transfer.ckpt'}")
    
    print("\n" + "="*80)
    print("✅ TRANSFER LEARNING COMPLETE!")
    print("="*80)
    print(f"\n🎯 Final model trained on {args.target} with transfer learning")
    print(f"📁 Model location: {final_checkpoint}")
    print("\nNext steps:")
    print("  1. Test predictions: python test_predictions.py")
    print("  2. Integrate with GCDM diffusion model")
    print("  3. Generate optimized molecules!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Simple training script for ACVR1 Affinity Predictor
Run with: python train_model.py
"""

import os
import sys
import argparse
from datetime import datetime

def check_environment():
    """Verify all dependencies are available"""
    print("="*60)
    print("Checking Environment")
    print("="*60)
    
    missing = []
    device = 'cpu'
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device.upper()}")
    except ImportError:
        missing.append('torch')
    
    try:
        from rdkit import Chem
        print(f"✓ RDKit: Available")
    except ImportError:
        missing.append('rdkit')
    
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print(f"✓ NumPy, Pandas, scikit-learn: Available")
    except ImportError as e:
        missing.append('numpy/pandas/scikit-learn')
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch rdkit numpy pandas scikit-learn scipy")
        sys.exit(1)
    
    print("\n✓ Environment ready!")
    return device

def check_data(data_path):
    """Check if BindingDB data is available"""
    print("\n" + "="*60)
    print("Checking Data")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"❌ BindingDB not found at: {data_path}")
        print("\n📥 Download with: bash download_data.sh")
        sys.exit(1)
    
    size_gb = os.path.getsize(data_path) / 1e9
    print(f"✓ BindingDB dataset found ({size_gb:.1f} GB)")
    return True

def train_model(args):
    """Train the affinity predictor"""
    from quick_start import AffinityPredictor
    
    print("\n" + "="*60)
    print("Training Affinity Predictor")
    print("="*60)
    
    print(f"\n📋 Training Configuration:")
    print(f"  Target: {args.target}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output dir: {args.output_dir}")
    
    # Initialize predictor
    predictor = AffinityPredictor()
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   This may take a while depending on your hardware")
    
    checkpoint_path = predictor.train(
        bindingdb_path=args.data_path,
        target_name=args.target,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices
    )
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {checkpoint_path}")
    
    return checkpoint_path

def export_model(checkpoint_path):
    """Export model for downstream use"""
    import torch
    import json
    import shutil
    
    print("\n" + "="*60)
    print("Exporting Model")
    print("="*60)
    
    # Create export directory
    export_dir = 'models/pretrained'
    os.makedirs(export_dir, exist_ok=True)
    
    # Copy model
    export_path = f"{export_dir}/affinity_predictor.ckpt"
    shutil.copy(checkpoint_path, export_path)
    
    # Get model info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    # Create metadata
    metadata = {
        'model_name': 'ACVR1_Affinity_Predictor',
        'model_type': 'Bayesian_Neural_Network',
        'target': 'ACVR1',
        'training_date': datetime.now().isoformat(),
        'model_path': export_path,
        'size_mb': round(model_size_mb, 2),
        'epochs': checkpoint.get('epoch', 'unknown'),
        'description': 'Pretrained affinity predictor for FOP drug discovery',
        'usage': f'predictor = AffinityPredictor(checkpoint_path="{export_path}")'
    }
    
    metadata_path = f"{export_dir}/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model exported to: {export_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"\n🔗 Ready for integration with diffusion model!")

def main():
    parser = argparse.ArgumentParser(
        description='Train ACVR1 Affinity Predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-path', type=str, 
                        default='data/bindingdb_data/BindingDB_All.tsv',
                        help='Path to BindingDB TSV file')
    parser.add_argument('--target', type=str, default='ACVR1',
                        help='Target protein name (use "None" for multi-target training on ALL targets)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4 - balanced for convergence and stability)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use (default: 1, use 2 for multi-GPU training)')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (auto, cpu, gpu, tpu)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--no-export', action='store_true',
                        help='Skip exporting model to models/pretrained/')
    
    args = parser.parse_args()
    
    # Convert string "None" to Python None for multi-target training
    if args.target == "None":
        args.target = None
    
    print("\n🧬 ACVR1 Affinity Predictor - Training Pipeline")
    print("=" * 60)
    
    # Check environment
    device = check_environment()
    
    # Check data
    check_data(args.data_path)
    
    # Train model
    checkpoint_path = train_model(args)
    
    # Export model (unless disabled)
    if not args.no_export:
        export_model(checkpoint_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nYour trained model is ready at: {checkpoint_path}")
    print(f"\nNext steps:")
    print(f"  1. Test predictions: python test_predictions.py")
    print(f"  2. Integrate with GCDM diffusion model")
    print(f"  3. Generate optimized molecules!")
    print()

if __name__ == '__main__':
    main()

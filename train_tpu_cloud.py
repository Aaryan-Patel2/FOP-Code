#!/usr/bin/env python3
"""
TPU Training Script for Cloud Environments (Google Colab or Cloud TPU)

This script handles TPU-specific setup and training.
Works around ARM64 limitations by using cloud TPUs.

Requirements:
- Google Colab with TPU runtime, OR
- Google Cloud TPU VM

Usage:
    # On Google Colab:
    !python train_tpu_cloud.py --target kinase --epochs 25
    
    # On Cloud TPU VM:
    python train_tpu_cloud.py --target kinase --epochs 25
"""

import os
import sys
import argparse
from pathlib import Path


def setup_tpu_environment():
    """Setup TPU environment and check availability (Cloud TPU VM)
    
    NOTE: This script must be run ON the Cloud TPU VM, not locally.
    torch_xla is only available on x86_64, not ARM64.
    """
    print("="*80)
    print("TPU ENVIRONMENT SETUP (Google Cloud TPU VM)")
    print("="*80)
    
    import platform
    arch = platform.machine()
    print(f"Architecture: {arch}")
    
    if arch == 'aarch64':
        print("\n⚠️  WARNING: You're running on ARM64 (aarch64)")
        print("   torch_xla is not available for ARM64!")
        print("\n📌 This script must be run on a Cloud TPU VM:")
        print("   1. Create TPU VM: ./setup_cloud_tpu.sh")
        print("   2. SSH into it: gcloud compute tpus tpu-vm ssh [TPU_NAME] --zone=[ZONE]")
        print("   3. Run this script there")
        return False, None
    
    # Check TPU-specific dependencies
    print("\n📦 Checking TPU dependencies...")
    
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print(f"✓ torch_xla installed: {torch_xla.__version__}")
    except ImportError:
        print("❌ torch_xla not found!")
        print("\n📥 Install with:")
        print("   pip install torch==2.0.0")
        print("   pip install torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/tpu-pytorch-releases/torch_xla_wheels.html")
        return False, None
    
    # Check TPU availability
    print("\n🔍 Detecting TPU devices...")
    try:
        device = xm.xla_device()
        num_devices = xm.xrt_world_size()
        print(f"✓ TPU device available: {device}")
        print(f"✓ Number of TPU cores: {num_devices}")
        
        # Test TPU with simple operation
        import torch
        test_tensor = torch.randn(2, 2).to(device)
        result = test_tensor + test_tensor
        xm.mark_step()  # Force execution
        print(f"✓ TPU test successful: {result.device}")
        
        return True, device
    except Exception as e:
        print(f"❌ TPU not available: {e}")
        print("\n💡 Make sure you're running on a Cloud TPU VM:")
        print("   gcloud compute tpus tpu-vm create [TPU_NAME] --accelerator-type=v3-8")
        return False, None


def train_with_tpu(args):
    """Train model using TPU acceleration"""
    import torch
    import pytorch_lightning as pl
    from quick_start import AffinityPredictor
    
    print("\n" + "="*80)
    print("TRAINING WITH TPU ACCELERATION")
    print("="*80)
    
    print(f"\n📋 Configuration:")
    print(f"  Target: {args.target}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  TPU cores: {args.tpu_cores}")
    
    # Create predictor
    predictor = AffinityPredictor()
    
    # Train with TPU-specific settings
    print(f"\n🚀 Starting TPU training...")
    
    checkpoint_path = predictor.train(
        bindingdb_path=args.data_path,
        target_name=args.target,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        accelerator='tpu',  # Use TPU
        devices=args.tpu_cores
    )
    
    return checkpoint_path


def train_fallback_cpu(args):
    """Fallback to CPU training if TPU unavailable"""
    print("\n" + "="*80)
    print("FALLBACK: TRAINING WITH CPU")
    print("="*80)
    
    from quick_start import AffinityPredictor
    
    predictor = AffinityPredictor()
    
    checkpoint_path = predictor.train(
        bindingdb_path=args.data_path,
        target_name=args.target,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='TPU Training for Affinity Prediction'
    )
    
    # Data arguments
    parser.add_argument('--target', type=str, default='kinase',
                        help='Target protein family')
    parser.add_argument('--data-path', type=str,
                        default='data/bindingdb_data/BindingDB_All.tsv',
                        help='Path to BindingDB data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (can be larger on TPU)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    
    # TPU arguments
    parser.add_argument('--tpu-cores', type=int, default=8,
                        help='Number of TPU cores to use')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU training even if TPU available')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    print("\n🧬 TPU-Accelerated Affinity Predictor Training")
    print("="*80)
    
    # Setup and check TPU
    tpu_available, tpu_device = setup_tpu_environment()
    
    # Train with TPU or fallback to CPU
    if tpu_available and not args.force_cpu:
        print("\n✅ Using TPU for training!")
        checkpoint_path = train_with_tpu(args)
    else:
        if args.force_cpu:
            print("\n⚠ CPU training forced by --force-cpu flag")
        else:
            print("\n⚠ TPU not available, falling back to CPU")
        checkpoint_path = train_fallback_cpu(args)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Model saved to: {checkpoint_path}")
    print("\n💡 Next steps:")
    print("  1. Test predictions: python test_predictions.py")
    print("  2. Integrate with diffusion model")
    print("\n📥 To download model from Cloud TPU VM:")
    print(f"  gcloud compute tpus tpu-vm scp \\")
    print(f"    [TPU_NAME]:{checkpoint_path} \\")
    print(f"    ./trained_models/ \\")
    print(f"    --zone=[ZONE]")


if __name__ == '__main__':
    main()

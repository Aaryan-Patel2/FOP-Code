"""
Simple test script for Bayesian Affinity Prediction System
Tests all fundamental components with minimal data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from models.bayesian_affinity_predictor import (
    ProteinCNN, LigandCNN, ComplexDescriptorEncoder,
    HybridBayesianAffinityNetwork
)
from models.pdbbind_data_preparation import (
    ProteinSequenceEncoder, SMILESEncoder, ComplexDescriptorCalculator
)
from models.bayesian_training_pipeline import (
    BayesianAffinityLoss, EnsembleMLModels
)

print("=" * 60)
print("BAYESIAN AFFINITY SYSTEM - FUNDAMENTAL TESTS")
print("=" * 60)

# Test 1: Data Encoders
print("\n[Test 1/5] Data Encoders...")
try:
    protein_encoder = ProteinSequenceEncoder()
    smiles_encoder = SMILESEncoder()
    descriptor_calc = ComplexDescriptorCalculator()
    
    # Test protein encoding
    test_protein = "MKTIIALSYIFCLVFA"
    protein_encoded = protein_encoder.encode(test_protein, max_length=50)
    assert protein_encoded.shape == (50,), f"Wrong protein shape: {protein_encoded.shape}"
    
    # Test SMILES encoding
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    smiles_encoded = smiles_encoder.encode(test_smiles, max_length=50)
    assert smiles_encoded.shape == (50,), f"Wrong SMILES shape: {smiles_encoded.shape}"
    
    # Test descriptor calculation
    descriptors = descriptor_calc.calculate_from_smiles(test_smiles)
    assert descriptors.shape == (200,), f"Wrong descriptor shape: {descriptors.shape}"
    
    print("✓ All encoders working correctly")
    print(f"  - Protein vocab size: {protein_encoder.vocab_size}")
    print(f"  - SMILES vocab size: {smiles_encoder.vocab_size}")
    print(f"  - Descriptor dimension: {len(descriptors)}")
except Exception as e:
    print(f"✗ Encoder test failed: {e}")
    sys.exit(1)

# Test 2: CNN Encoders
print("\n[Test 2/5] CNN Encoders...")
try:
    batch_size = 4
    
    # Protein CNN
    protein_cnn = ProteinCNN(vocab_size=25, embedding_dim=64, output_dim=128)
    protein_input = torch.randint(0, 25, (batch_size, 100))
    protein_output = protein_cnn(protein_input)
    assert protein_output.shape == (batch_size, 128), f"Wrong protein CNN output: {protein_output.shape}"
    
    # Ligand CNN
    ligand_cnn = LigandCNN(vocab_size=70, embedding_dim=64, output_dim=128)
    ligand_input = torch.randint(0, 70, (batch_size, 50))
    ligand_output = ligand_cnn(ligand_input)
    assert ligand_output.shape == (batch_size, 128), f"Wrong ligand CNN output: {ligand_output.shape}"
    
    # Complex descriptor encoder
    complex_encoder = ComplexDescriptorEncoder(input_dim=200, output_dim=64)
    complex_input = torch.randn(batch_size, 200)
    complex_output = complex_encoder(complex_input)
    assert complex_output.shape == (batch_size, 64), f"Wrong complex encoder output: {complex_output.shape}"
    
    print("✓ All CNN encoders working correctly")
    print(f"  - Protein CNN output: {protein_output.shape}")
    print(f"  - Ligand CNN output: {ligand_output.shape}")
    print(f"  - Complex encoder output: {complex_output.shape}")
except Exception as e:
    print(f"✗ CNN encoder test failed: {e}")
    sys.exit(1)

# Test 3: Bayesian Hybrid Network
print("\n[Test 3/5] Bayesian Hybrid Network...")
try:
    model = HybridBayesianAffinityNetwork(
        protein_vocab_size=25,
        ligand_vocab_size=70,
        complex_descriptor_dim=200,
        fusion_hidden_dims=[256, 128],
        dropout=0.3,
        prior_sigma=1.0
    )
    
    batch_size = 4
    protein_seq = torch.randint(0, 25, (batch_size, 100))
    ligand_smiles = torch.randint(0, 70, (batch_size, 50))
    complex_desc = torch.randn(batch_size, 200)
    
    # Forward pass
    predictions = model(protein_seq, ligand_smiles, complex_desc)
    assert predictions.shape == (batch_size,), f"Wrong prediction shape: {predictions.shape}"
    
    # KL divergence
    kl_divergence = model.kl_divergence()
    assert kl_divergence.item() > 0, "KL divergence should be positive"
    
    # Uncertainty quantification (small sample size for speed)
    mean_pred, std_pred = model.predict_with_uncertainty(
        protein_seq, ligand_smiles, complex_desc, n_samples=10
    )
    assert mean_pred.shape == (batch_size,), f"Wrong mean prediction shape: {mean_pred.shape}"
    assert std_pred.shape == (batch_size,), f"Wrong std prediction shape: {std_pred.shape}"
    assert torch.all(std_pred >= 0), "Standard deviations must be non-negative"
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✓ Bayesian network working correctly")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - KL divergence: {kl_divergence.item():.2f}")
    print(f"  - Mean uncertainty: {std_pred.mean().item():.4f}")
except Exception as e:
    print(f"✗ Bayesian network test failed: {e}")
    sys.exit(1)

# Test 4: Loss Function
print("\n[Test 4/5] ELBO Loss Function...")
try:
    criterion = BayesianAffinityLoss(kl_weight=0.01, dataset_size=100)
    
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    kl_divergence = torch.tensor(1000.0)
    
    loss, metrics = criterion(predictions, targets, kl_divergence)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"
    
    print("✓ Loss function working correctly")
    print(f"  - Total loss: {metrics['total_loss']:.4f}")
    print(f"  - KL weight: {criterion.kl_weight}")
except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    sys.exit(1)

# Test 5: Ensemble ML Models (minimal data)
print("\n[Test 5/5] Ensemble ML Models...")
try:
    ensemble = EnsembleMLModels()
    
    # Create minimal training data
    X_train = np.random.randn(20, 200)  # 20 samples
    y_train = np.random.randn(20)
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Test predictions
    X_test = np.random.randn(5, 200)
    predictions = ensemble.predict(X_test)
    assert predictions.shape == (5,), f"Wrong prediction shape: {predictions.shape}"
    assert not np.any(np.isnan(predictions)), "Predictions should not be NaN"
    
    # Get individual predictions
    individual_preds = ensemble.predict_all(X_test)
    assert 'rf' in individual_preds, "Missing RF predictions"
    assert 'gb' in individual_preds, "Missing GB predictions"
    assert 'dtboost' in individual_preds, "Missing DTBoost predictions"
    
    print("✓ Ensemble models working correctly")
    print(f"  - Random Forest predictions: {individual_preds['rf'][:3]}")
    print(f"  - Gradient Boosting predictions: {individual_preds['gb'][:3]}")
    print(f"  - DTBoost predictions: {individual_preds['dtboost'][:3]}")
except Exception as e:
    print(f"✗ Ensemble test failed: {e}")
    sys.exit(1)

# Test 6: End-to-End Mini Pipeline
print("\n[Test 6/6] End-to-End Mini Pipeline...")
try:
    # Encode real molecules
    test_protein = "MKTIIALSYIFCLVFAGGRHAPLQWERTYL"
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    
    protein_seq = torch.tensor(protein_encoder.encode(test_protein, max_length=100)).unsqueeze(0)
    ligand_seq = torch.tensor(smiles_encoder.encode(test_smiles, max_length=50)).unsqueeze(0)
    complex_feat = torch.tensor(descriptor_calc.calculate_from_smiles(test_smiles)).unsqueeze(0)
    
    # Get Bayesian prediction with uncertainty (use smaller batch by duplicating)
    protein_batch = protein_seq.repeat(4, 1)
    ligand_batch = ligand_seq.repeat(4, 1)
    complex_batch = complex_feat.repeat(4, 1)
    
    model.eval()
    with torch.no_grad():
        mean_pred, std_pred = model.predict_with_uncertainty(
            protein_batch, ligand_batch, complex_batch, n_samples=20
        )
    
    # Get ensemble prediction
    ensemble_pred = ensemble.predict(complex_feat.numpy())
    
    # Consensus (60% Bayesian, 40% Ensemble)
    consensus = 0.6 * mean_pred[0].item() + 0.4 * ensemble_pred[0]
    
    print("✓ End-to-end pipeline working correctly")
    print(f"  - Test protein: {test_protein[:20]}...")
    print(f"  - Test ligand (SMILES): {test_smiles[:30]}...")
    print(f"  - Bayesian prediction: {mean_pred[0].item():.4f} ± {std_pred[0].item():.4f}")
    print(f"  - Ensemble prediction: {ensemble_pred[0]:.4f}")
    print(f"  - Consensus prediction: {consensus:.4f}")
except Exception as e:
    print(f"✗ End-to-end test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nSystem is ready for full training.")
print("To train on real data, run:")
print("  python3 main/train_bayesian_affinity.py --num_epochs 5 --batch_size 16")

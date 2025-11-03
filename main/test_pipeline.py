"""
Quick demo script to test the affinity prediction pipeline
Use this to verify everything is working before full training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_preparation import AffinityDataPreparator
import pandas as pd

print("=" * 80)
print("AFFINITY PREDICTION PIPELINE - QUICK TEST")
print("=" * 80)

# Test 1: Data preparation from SMILES
print("\n[TEST 1] Preparing candidate ligands...")
print("-" * 80)

preparator = AffinityDataPreparator()

smiles_file = 'data/initial_SMILES/SMILES_strings.txt'
if os.path.exists(smiles_file):
    df_ligands = preparator.prepare_from_smiles_list(
        smiles_file=smiles_file,
        output_dir='data/processed_ligands_test'
    )
    print(f"\n✓ Successfully processed {len(df_ligands)} ligands")
    print("\nSample features:")
    print(df_ligands[['ligand_id', 'mol_weight', 'logp', 'tpsa', 'qed']].head())
else:
    print(f"⚠ SMILES file not found: {smiles_file}")

# Test 2: Check BindingDB data
print("\n[TEST 2] Checking BindingDB data...")
print("-" * 80)

bindingdb_path = 'data/bindingdb_data/BindingDB_All.tsv'
if os.path.exists(bindingdb_path):
    # Load just a few rows to test
    df_sample = pd.read_csv(bindingdb_path, sep='\t', nrows=10,
                            usecols=['Ligand SMILES', 'Target Name', 'Ki (nM)', 
                                    'IC50 (nM)', 'Kd (nM)'])
    print(f"✓ BindingDB file found")
    print(f"Sample entries:\n{df_sample.head()}")
    
    print("\n✓ BindingDB file is readable")
    print("  Note: Use --target_name in training to filter by your target")
    print("  Suggestion: Start with --target_name 'kinase' for broader training data")
else:
    print(f"⚠ BindingDB file not found: {bindingdb_path}")
    print("   Download from: https://www.bindingdb.org/bind/downloads.jsp")

# Test 3: Check for PyTorch
print("\n[TEST 3] Checking dependencies...")
print("-" * 80)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not found. Install with: pip install torch")

try:
    from rdkit import Chem
    print(f"✓ RDKit available")
except ImportError:
    print("✗ RDKit not found. Install with: conda install -c conda-forge rdkit")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn not found. Install with: pip install scikit-learn")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib not found. Install with: pip install matplotlib")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nIf all tests passed, you can run the full training pipeline:")
print("  python3 main/train_affinity_predictor.py")
print("\nFor a quick test with minimal data:")
print("  python3 main/train_affinity_predictor.py --num_epochs 10 --max_affinity 1000")

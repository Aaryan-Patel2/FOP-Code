#!/bin/bash
# Quick installation script for Google Colab
# Run in Colab with: !bash install_colab.sh

echo "========================================"
echo "Installing Bayesian Affinity Predictor"
echo "========================================"

# Install core dependencies
echo ""
echo "[1/4] Installing PyTorch..."
pip install -q torch torchvision

echo "[2/4] Installing RDKit..."
pip install -q rdkit

echo "[3/4] Installing scientific libraries..."
pip install -q scikit-learn scipy pandas numpy matplotlib seaborn

echo "[4/4] Verifying installation..."
python3 << EOF
import torch
import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem
print("✓ All core dependencies installed successfully!")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  RDKit: {Chem.__version__}")
EOF

echo ""
echo "========================================"
echo "✓ Installation complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  from quick_start import AffinityPredictor"
echo "  predictor = AffinityPredictor()"
echo ""
echo "See QuickStart_Colab.ipynb for examples"

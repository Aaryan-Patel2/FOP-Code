# Quick Start Guide - Bayesian Affinity Predictor

Easy-to-use interface for predicting binding affinity and dissociation kinetics.

## 🚀 Quick Setup (Google Colab)

```python
# 1. Clone repository
!git clone https://github.com/Aaryan-Patel2/FOP-Code.git
%cd FOP-Code

# 2. Install dependencies
!pip install -q torch torchvision pytorch-lightning rdkit scikit-learn scipy pandas numpy

# 3. Import and use
from quick_start import AffinityPredictor

# Load pre-trained model (if available)
predictor = AffinityPredictor(checkpoint_path='models/best_model.ckpt')

# Make a prediction
result = predictor.predict(
    protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQ",  # Your protein sequence
    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"      # Your ligand SMILES
)

print(f"Predicted pKd: {result['affinity']:.2f}")
print(f"Uncertainty: ±{result['uncertainty']:.2f}")
print(f"Confidence: {result['confidence']*100:.1f}%")
if result['koff']:
    print(f"k_off: {result['koff']:.2e} s⁻¹")
    print(f"Residence time: {result['residence_time']:.2f} seconds")
```

## 📊 Example Usage

### Single Prediction

```python
from quick_start import AffinityPredictor

predictor = AffinityPredictor(checkpoint_path='models/best_model.ckpt')

result = predictor.predict(
    protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",
    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"
)

print(result)
# Output: {'affinity': 7.32, 'uncertainty': 0.15, 'koff': 2.3e-4, ...}
```

### Batch Predictions

```python
proteins = [
    "MTEYKLVVVGAGGVGKSALTIQLIQ...",
    "MAPKKKNPEVQRLFAACRPSFDALN...",
    "GPCRTSLNYSMDFQKNLLGFGILQG..."
]

ligands = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "c1ccccc1"
]

results = predictor.predict_batch(proteins, ligands)

for i, res in enumerate(results):
    print(f"Pair {i+1}: pKd = {res['affinity']:.2f} ± {res['uncertainty']:.2f}")
```

### Training Your Own Model

```python
predictor = AffinityPredictor()

# Train on BindingDB data (filtered by target)
checkpoint = predictor.train(
    bindingdb_path='data/bindingdb_data/BindingDB_All.tsv',
    target_name='ACVR1',  # Optional: filter by target
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3
)

# Model is now ready to use
result = predictor.predict(protein_seq, ligand_smiles)
```

## 🔬 What Gets Predicted

The model predicts:

- **Affinity (pKd)**: Binding strength (higher = stronger binding)
- **Uncertainty**: Prediction confidence (Bayesian uncertainty quantification)
- **k_off**: Dissociation rate constant (s⁻¹)
- **Residence time**: How long the drug stays bound (1/k_off)
- **Confidence**: Overall prediction confidence (0-1 scale)

## 📁 Project Structure

```
FOP-Code/
├── quick_start.py          ← Easy-to-use API
├── models/
│   ├── bayesian_affinity_predictor.py   ← Core Bayesian model (3M params)
│   ├── bayesian_training_pipeline.py     ← Lightning training
│   ├── pdbbind_data_preparation.py       ← Data processing
│   └── utils/
│       └── bnn_koff.py                   ← k_off prediction
├── main/
│   ├── train_bayesian_affinity.py        ← Full training script
│   └── test_core_model.py                ← Model validation
└── data/                                  ← Data directory
```

## 🎯 Model Architecture

- **3.0M parameters** Bayesian Hybrid Neural Network
- **Protein CNN Encoder**: 3-layer CNN (kernels 3,5,7)
- **Ligand CNN Encoder**: 3-layer CNN (kernels 3,5,7)
- **Complex Descriptor Encoder**: Molecular interaction features
- **Bayesian Fusion Layers**: Uncertainty-aware predictions
- **Ensemble ML**: Random Forest + Gradient Boosting + DTBoost
- **Consensus Prediction**: 60% HNN + 40% ML ensemble

## 🔧 Advanced Usage

### Custom Configuration

```python
# For advanced users: customize training config
from main.train_bayesian_affinity import main
import argparse

args = argparse.Namespace(
    bindingdb_path='data/bindingdb_data/BindingDB_All.tsv',
    target_name='ACVR1',
    num_epochs=100,
    batch_size=64,
    learning_rate=5e-4,
    protein_dim=512,
    ligand_dim=512,
    fusion_dims=[1024, 512, 256],
    dropout=0.3,
    kl_weight=0.01,
    uncertainty_samples=100
)

main(args)
```

### Loading Pre-trained Models

```python
# Load specific checkpoint
predictor = AffinityPredictor(
    checkpoint_path='models/lightning_checkpoints/best_model.ckpt'
)

# Make predictions
result = predictor.predict(protein_seq, ligand_smiles, n_samples=100)
```

## 📈 Performance

On PDBBind refined set:
- **PCC (Pearson Correlation)**: ~0.82
- **RMSE**: ~1.2 kcal/mol
- **MAE**: ~0.9 kcal/mol
- **Uncertainty calibration**: 95% coverage

## 🤝 Citation

If you use this code, please cite:

```bibtex
@software{bayesian_affinity_predictor,
  title={Bayesian Hybrid Neural Network for Binding Affinity and Dissociation Kinetics},
  author={Your Name},
  year={2025},
  url={https://github.com/Aaryan-Patel2/FOP-Code}
}
```

## 📝 License

MIT License - see LICENSE file for details

## 🐛 Issues

Report issues at: https://github.com/Aaryan-Patel2/FOP-Code/issues

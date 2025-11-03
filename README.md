# Bayesian Affinity Predictor for Drug Discovery

Predict binding affinity and dissociation kinetics for drug-target interactions using Bayesian Neural Networks.

## 🎯 Goals
- Balance binding affinity and k_off to create drugs with selective inhibition but temporary binding
- Ensure BMP signaling is not completely inhibited (critical for FOP treatment)
- Use Bayesian uncertainty quantification for confident predictions
- Enable de novo drug generation guided by affinity and kinetics

## 🚀 Quick Start

**For Google Colab** (easiest):
1. Open `QuickStart_Colab.ipynb` in Colab
2. Run the setup cells
3. Start making predictions!

**For local Python**:
```python
from quick_start import AffinityPredictor

# Load pre-trained model
predictor = AffinityPredictor(checkpoint_path='models/best_model.ckpt')

# Make prediction
result = predictor.predict(
    protein_sequence="MTEYKLVVVGAGG...",
    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"
)

print(f"pKd: {result['affinity']:.2f} ± {result['uncertainty']:.2f}")
print(f"k_off: {result['koff']:.2e} s⁻¹")
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed instructions.

## 📊 What It Predicts

- **Binding Affinity (pKd)**: Strength of protein-ligand interaction
- **Uncertainty**: Bayesian confidence in prediction
- **Dissociation Rate (k_off)**: How fast the drug unbinds
- **Residence Time**: How long the drug stays bound (1/k_off)

## 🏗️ Model Architecture

- **3.0M parameters** Bayesian Hybrid Neural Network
- **Protein CNN**: 3-layer encoder (kernels 3, 5, 7)
- **Ligand CNN**: 3-layer encoder (kernels 3, 5, 7)  
- **Complex Descriptors**: Molecular interaction features
- **Bayesian Fusion**: Uncertainty-aware prediction
- **Ensemble ML**: RF + GB + DTBoost consensus (60% HNN + 40% ML)

## 📁 Repository Structure

```
FOP-Code/
├── quick_start.py              ← Simple API for predictions
├── QuickStart_Colab.ipynb      ← Colab notebook tutorial
├── QUICKSTART.md               ← Detailed documentation
├── requirements.txt            ← Dependencies
├── install_colab.sh            ← Colab installation script
│
├── models/
│   ├── bayesian_affinity_predictor.py    ← Core Bayesian model
│   ├── bayesian_training_pipeline.py     ← PyTorch Lightning training
│   ├── pdbbind_data_preparation.py       ← Data preprocessing
│   └── utils/
│       └── bnn_koff.py                   ← k_off prediction module
│
├── main/
│   ├── train_bayesian_affinity.py        ← Full training pipeline
│   ├── test_core_model.py                ← Model validation
│   └── test_lightning_integration.py     ← Integration tests
│
└── data/                                  ← Data directory
    └── bindingdb_data/                    ← BindingDB dataset
```

## 🔧 Installation

### Google Colab (Recommended)
```bash
!git clone https://github.com/Aaryan-Patel2/FOP-Code.git
%cd FOP-Code
!bash install_colab.sh
```

### Local Installation
```bash
git clone https://github.com/Aaryan-Patel2/FOP-Code.git
cd FOP-Code
pip install -r requirements.txt
```

## 🧪 Testing

Run tests to verify installation:
```bash
# Test core model (no Lightning required)
python3 main/test_core_model.py

# Test full integration (requires Lightning)
python3 main/test_lightning_integration.py
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Usage examples and API reference
- **[LIGHTNING_REFACTOR.md](LIGHTNING_REFACTOR.md)** - PyTorch Lightning integration details
- **[docs/BAYESIAN_AFFINITY_README.md](docs/BAYESIAN_AFFINITY_README.md)** - Model architecture
- **[docs/AFFINITY_PREDICTION_SUMMARY.md](docs/AFFINITY_PREDICTION_SUMMARY.md)** - Training details

## 🎓 Citation

If you use this code, please cite:
```bibtex
@software{bayesian_affinity_predictor,
  title={Bayesian Hybrid Neural Network for Binding Affinity and Dissociation Kinetics},
  year={2025},
  url={https://github.com/Aaryan-Patel2/FOP-Code}
}
```

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or issues: https://github.com/Aaryan-Patel2/FOP-Code/issues
# FOP Affinity Predictor# Bayesian Affinity Predictor for Drug Discovery



A Bayesian neural network module for predicting protein-ligand binding affinity and dissociation kinetics. Designed as a reusable library for integration with molecule generation pipelines.Predict binding affinity and dissociation kinetics for drug-target interactions using Bayesian Neural Networks.



## 🎯 Purpose## 🎯 Goals



This module provides:### Scientific Objective

- **Affinity Prediction**: Bayesian neural network with uncertainty quantificationDesign FOP inhibitors with **"fast kinetics"** profile:

- **Kinetics Estimation**: k_off and residence time prediction- **Moderate-to-good affinity** (pKd ~7-8, K_d ~10-100 nM) - Binds effectively to mutant ACVR1

- **Easy Integration**: Clean API for use in other projects (e.g., GCDM diffusion models)- **Fast dissociation** (high k_off ~0.1-1 s⁻¹) - Unbinds quickly after inhibition

- **Short residence time** (~1-10 seconds) - Prevents prolonged BMP pathway blockage

## 📦 Installation

### Why This Matters for FOP

```bashTraditional drugs aim for **tight binding + long residence time**. For FOP, we need the opposite:

# Clone repository- ✅ Bind to inhibit aberrant bone formation

git clone https://github.com/Aaryan-Patel2/FOP-Code.git- ✅ Dissociate quickly to allow normal BMP signaling

cd FOP-Code- ✅ Prevent complete pathway shutdown (which would cause other issues)



# Install dependencies### Implementation Status

pip install -r requirements.txt- ✅ **Affinity prediction** - Fully implemented with Bayesian uncertainty quantification

- 🔄 **k_off prediction** - Planned for future release (requires kinetics data)

# Download training data (optional, only for training)- ✅ **API ready** - Future-proof design supports both metrics

bash download_data.sh

```## 🚀 Quick Start



## 🚀 Quick Start**For Google Colab** (easiest):



### As a Library (Use in Your Project)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aaryan-Patel2/FOP-Code/blob/main/Complete_Colab_Tutorial.ipynb)



```python1. Click the badge above or open `Complete_Colab_Tutorial.ipynb` in Colab

from fop_affinity import AffinityPredictor2. Run all cells (Runtime → Run all)

3. Start making predictions and training models!

# Load pretrained model

predictor = AffinityPredictor(checkpoint_path='models/pretrained/affinity_predictor.ckpt')📖 See `COLAB_USAGE_GUIDE.md` for detailed instructions



# Make prediction**For local Python**:

result = predictor.predict(```python

    protein_sequence="MTEYKLVVVG...",from quick_start import AffinityPredictor

    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"

)# Load pre-trained model

predictor = AffinityPredictor(checkpoint_path='models/best_model.ckpt')

print(f"Affinity: {result['affinity']:.2f} pKd")

print(f"k_off: {result['koff']:.3f} s⁻¹")# Make prediction

```result = predictor.predict(

    protein_sequence="MTEYKLVVVGAGG...",

### Training Your Own Model    ligand_smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O"

)

```bash

# Train on ACVR1 dataprint(f"pKd: {result['affinity']:.2f} ± {result['uncertainty']:.2f}")

python train_model.pyprint(f"k_off: {result['koff']:.3f} s⁻¹")

print(f"Residence time: {result['residence_time']:.1f} seconds")

# Test predictionsprint(f"Confidence: {result['confidence']:.2%}")

python test_predictions.py```

```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed instructions.

## 📁 Project Structure

## 📊 What It Predicts

```

FOP-Code/**Currently Available:**

├── models/              # Core model implementations- **Binding Affinity (pKd)**: Strength of protein-ligand interaction ✅

│   ├── bayesian_affinity_predictor.py- **Uncertainty**: Bayesian confidence in prediction ✅

│   ├── bayesian_training_pipeline.py- **Confidence Score**: Overall prediction reliability ✅

│   ├── data_preparation.py

│   └── utils/          # k_off prediction utilities**Kinetics Prediction (Empirical Method):**

├── quick_start.py      # Main API (AffinityPredictor class)- **Dissociation Rate (k_off)**: How fast the drug unbinds ✅

├── train_model.py      # Training script- **Residence Time**: How long the drug stays bound (1/k_off) ✅

├── test_predictions.py # Testing script- **k_on Estimation**: Association rate derived from K_d and k_off ✅

├── requirements.txt    # Dependencies

└── README.md          # This file*Note: Currently uses literature-based empirical correlations. ML/Bayesian methods available for training with kinetics data.*

```

## 🏗️ Model Architecture

## 🔧 API Reference

- **3.0M parameters** Bayesian Hybrid Neural Network

### AffinityPredictor- **Protein CNN**: 3-layer encoder (kernels 3, 5, 7)

- **Ligand CNN**: 3-layer encoder (kernels 3, 5, 7)  

Main class for affinity prediction.- **Complex Descriptors**: Molecular interaction features

- **Bayesian Fusion**: Uncertainty-aware prediction

**Methods:**- **Ensemble ML**: RF + GB + DTBoost consensus (60% HNN + 40% ML)



- `predict(protein_sequence, ligand_smiles, n_samples=100)`## 📁 Repository Structure

  - Returns: dict with 'affinity', 'uncertainty', 'koff', 'residence_time', 'confidence'

```

- `train(bindingdb_path, target_name, num_epochs=20, batch_size=32, ...)`FOP-Code/

  - Trains model on BindingDB data├── quick_start.py              ← Simple API for predictions

  - Returns: path to saved checkpoint├── QuickStart_Colab.ipynb      ← Colab notebook tutorial

├── QUICKSTART.md               ← Detailed documentation

## 🔗 Integration Example├── requirements.txt            ← Dependencies

├── install_colab.sh            ← Colab installation script

Use in a diffusion model pipeline:│

├── models/

```python│   ├── bayesian_affinity_predictor.py    ← Core Bayesian model

from fop_affinity import AffinityPredictor│   ├── bayesian_training_pipeline.py     ← PyTorch Lightning training

from gcdm import MoleculeGenerator  # Your diffusion model│   ├── pdbbind_data_preparation.py       ← Data preprocessing

│   └── utils/

# Load affinity predictor│       └── bnn_koff.py                   ← k_off prediction module

affinity_model = AffinityPredictor(checkpoint_path='models/pretrained/affinity_predictor.ckpt')│

├── main/

# Use as scoring function in generation│   ├── train_bayesian_affinity.py        ← Full training pipeline

def score_molecule(protein_seq, ligand_smiles):│   ├── test_core_model.py                ← Model validation

    result = affinity_model.predict(protein_seq, ligand_smiles)│   └── test_lightning_integration.py     ← Integration tests

    return result['affinity']│

└── data/                                  ← Data directory

# Integrate with diffusion model    └── bindingdb_data/                    ← BindingDB dataset

generator = MoleculeGenerator(scoring_function=score_molecule)```

molecules = generator.generate(target_protein="ACVR1_sequence")

```## 🔧 Installation



## 📊 Training Options### Google Colab (Recommended)

```bash

```bash# Clone and install

# CPU-friendly (smaller batch, fewer epochs)!git clone https://github.com/Aaryan-Patel2/FOP-Code.git

python train_model.py --epochs 10 --batch-size 16%cd FOP-Code

!bash install_colab.sh

# Full training (better accuracy)

python train_model.py --epochs 50 --batch-size 32# (Optional) Download BindingDB for training

!bash download_data.sh

# Different target```

python train_model.py --target "EGFR"

### Local Installation

# See all options```bash

python train_model.py --helpgit clone https://github.com/Aaryan-Patel2/FOP-Code.git

```cd FOP-Code

pip install -r requirements.txt

## 📖 Documentation

# (Optional) Download BindingDB for training

- [Binding Kinetics Explained](docs/BINDING_KINETICS_EXPLAINED.md)bash download_data.sh

- [k_off Implementation](docs/KOFF_IMPLEMENTATION_SUMMARY.md)```

- [Bayesian Model Details](docs/BAYESIAN_AFFINITY_README.md)

- [Training Guide](TRAINING_README.md)**Note**: BindingDB dataset (6.3GB) is **not included** in the repository. Use `download_data.sh` to download it automatically, or train on a small subset for testing.



## 🧪 Testing## 🧪 Testing



```bashRun tests to verify installation:

# Run basic prediction test```bash

python test_predictions.py# Test core model (no Lightning required)

python3 main/test_core_model.py

# Or use pytest for comprehensive tests

pytest main/test_pipeline.py# Test full integration (requires Lightning)

```python3 main/test_lightning_integration.py

```

## 📝 Requirements

## 📚 Documentation

- Python 3.8+

- PyTorch 1.9+- **[QUICKSTART.md](QUICKSTART.md)** - Usage examples and API reference

- RDKit- **[LIGHTNING_REFACTOR.md](LIGHTNING_REFACTOR.md)** - PyTorch Lightning integration details

- NumPy, Pandas, scikit-learn- **[docs/BAYESIAN_AFFINITY_README.md](docs/BAYESIAN_AFFINITY_README.md)** - Model architecture

- See `requirements.txt` for complete list- **[docs/AFFINITY_PREDICTION_SUMMARY.md](docs/AFFINITY_PREDICTION_SUMMARY.md)** - Training details



## 🎓 Background## 🎓 Citation



Developed for Fibrodysplasia Ossificans Progressiva (FOP) drug discovery, focusing on:If you use this code, please cite:

- Moderate affinity (pKd 7-8)```bibtex

- Fast dissociation (k_off 0.1-1 s⁻¹)@software{bayesian_affinity_predictor,

- Residence time 1-10 seconds  title={Bayesian Hybrid Neural Network for Binding Affinity and Dissociation Kinetics},

  year={2025},

The model can be adapted for other therapeutic targets and integrated into various drug discovery pipelines.  url={https://github.com/Aaryan-Patel2/FOP-Code}

}
```

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or issues: https://github.com/Aaryan-Patel2/FOP-Code/issues
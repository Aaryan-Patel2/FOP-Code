# Project Structure

Clean, modular organization for the FOP Affinity Predictor library.

## 📁 Directory Structure

```
FOP-Code/
├── models/                          # Core model implementations
│   ├── __init__.py                  # Package exports
│   ├── bayesian_affinity_predictor.py   # Bayesian Neural Network
│   ├── random_forest_model.py       # Random Forest model
│   ├── gradient_boosting_models.py  # GB and DTBoost models  
│   ├── ensemble_model.py            # Complete ensemble (all models combined)
│   ├── data_preparation.py          # Data loading and preprocessing
│   ├── pdbbind_data_preparation.py  # PDBBind format preparation
│   ├── utils/                       # Utilities
│   │   ├── losses.py               # Loss functions (ELBO, MSE, etc.)
│   │   ├── metrics.py              # Evaluation metrics (PCC, RMSE, MAE)
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── bnn_koff.py            # k_off prediction utilities
│   │   ├── fix_lzma.py            # LZMA module fix
│   │   ├── prepare_ligands.py     # Ligand preparation
│   │   └── scoring.py             # Docking scoring functions
│   └── generator/                   # [Reserved for diffusion model integration]
├── main/                            # Testing and validation scripts
│   ├── test_bayesian_system.py
│   ├── test_pipeline.py
│   ├── train_affinity_predictor.py
│   └── train_bayesian_affinity.py
├── data/                            # Data directory
│   ├── bindingdb_data/             # BindingDB dataset
│   ├── structures/                  # Protein structures
│   └── [other data folders]
├── docs/                            # Documentation
│   ├── BAYESIAN_AFFINITY_README.md
│   ├── BINDING_KINETICS_EXPLAINED.md
│   └── KOFF_IMPLEMENTATION_SUMMARY.md
├── quick_start.py                   # Main API (AffinityPredictor class)
├── train_model.py                   # Simple training script
├── test_predictions.py              # Testing script
├── requirements.txt                 # Dependencies
├── TRAINING_README.md              # Training guide
└── README.md                        # Main documentation
```

## 🎯 Model Files

### Individual Models (Easy to Debug)

1. **`bayesian_affinity_predictor.py`** - Bayesian Neural Network
   - `BayesianLinear`: Bayesian linear layers with uncertainty
   - `ProteinCNN`: Protein sequence encoder
   - `LigandCNN`: Ligand SMILES encoder
   - `HybridBayesianAffinityNetwork`: Complete BNN architecture
   - `create_hnn_affinity_model()`: Factory function

2. **`random_forest_model.py`** - Random Forest
   - `RandomForestAffinityModel`: RF regressor for molecular descriptors
   - Methods: `train()`, `predict()`, `get_feature_importance()`

3. **`gradient_boosting_models.py`** - Gradient Boosting
   - `GradientBoostingAffinityModel`: Standard GB
   - `DTBoostAffinityModel`: Deeper trees, slower learning (for diversity)
   - Methods: `train()`, `predict()`

4. **`ensemble_model.py`** - Complete Ensemble
   - `EnsembleAffinityPredictor`: Combines all models
   - Methods: `train_bnn()`, `train_ml_models()`, `predict_ensemble()`
   - Ensemble weights: BNN (60%), RF (15%), GB (15%), DTBoost (10%)

## 🛠️ Utilities

### `utils/losses.py`
- `BayesianAffinityLoss`: ELBO loss for Bayesian training
- `create_loss_function()`: Factory for different loss types

### `utils/metrics.py`
- `calculate_metrics()`: PCC, RMSE, MAE
- `calculate_pcc()`, `calculate_rmse()`, `calculate_mae()`

### `utils/dataset.py`
- `AffinityDataset`: PyTorch Dataset for protein-ligand data

### `utils/bnn_koff.py`
- k_off prediction from affinity
- Residence time estimation
- FOP suitability scoring

### `utils/fix_lzma.py`
- Auto-fixes lzma import issues
- Import at top of any module using pandas

## 🔧 Usage Examples

### Use Individual Models

```python
# Just Random Forest
from models.random_forest_model import RandomForestAffinityModel

rf_model = RandomForestAffinityModel()
rf_model.train(X_train, y_train)
predictions = rf_model.predict(X_test)
```

### Use Complete Ensemble

```python
from models.ensemble_model import EnsembleAffinityPredictor

ensemble = EnsembleAffinityPredictor()

# Train BNN
ensemble.train_bnn(train_loader, val_loader, num_epochs=20)

# Train ML models
ensemble.train_ml_models(X_train, y_train)

# Predict with full ensemble
result = ensemble.predict_ensemble(protein_seq, ligand_smiles, descriptors)
print(f"Affinity: {result['affinity']} ± {result['uncertainty']}")
```

### Use via Main API

```python
from quick_start import AffinityPredictor

predictor = AffinityPredictor()
predictor.train(bindingdb_path='data/bindingdb_data/BindingDB_All.tsv', target_name='ACVR1')

result = predictor.predict(protein_seq, ligand_smiles)
```

## 📝 Key Features

✅ **Modular**: Each model in separate file for easy debugging  
✅ **Clean**: No deprecated code, Colab-specific content removed  
✅ **Focused**: Library purpose - easy to integrate into other projects  
✅ **Documented**: Clear docstrings and comments  
✅ **Tested**: Test scripts in `main/` folder  
✅ **Ready**: For GCDM/diffusion model integration

## 🔗 Integration Ready

The `models/generator/` folder is reserved for future diffusion model integration. When ready:
1. Add GCDM components to `models/generator/`
2. Use `EnsembleAffinityPredictor` as scoring function
3. Guide molecule generation with predicted affinity

## 🧪 Testing

```bash
# Test individual models
python main/test_bayesian_system.py

# Test complete pipeline
python main/test_pipeline.py

# Quick prediction test
python test_predictions.py
```

## 📦 As a Library

Install as package:
```bash
pip install -e .  # From repository root
```

Use in other projects:
```python
from fop_affinity import AffinityPredictor
predictor = AffinityPredictor(checkpoint_path='path/to/model.ckpt')
```

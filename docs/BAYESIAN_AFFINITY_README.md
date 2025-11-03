# Bayesian Hybrid Neural Network for Binding Affinity Prediction

## Overview

This is a complete implementation of a **Bayesian Hybrid Neural Network (HNN-Affinity)** for protein-ligand binding affinity prediction, following the architecture from state-of-the-art research papers.

The system combines:
- **Convolutional Neural Networks (CNNs)** for sequence and SMILES encoding
- **Bayesian Neural Networks (BNNs)** for uncertainty quantification
- **Ensemble Machine Learning** (Random Forest, Gradient Boosting, DTBoost)
- **Consensus predictions** with uncertainty estimates

## Architecture

```
PDBBind Data
    ↓
┌───────────────────────────────────────────────────────┐
│  Refined Set (Kd, Ki)    General Set (Kd + Ki)       │
└───────────────────────────────────────────────────────┘
    ↓                ↓                      ↓
Protein Seq      Ligand SMILES      Complex Descriptors
    ↓                ↓                      ↓
  CNN              CNN              RF / GB / DTBoost
    ↓                ↓                      ↓
    └────────────────┴──────────────────────┘
                     ↓
          Bayesian Fusion Layers
          (Variational Inference)
                     ↓
              Predictions ± Uncertainty
                     ↓
          Consensus (Weighted Average)
                     ↓
            Final pKd Prediction
```

### Key Features

1. **Bayesian Inference**
   - Weight uncertainty quantification
   - Epistemic uncertainty estimation
   - ELBO loss optimization
   - Monte Carlo dropout for predictions

2. **Multi-Modal Learning**
   - Protein sequence encoder (3-layer CNN)
   - Ligand SMILES encoder (3-layer CNN)
   - Complex interaction descriptors

3. **Ensemble Methods**
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - DTBoost (decision tree boosting variant)
   - Consensus via weighted averaging

4. **Uncertainty Quantification**
   - Bayesian posterior sampling
   - Standard deviation estimates
   - Confidence intervals for predictions

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install scikit-learn scipy
pip install pandas numpy
pip install matplotlib seaborn

# RDKit for molecular features
conda install -c conda-forge rdkit
```

## Quick Start

### 1. Test the Architecture

```bash
cd /home/aaryan0302/FOP-Code

# Test Bayesian model
python3 models/bayesian_affinity_predictor.py

# Test data preparation
python3 models/pdbbind_data_preparation.py
```

### 2. Prepare Data

Process BindingDB data into the required format:

```bash
python3 main/train_bayesian_affinity.py \
    --bindingdb_path data/bindingdb_data/BindingDB_All.tsv \
    --target_name "ACVR1" \
    --refined_output_dir data/pdbbind_refined
```

### 3. Train the Model

Full training pipeline:

```bash
python3 main/train_bayesian_affinity.py \
    --target_name "ACVR1" \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --kl_weight 0.01 \
    --checkpoint_dir models/bayesian_checkpoints
```

### 4. Quick Test (5 epochs)

```bash
python3 main/train_bayesian_affinity.py \
    --target_name "kinase" \
    --num_epochs 5 \
    --batch_size 16
```

## Model Architecture Details

### Bayesian Linear Layer

Each Bayesian linear layer maintains:
- **Weight mean** (μ_w): Learned weight parameters
- **Weight log-variance** (log σ²_w): Learned uncertainty
- **Bias mean** (μ_b): Learned bias parameters
- **Bias log-variance** (log σ²_b): Learned bias uncertainty

During training, weights are sampled: `w = μ + σ * ε` where `ε ~ N(0,1)`

### CNN Encoders

**Protein Sequence CNN:**
```
Embedding (vocab=25, dim=128)
    ↓
Conv1D (kernel=3) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Conv1D (kernel=5) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Conv1D (kernel=7) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Global MaxPool → FC(256)
```

**Ligand SMILES CNN:**
```
Embedding (vocab=70, dim=128)
    ↓
Conv1D (kernel=3) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Conv1D (kernel=5) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Conv1D (kernel=7) → BatchNorm → ReLU → Dropout → MaxPool
    ↓
Global MaxPool → FC(256)
```

### Bayesian Fusion Network

```
Input: [Protein(256) + Ligand(256) + Complex(128)] = 640
    ↓
BayesianLinear(640 → 512) → ReLU → Dropout
    ↓
BayesianLinear(512 → 256) → ReLU → Dropout
    ↓
BayesianLinear(256 → 128) → ReLU → Dropout
    ↓
BayesianLinear(128 → 1)
    ↓
Output: pKd prediction
```

### Loss Function (ELBO)

```
Loss = -log p(y|x,w) + β * KL(q(w)||p(w))

Where:
- First term: Negative log-likelihood (reconstruction loss)
- Second term: KL divergence (regularization)
- β: KL weight (controls posterior tightness)
```

## Data Format

### Input Files

**BindingDB TSV** should contain:
- `Ligand SMILES`: Molecular structure
- `BindingDB Target Chain Sequence 1`: Protein sequence
- `Kd (nM)`, `Ki (nM)`: Binding affinity measurements
- `Target Name`: Protein target name

### Processed Files

After data preparation, you'll have:

```
data/pdbbind_refined/
├── refined_protein_sequences.npy    # [N, 1000] int64
├── refined_ligand_smiles.npy        # [N, 200] int64
├── refined_complex_descriptors.npy  # [N, 200] float32
├── refined_affinities.npy           # [N] float32 (pKd values)
├── refined_metadata.csv             # Human-readable metadata
└── refined_stats.json               # Dataset statistics
```

### Data Splits

- **Refined Set**: High-quality Kd/Ki measurements
- **General Set**: All affinity measurements (optional, larger)

Default splits:
- Train: 70%
- Validation: 15%
- Test: 15%

## Training

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.001 | Adam learning rate |
| `kl_weight` | 0.01 | KL divergence weight |
| `dropout` | 0.3 | Dropout rate |
| `prior_sigma` | 1.0 | Bayesian prior std |
| `fusion_dims` | [512, 256, 128] | Fusion layer sizes |

### Training Procedure

1. **Data preparation**: Encode sequences and SMILES
2. **HNN training**: Train Bayesian neural network
3. **Ensemble training**: Train RF, GB, DTBoost on complex descriptors
4. **Consensus**: Combine predictions (60% HNN + 40% ensemble)
5. **Evaluation**: PCC, RMSE, MAE on test set

### Uncertainty Quantification

The model provides two types of uncertainty:

1. **Epistemic (Model) Uncertainty**
   - From Bayesian weight distributions
   - Quantified by Monte Carlo sampling (n=50-100)
   - `std = std(predictions over samples)`

2. **Predictive Uncertainty**
   - Combination of epistemic + aleatoric
   - Useful for active learning and confidence

## Evaluation Metrics

Following the paper's evaluation protocol:

1. **PCC (Pearson Correlation Coefficient)**
   ```
   PCC = cov(Y, Ŷ) / (σ_Y * σ_Ŷ)
   ```

2. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = √(Σ(y_i - ŷ_i)² / n)
   ```

3. **MAE (Mean Absolute Error)**
   ```
   MAE = Σ|y_i - ŷ_i| / n
   ```

## Expected Performance

Based on similar architectures on PDBBind:

| Dataset | PCC | RMSE | MAE |
|---------|-----|------|-----|
| Refined Set | 0.80-0.85 | 1.2-1.5 | 0.9-1.2 |
| General Set | 0.75-0.80 | 1.4-1.7 | 1.0-1.3 |

Performance depends heavily on:
- Dataset size and quality
- Target diversity
- Hyperparameter tuning

## Advanced Usage

### Custom Architecture

```python
from models.bayesian_affinity_predictor import HybridBayesianAffinityNetwork

model = HybridBayesianAffinityNetwork(
    protein_vocab_size=25,
    ligand_vocab_size=70,
    complex_descriptor_dim=200,
    fusion_hidden_dims=[1024, 512, 256, 128],  # Deeper network
    dropout=0.4,
    prior_sigma=0.5  # Tighter prior
)
```

### Prediction with Uncertainty

```python
# Single prediction
mean_pred, std_pred = model.predict_with_uncertainty(
    protein_seq, ligand_smiles, complex_desc,
    n_samples=100
)

print(f"Predicted pKd: {mean_pred:.2f} ± {std_pred:.2f}")
```

### Ensemble Predictions

```python
from models.bayesian_training_pipeline import BayesianAffinityTrainer

trainer = BayesianAffinityTrainer(model)
trainer.load_checkpoint('models/bayesian_checkpoints/best_model.pt')

# Get all predictions
results = trainer.get_consensus_prediction(
    protein_seq, ligand_smiles, complex_desc
)

print(f"HNN prediction: {results['hnn_mean']}")
print(f"Random Forest: {results['ml_rf']}")
print(f"Gradient Boosting: {results['ml_gb']}")
print(f"Final consensus: {results['final_consensus']}")
```

## Differences from Standard Neural Networks

| Aspect | Standard NN | Bayesian NN |
|--------|-------------|-------------|
| Weights | Fixed values | Probability distributions |
| Prediction | Single output | Mean ± std |
| Training | MSE loss | ELBO loss (NLL + KL) |
| Overfitting | Dropout only | Regularization via KL |
| Uncertainty | None | Epistemic uncertainty |

## Benefits of Bayesian Approach

1. **Uncertainty Quantification**: Know when the model is uncertain
2. **Better Generalization**: KL regularization prevents overfitting
3. **Active Learning**: Select most informative samples
4. **Robust Predictions**: Weight averaging over posterior
5. **Calibrated Confidence**: Uncertainty correlates with error

## Computational Considerations

- **Training time**: ~2-3x slower than standard NN (due to KL computation)
- **Memory**: Similar to standard NN
- **Inference**: ~10-100x slower (due to MC sampling for uncertainty)
- **GPU**: Highly recommended for CNN layers

## Troubleshooting

### High KL Divergence

If KL divergence is too high:
```bash
# Reduce KL weight
--kl_weight 0.001

# Or use KL annealing (start small, increase gradually)
```

### Uncertainty Too High/Low

Adjust prior:
```bash
# Tighter prior (lower uncertainty)
--prior_sigma 0.5

# Looser prior (higher uncertainty)
--prior_sigma 2.0
```

### Poor Convergence

Try:
```bash
# Lower learning rate
--learning_rate 0.0001

# Larger batch size
--batch_size 64

# More epochs
--num_epochs 200
```

## Citation

If you use this implementation, please cite the relevant papers:

```bibtex
@article{bayesian_affinity_prediction,
  title={Bayesian Neural Networks for Protein-Ligand Binding Affinity Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]}
}
```

## References

1. **Bayesian Deep Learning**: Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015
2. **Protein-Ligand Prediction**: Stepniewska-Dziubinska et al., "Development and evaluation of a deep learning model for protein–ligand binding affinity prediction", Bioinformatics 2018
3. **PDBBind Database**: Wang et al., "The PDBbind database: collection of binding affinities for protein-ligand complexes", J. Med. Chem. 2004

## License

[Your license information]

## Contact

For questions or issues, please contact [your contact information]

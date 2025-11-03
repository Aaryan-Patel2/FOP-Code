# Binding Affinity Prediction System - Summary

## What Was Created

I've built a complete binding affinity prediction pipeline for your FOP drug discovery project. Here's what you now have:

### 📁 Core Components

1. **`models/data_preparation.py`** (400+ lines)
   - Extracts and processes BindingDB data
   - Converts affinity metrics (Ki, IC50, Kd, EC50) to free energies
   - Computes 20+ molecular descriptors (2D and 3D)
   - Generates Morgan fingerprints
   - Handles your candidate ligands

2. **`models/affinity_predictor.py`** (450+ lines)
   - Deep neural network architecture
   - Separate encoders for descriptors and fingerprints
   - Training, validation, and prediction functionality
   - Model checkpointing and evaluation metrics
   - Scalable to large datasets

3. **`main/train_affinity_predictor.py`** (300+ lines)
   - Complete end-to-end training pipeline
   - Automatic data preparation and splitting
   - Model training with progress tracking
   - Evaluation and visualization
   - Predictions on your 26 candidate ligands

4. **`main/test_pipeline.py`** (100+ lines)
   - Quick validation script
   - Tests all components before full training
   - Checks dependencies and data availability

5. **`docs/AFFINITY_PREDICTION_README.md`**
   - Comprehensive documentation
   - Usage examples
   - Troubleshooting guide

## 🚀 Quick Start (3 Steps)

### Step 1: Test the Pipeline
```bash
cd /home/aaryan0302/FOP-Code
python3 main/test_pipeline.py
```

This will:
- Process your 26 candidate ligands
- Check if BindingDB data is available
- Verify all dependencies

### Step 2: Train the Model
```bash
python3 main/train_affinity_predictor.py
```

This will:
- Extract ACVR1 binding data from BindingDB
- Train a deep learning model (100 epochs, ~10-30 min depending on data size)
- Evaluate on test set
- **Predict affinities for your 26 candidate ligands**

### Step 3: View Results
```bash
# See predicted affinities
cat data/processed_ligands/predicted_affinities.csv

# View training curves
# Open: models/checkpoints/training_curves.png

# View prediction quality
# Open: models/checkpoints/test_predictions.png
```

## 📊 What the Model Predicts

For each of your 26 ligands, you'll get:

1. **`predicted_delta_g_kcal_mol`**: Free energy of binding (more negative = stronger binding)
   - Example: -9.5 kcal/mol (strong binder), -6.0 kcal/mol (weak binder)

2. **`predicted_Kd_nM`**: Dissociation constant in nanomolar
   - Example: 100 nM (good affinity), 1000 nM (moderate affinity)

3. **`predicted_pKd`**: -log10(Kd in M), standard metric in medicinal chemistry
   - Example: 7.0 (good), 6.0 (moderate), 5.0 (weak)

## 🎯 Integration with Your Workflow

### Current State
You have:
- ✅ 26 candidate ligands (SMILES)
- ✅ Molecular docking scores
- ✅ Scoring functions for FOP specificity

### New Capability
Now you can add:
- ✅ **Predicted binding affinity** (Kd, ΔG)
- ✅ ML-based assessment complementing docking
- ✅ Feature-rich molecular representations

### Combined Scoring
Update your `models/utils/scoring.py` to incorporate ML predictions:

```python
def combined_score(mutant_docking, wt_docking, predicted_affinity):
    """
    Combine docking scores with ML-predicted affinity
    """
    # Docking-based specificity
    specificity = wt_docking - mutant_docking
    
    # ML-predicted affinity (converted to approximate docking score scale)
    # ΔG to approximate docking score: multiply by ~0.7
    affinity_score = predicted_affinity * 0.7
    
    # Weighted combination
    total_score = 0.4 * specificity + 0.6 * affinity_score
    
    return total_score
```

## 📈 Expected Workflow

```
1. Generate/identify candidate molecules
   ↓
2. Prepare 3D structures (prepare_ligands.py)
   ↓
3. Perform molecular docking (run_docking.py)
   ↓
4. Predict binding affinity (NEW: affinity_predictor)
   ↓
5. Combined scoring and ranking
   ↓
6. Select top candidates for synthesis/testing
```

## 🔧 Customization Options

### Train on Different Target
```bash
python3 main/train_affinity_predictor.py --target_name "ALK2"
```

### Train on More Data (General Kinases)
```bash
python3 main/train_affinity_predictor.py --target_name "kinase" --min_affinity 1 --max_affinity 100000
```

### Adjust Model Complexity
```bash
# Larger model
python3 main/train_affinity_predictor.py --hidden_dims 1024 512 256 128 64

# Smaller model (faster, less data)
python3 main/train_affinity_predictor.py --hidden_dims 256 128 64
```

### Quick Test Run (10 epochs)
```bash
python3 main/train_affinity_predictor.py --num_epochs 10 --max_affinity 1000
```

## 📦 Output Structure

After training, you'll have:

```
FOP-Code/
├── data/
│   ├── processed_affinity/           # BindingDB training data
│   │   ├── affinity_dataset.csv
│   │   ├── morgan_fingerprints.npy
│   │   └── metadata.json
│   └── processed_ligands/            # Your candidate ligands
│       ├── ligand_features.csv
│       ├── ligand_fingerprints.npy
│       └── predicted_affinities.csv  ⭐ YOUR PREDICTIONS
│
├── models/
│   ├── checkpoints/
│   │   ├── best_model.pt             # Trained model
│   │   ├── training_curves.png       # Loss plots
│   │   └── test_predictions.png      # Quality assessment
│   ├── data_preparation.py
│   └── affinity_predictor.py
│
└── main/
    ├── train_affinity_predictor.py
    └── test_pipeline.py
```

## 🎓 Understanding the Predictions

### What is ΔG?
- Free energy of binding (thermodynamic quantity)
- **More negative = stronger binding**
- Typical range: -5 to -12 kcal/mol for drug-like molecules
- Related to Kd by: ΔG = RT ln(Kd)

### What is Kd?
- Dissociation constant (equilibrium quantity)
- **Lower = stronger binding**
- Drug targets typically: 1 nM - 10 µM
- Your FOP goal: ~1 µM (moderate, temporary binding)

### What is pKd?
- -log10(Kd in M)
- **Higher = stronger binding**
- Easy to interpret: each unit = 10× change in affinity
- Drug-like molecules: pKd 5-9

## 🔬 Next Steps for Property-Guided Generation

Now that you have an affinity predictor, you can:

### 1. **Screening Filter**
Use the model to quickly screen large virtual libraries before expensive docking

### 2. **Active Learning**
- Synthesize/test molecules with high uncertainty
- Add new data to retrain model
- Iteratively improve predictions

### 3. **Generative Model Guidance** (Future)
Integrate with Graphormer or other generative models:
```python
def guided_generation(molecule):
    # Generate candidate
    candidate = generator.generate()
    
    # Score with affinity predictor
    affinity_score = affinity_predictor.predict(candidate)
    
    # Use as reward signal
    reward = calculate_reward(affinity_score, specificity, ...)
    
    # Update generator
    generator.update(reward)
```

### 4. **Multi-Objective Optimization**
Optimize for:
- Binding affinity (your new model)
- Specificity (mutant vs WT)
- Kinetics (residence time)
- Drug-likeness (QED, SA score)
- ADME properties

## ⚠️ Important Notes

1. **Data Availability**: If BindingDB has limited ACVR1 data, consider:
   - Training on broader kinase family
   - Transfer learning from general protein-ligand model
   - Using docking scores as pseudo-labels

2. **Model Uncertainty**: Deep learning predictions should be:
   - Combined with physics-based methods (docking)
   - Validated experimentally
   - Used as one of multiple filters

3. **Applicability Domain**: Model is most reliable for molecules:
   - Similar to training set
   - Within chemical space of drug-like molecules
   - Not too different from known ACVR1 binders

## 📚 Theoretical Background

### Why This Approach Works

1. **Rich Feature Representation**
   - 2D descriptors capture chemical properties
   - 3D descriptors capture shape
   - Fingerprints capture substructure patterns

2. **Deep Learning Architecture**
   - Learns complex non-linear relationships
   - Fuses multiple information sources
   - Generalizes to unseen molecules

3. **Thermodynamic Grounding**
   - Predicts ΔG (fundamental quantity)
   - Converts to Kd (measurable quantity)
   - Temperature-aware predictions

### Comparison to Docking

| Aspect | Docking | ML Affinity Prediction |
|--------|---------|------------------------|
| Speed | Slow (~min/molecule) | Fast (~ms/molecule) |
| Physics | Explicit | Learned |
| Accuracy | Moderate | Moderate-High (with data) |
| Interpretability | High | Low |
| Data Required | Protein structure | Training examples |

**Best Practice**: Use both! ML for screening, docking for final validation.

## 🆘 Troubleshooting

### "Not enough ACVR1 data"
```bash
# Use broader target
python3 main/train_affinity_predictor.py --target_name "kinase"
```

### "Out of memory"
```bash
# Reduce model size
python3 main/train_affinity_predictor.py --hidden_dims 256 128 --batch_size 16
```

### "Poor predictions"
- Check if training R² > 0.6 (if not, need more/better data)
- Ensure candidate ligands are similar to training set
- Consider ensemble of multiple models

## 💡 Pro Tips

1. **Start Small**: Test with `--num_epochs 10` first
2. **Check Data**: Run `test_pipeline.py` before full training
3. **Monitor Training**: Watch for overfitting (val loss increasing while train loss decreases)
4. **Ensemble**: Train multiple models with different seeds, average predictions
5. **Uncertainty**: Use prediction variance across ensemble as confidence measure

## 🎉 You're Ready!

Run the test, train the model, and you'll have ML-predicted binding affinities for all your candidate ligands!

```bash
# Quick test (< 1 min)
python3 main/test_pipeline.py

# Full training (10-30 min)
python3 main/train_affinity_predictor.py

# Check results
head -20 data/processed_ligands/predicted_affinities.csv
```

Good luck with your FOP drug discovery! 🚀

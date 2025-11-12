# Data Directory

This directory contains datasets and processed files for the Bayesian Affinity Predictor.

## 📦 Required Data

### BindingDB Dataset
Due to file size (6.3GB), the BindingDB_All.tsv file is **not included** in the repository.

**To download:**
```bash
# Option 1: Download from BindingDB
cd data/bindingdb_data
wget https://www.bindingdb.org/bind/downloads/BindingDB_All_202510_tsv.zip
unzip BindingDB_All_202510_tsv.zip

# Option 2: Download from our GitHub release
# (If you've uploaded it to GitHub releases)
wget https://github.com/Aaryan-Patel2/FOP-Code/releases/download/v1.0/BindingDB_All.tsv
mv BindingDB_All.tsv data/bindingdb_data/
```

**Or use the automatic downloader:**
```python
from models.data_preparation import download_bindingdb
download_bindingdb(output_dir='data/bindingdb_data')
```

## 📁 Directory Structure

```
data/
├── bindingdb_data/          # BindingDB dataset (download required)
│   └── BindingDB_All.tsv    # Main dataset (6.3GB - not in git)
│
├── structures/              # Example protein structures
│   └── *.pdb                # PDB files for ACVR1, etc.
│
├── converted_structures/    # PDBQT files for docking
│   └── *.pdbqt              # Converted structures
│
├── ligands_3d/              # 3D ligand structures
│   └── *.sdf                # Generated conformers
│
├── initial_SMILES/          # SMILES input files
│   └── SMILES_strings.txt   # Ligand SMILES strings
│
├── converted_ligands/       # PDBQT ligands for docking
│   └── ligand_*.pdbqt       # Converted ligands
│
└── processed_ligands_test/  # Pre-processed features
    ├── ligand_features.csv
    └── ligand_fingerprints.npy
```

## 🔄 Data Processing Pipeline

1. **Download BindingDB** (see instructions above)
2. **Filter by target**: Extract specific protein targets (e.g., ACVR1)
3. **Process sequences**: Encode proteins and SMILES
4. **Calculate descriptors**: Generate molecular features
5. **Train model**: Use processed data for training

See `QUICKSTART.md` for training examples.

## 📊 Example Data

The repository includes **example data** for quick testing:
- 21 example ligands (PDBQT format)
- 3 ACVR1 structures (wild-type, mutant, inhibitor complex)
- Pre-processed features for testing

## 💾 Data Size Requirements

- **Full BindingDB**: ~6.3GB (2.7M binding data points)
- **Filtered target** (e.g., ACVR1): ~10-100MB (depends on target)
- **Processed features**: ~100-500MB (depends on dataset size)
- **Model checkpoints**: ~50MB per model

## 🚀 Quick Start Without Full Dataset

You can train on a **small subset** for testing:

```python
from quick_start import AffinityPredictor

predictor = AffinityPredictor()

# Train on first 1000 rows only (for quick testing)
predictor.train(
    bindingdb_path='data/bindingdb_data/BindingDB_All.tsv',
    target_name='ACVR1',  # Will filter automatically
    num_epochs=10,
    batch_size=32
)
```

## 📝 Notes

- Add `data/bindingdb_data/BindingDB_All.tsv` to `.gitignore`
- Keep example data (structures, converted files) in repo
- Consider using Git LFS for files > 100MB
- Download full dataset only when needed for training

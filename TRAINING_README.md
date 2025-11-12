# Training Pipeline - Quick Start

Simple command-line tools to train the ACVR1 affinity predictor.

## 🚀 Quick Start

### 1. Train the model (one command!)

```bash
python train_model.py
```

This will:
- ✅ Check your environment
- ✅ Verify BindingDB data is available
- ✅ Train the Bayesian affinity predictor
- ✅ Export the model to `models/pretrained/`

**Training time:** ~30-60 minutes on CPU (faster with GPU)

### 2. Test predictions

```bash
python test_predictions.py
```

This will load your trained model and make a test prediction to verify it works.

## ⚙️ Training Options

Customize training with command-line arguments:

```bash
# CPU-friendly training (smaller batch, fewer epochs)
python train_model.py --epochs 10 --batch-size 16

# Full training (better accuracy)
python train_model.py --epochs 50 --batch-size 32

# Use different target protein
python train_model.py --target "EGFR"

# Custom learning rate
python train_model.py --lr 0.0001
```

See all options:
```bash
python train_model.py --help
```

## 📋 Prerequisites

1. **Install dependencies:**
   ```bash
   pip install torch rdkit numpy pandas scikit-learn scipy
   ```

2. **Download BindingDB data:**
   ```bash
   bash download_data.sh
   ```
   
   Or manually download BindingDB_All.tsv to `data/bindingdb_data/`

## 📁 Output Files

After training, you'll have:

- `trained_models/best_model.ckpt` - Your trained model
- `models/pretrained/affinity_predictor.ckpt` - Exported model for integration
- `models/pretrained/model_metadata.json` - Model information

## 🔗 Next Steps

Once trained, integrate the model with:

1. **GCDM Diffusion Model** - Guide molecule generation
2. **Virtual Screening** - Screen large compound libraries
3. **Optimization Pipelines** - Optimize lead compounds

## 💡 Tips

- **First time?** Start with `--epochs 10` to test quickly
- **Limited RAM?** Use `--batch-size 8` or `--batch-size 16`
- **Have GPU?** Training will automatically use it (much faster!)
- **Want better accuracy?** Increase `--epochs` to 50-100

## 🐛 Troubleshooting

**Kernel crashes / Out of memory:**
- Reduce batch size: `--batch-size 8`
- Close other applications
- Consider using a machine with more RAM

**Model not found:**
- Make sure `train_model.py` completed successfully
- Check `trained_models/best_model.ckpt` exists

**Data not found:**
- Run `bash download_data.sh`
- Or specify custom path: `--data-path /path/to/BindingDB_All.tsv`

## 📖 Documentation

- Full API docs: `QUICKSTART.md`
- Binding kinetics: `docs/BINDING_KINETICS_EXPLAINED.md`
- k_off implementation: `docs/KOFF_IMPLEMENTATION_SUMMARY.md`

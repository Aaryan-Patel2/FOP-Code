# Google Cloud TPU Training Setup

Complete guide for training the affinity predictor on Google Cloud TPU VMs.

## Prerequisites

1. Google Cloud account with TPU quotas
2. `gcloud` CLI installed locally
3. TPU credits activated

## Step 1: Create TPU VM Instance

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"  # TPU v2/v3 zones
export TPU_NAME="affinity-tpu"

# Create TPU VM (v2-8 or v3-8)
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0 \
  --project=$PROJECT_ID
```

**TPU Options:**
- `v2-8`: 8 cores, good for testing (~$4.5/hour)
- `v3-8`: 8 cores, faster training (~$8/hour)
- `v2-32`: 32 cores, production scale (~$18/hour)

## Step 2: SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID
```

## Step 3: Setup Environment on TPU VM

Once SSH'd in:

```bash
# Update system
sudo apt-get update
sudo apt-get install -y git wget

# Install Python dependencies
pip install --upgrade pip
pip install torch==2.0.0
pip install torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/tpu-pytorch-releases/torch_xla_wheels.html
pip install pytorch-lightning
pip install numpy pandas scipy scikit-learn
pip install rdkit-pypi  # RDKit for TPU VM
pip install matplotlib seaborn

# Clone your repository
git clone https://github.com/Aaryan-Patel2/FOP-Code.git
cd FOP-Code
```

## Step 4: Upload BindingDB Data

From your local machine:

```bash
# Upload BindingDB data to TPU VM
gcloud compute tpus tpu-vm scp \
  data/bindingdb_data/BindingDB_All.tsv \
  $TPU_NAME:~/FOP-Code/data/bindingdb_data/ \
  --zone=$ZONE \
  --project=$PROJECT_ID
```

## Step 5: Run TPU Training

On the TPU VM:

```bash
cd ~/FOP-Code

# Basic training
python train_tpu_cloud.py --target kinase --epochs 50 --batch-size 128

# Transfer learning (recommended for better results)
python train_transfer_learning.py \
  --target kinase \
  --pretrain-epochs 15 \
  --finetune-epochs 25 \
  --batch-size 128
```

**TPU-Optimized Settings:**
- Larger batch size: 128 or 256 (TPUs handle large batches efficiently)
- More epochs: TPUs train faster, so run longer
- Lower learning rate: 1e-4 to 5e-4 for stability

## Step 6: Monitor Training

```bash
# Check TPU utilization
watch -n 1 'python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"'

# Monitor training logs
tail -f training.log
```

## Step 7: Download Trained Model

From your local machine:

```bash
# Download trained model
gcloud compute tpus tpu-vm scp \
  $TPU_NAME:~/FOP-Code/trained_models/best_model.ckpt \
  ./trained_models/ \
  --zone=$ZONE \
  --project=$PROJECT_ID
```

## Step 8: Cleanup (Important!)

```bash
# Delete TPU VM to stop charges
gcloud compute tpus tpu-vm delete $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID
```

## Cost Optimization Tips

1. **Use Preemptible TPUs**: Add `--preemptible` flag (60-90% cheaper)
2. **Start/Stop**: Stop TPU when not training
3. **Region Selection**: Use `us-central1` (cheapest)
4. **Monitor Usage**: Set budget alerts in Cloud Console

## Troubleshooting

### TPU Not Detected
```bash
# Check TPU status
gcloud compute tpus tpu-vm list --zone=$ZONE

# Verify PyTorch XLA
python3 -c "import torch_xla; print(torch_xla.__version__)"
```

### Out of Memory
- Reduce batch size: `--batch-size 64`
- Reduce model size in config

### Slow Training
- Ensure using TPU: Check logs for "TPU available: True"
- Increase batch size to utilize TPU better
- Use `torch_xla.core.xla_model.mark_step()` for graph compilation

## Quick Start Script

Save this as `setup_tpu.sh`:

```bash
#!/bin/bash
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export TPU_NAME="affinity-tpu"

# Create TPU
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0 \
  --project=$PROJECT_ID

# SSH and setup
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --command="
    pip install torch==2.0.0 torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/tpu-pytorch-releases/torch_xla_wheels.html &&
    pip install pytorch-lightning numpy pandas scipy scikit-learn rdkit-pypi &&
    git clone https://github.com/Aaryan-Patel2/FOP-Code.git
  "
```

## Expected Training Times (v3-8 TPU)

- **25 epochs, 837 samples**: ~10-15 minutes
- **50 epochs, 837 samples**: ~20-30 minutes
- **Transfer learning (10+15 epochs, 14K samples)**: ~45-60 minutes

## References

- [Google Cloud TPU Docs](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm)
- [PyTorch XLA](https://github.com/pytorch/xla)
- [TPU Pricing](https://cloud.google.com/tpu/pricing)

# Cloud TPU Quick Reference

## 🚀 Quick Start (One Command)

```bash
# Edit these first:
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"

# Run setup script
./setup_cloud_tpu.sh
```

## 📋 Manual Steps

### 1. Create TPU VM
```bash
gcloud compute tpus tpu-vm create affinity-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0
```

### 2. SSH into TPU
```bash
gcloud compute tpus tpu-vm ssh affinity-tpu --zone=us-central1-a
```

### 3. Train Model (on TPU VM)
```bash
cd FOP-Code

# Standard training
python train_tpu_cloud.py \
  --target kinase \
  --epochs 50 \
  --batch-size 128

# Transfer learning (recommended)
python train_transfer_learning.py \
  --target kinase \
  --pretrain-epochs 15 \
  --finetune-epochs 25 \
  --batch-size 128
```

### 4. Download Model (from local machine)
```bash
gcloud compute tpus tpu-vm scp \
  affinity-tpu:~/FOP-Code/trained_models/best_model.ckpt \
  ./trained_models/ \
  --zone=us-central1-a
```

### 5. Delete TPU (IMPORTANT!)
```bash
gcloud compute tpus tpu-vm delete affinity-tpu --zone=us-central1-a
```

## 💰 Cost Optimization

| TPU Type | Cores | Speed | Cost/hour | Best For |
|----------|-------|-------|-----------|----------|
| v2-8 | 8 | 1x | $4.50 | Testing |
| v3-8 | 8 | 2x | $8.00 | **Recommended** |
| v2-32 | 32 | 4x | $18.00 | Large scale |

**Cost Saving Tips:**
- Use preemptible: Add `--preemptible` flag (60% cheaper)
- Delete immediately after training
- Use budget alerts
- Train during off-peak hours

## 🎯 Recommended Settings for Your Data

### Kinase-only (837 samples)
```bash
python train_tpu_cloud.py \
  --target kinase \
  --epochs 50 \
  --batch-size 128 \
  --lr 5e-4
```
**Expected time:** 15-20 minutes on v3-8  
**Expected cost:** ~$2-3

### Transfer Learning (14,290 pre-train + 837 fine-tune)
```bash
python train_transfer_learning.py \
  --target kinase \
  --pretrain-epochs 15 \
  --finetune-epochs 25 \
  --batch-size 128 \
  --pretrain-lr 5e-4 \
  --finetune-lr 1e-4
```
**Expected time:** 45-60 minutes on v3-8  
**Expected cost:** ~$6-8  
**Better results** ✅

## 🔧 Troubleshooting

### TPU not detected
```bash
# Check TPU status
gcloud compute tpus tpu-vm list --zone=us-central1-a

# Test TPU from Python
python3 -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

### Out of memory
- Reduce batch size: `--batch-size 64`
- Use mixed precision training

### Slow training
- Increase batch size to 128 or 256
- Check TPU utilization: should be >80%

## 📊 Expected Performance

With transfer learning on TPU:
- **RMSE**: ~0.3-0.4 (normalized)
- **PCC**: >0.7 (target)
- **Training speed**: ~3-4 minutes/epoch on v3-8

## 🔗 Useful Commands

```bash
# List all TPUs
gcloud compute tpus tpu-vm list --zone=us-central1-a

# Check TPU details
gcloud compute tpus tpu-vm describe affinity-tpu --zone=us-central1-a

# Stop (but keep) TPU
gcloud compute tpus tpu-vm stop affinity-tpu --zone=us-central1-a

# Start stopped TPU
gcloud compute tpus tpu-vm start affinity-tpu --zone=us-central1-a

# Monitor costs
gcloud billing accounts list
```

## 📚 Resources

- [Cloud TPU Documentation](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm)
- [PyTorch XLA Guide](https://github.com/pytorch/xla)
- [TPU Pricing](https://cloud.google.com/tpu/pricing)
- [Your Project](https://github.com/Aaryan-Patel2/FOP-Code)

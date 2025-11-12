# Cloud TPU Workflow (ARM64 → Cloud TPU)

## 🎯 The Problem
- Your local machine: **ARM64 (aarch64)** 
- torch_xla requirement: **x86_64 only**
- ❌ Cannot install torch_xla locally

## ✅ The Solution
**Don't run training locally!** Use Cloud TPU VMs (x86_64) instead.

## 📋 Complete Workflow

### Phase 1: Local Machine (ARM64) - NO TPU REQUIRED
You only need `gcloud` CLI installed. No torch_xla needed!

```bash
# 1. Prepare your code locally (already done)
cd /home/aaryan0302/FOP-Code
git add .
git commit -m "Ready for TPU training"
git push

# 2. Set your GCP credentials
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"
export TPU_NAME="affinity-tpu"

# 3. Create Cloud TPU VM (x86_64 architecture)
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0 \
  --project=$PROJECT_ID

# This creates a remote x86_64 VM with TPU attached
# torch_xla WILL work there!
```

### Phase 2: Cloud TPU VM (x86_64) - HAS TPU
Now SSH into the Cloud VM and train there:

```bash
# 4. SSH from your ARM64 machine into the x86_64 TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE

# You're now on an x86_64 machine! 🎉

# 5. Install dependencies (x86_64 wheels available!)
pip install torch==2.0.0
pip install torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/tpu-pytorch-releases/torch_xla_wheels.html
pip install pytorch-lightning numpy pandas scipy scikit-learn rdkit-pypi

# 6. Clone your code
git clone https://github.com/Aaryan-Patel2/FOP-Code.git
cd FOP-Code

# 7. Upload data (from another terminal on your local machine)
# Keep SSH session open, open new terminal locally:
gcloud compute tpus tpu-vm scp \
  data/bindingdb_data/BindingDB_All.tsv \
  $TPU_NAME:~/FOP-Code/data/bindingdb_data/ \
  --zone=$ZONE

# 8. Back in SSH session, start training!
python train_tpu_cloud.py \
  --target kinase \
  --epochs 50 \
  --batch-size 128

# Or use transfer learning:
python train_transfer_learning.py \
  --target kinase \
  --pretrain-epochs 15 \
  --finetune-epochs 25 \
  --batch-size 128
```

### Phase 3: Download Results
Back on your local ARM64 machine:

```bash
# 9. Download trained model
gcloud compute tpus tpu-vm scp \
  $TPU_NAME:~/FOP-Code/trained_models/best_model.ckpt \
  ./trained_models/ \
  --zone=$ZONE

# 10. IMPORTANT: Delete TPU to stop charges!
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
```

## 🔄 Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│  Your ARM64 Laptop (aarch64)                            │
│  ├─ NO torch_xla needed                                 │
│  ├─ Just gcloud CLI                                     │
│  └─ Uploads code/data, downloads results                │
└─────────────────────────────────────────────────────────┘
                        │
                        │ SSH / gcloud commands
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Cloud TPU VM (x86_64)                                  │
│  ├─ torch_xla WORKS here! ✅                            │
│  ├─ Has TPU attached                                    │
│  ├─ Runs training                                       │
│  └─ Saves checkpoints                                   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (One Command)

I've created an automated setup script that does everything:

```bash
# Edit these variables first:
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"

# Run automated setup
./setup_cloud_tpu.sh

# Then just SSH in and train:
gcloud compute tpus tpu-vm ssh affinity-tpu --zone=$ZONE
cd FOP-Code
python train_tpu_cloud.py --target kinase --epochs 50 --batch-size 128
```

## ⚠️ Important Notes

1. **DON'T run `train_tpu_cloud.py` locally** - it will fail with torch_xla import error
2. **DO run it on the Cloud TPU VM** - torch_xla is pre-installed there
3. **Your ARM64 machine is just a terminal** - all training happens in the cloud
4. **Remember to delete the TPU** - or you'll keep getting charged!

## 💰 Cost Example

- Create TPU: Free (just VM creation)
- Training time: ~60 minutes for transfer learning
- TPU v3-8 cost: $8/hour
- **Total cost**: ~$8 for full training run

## 🐛 If You See Errors Locally

```bash
# This is NORMAL on your ARM64 machine:
$ python train_tpu_cloud.py
ImportError: No module named 'torch_xla'

# Solution: Don't run it locally!
# Upload to Cloud TPU VM and run there instead.
```

## ✅ Verification

To verify everything is set up correctly:

```bash
# On your local ARM64 machine (should work):
gcloud compute tpus tpu-vm list --zone=us-central1-a

# On Cloud TPU VM (after SSH):
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
# Should print: xla:0
```

## 📚 Files You Need

- ✅ `setup_cloud_tpu.sh` - Automated setup (run locally)
- ✅ `train_tpu_cloud.py` - Training script (run on TPU VM)
- ✅ `train_transfer_learning.py` - Transfer learning (run on TPU VM)
- ✅ This guide!

Everything is ready - you just need to run `setup_cloud_tpu.sh` from your ARM64 machine! 🚀

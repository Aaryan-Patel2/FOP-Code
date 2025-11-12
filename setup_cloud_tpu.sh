#!/bin/bash
# Quick setup script for Google Cloud TPU training
# Run this from your LOCAL machine

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-your-project-id}"
ZONE="${ZONE:-us-central1-a}"
TPU_NAME="${TPU_NAME:-affinity-tpu}"
TPU_TYPE="${TPU_TYPE:-v3-8}"

echo "========================================"
echo "Google Cloud TPU Setup for FOP-Code"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  TPU Name: $TPU_NAME"
echo "  TPU Type: $TPU_TYPE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✓ gcloud CLI found"

# Set project
echo ""
echo "Setting project..."
gcloud config set project $PROJECT_ID

# Create TPU VM
echo ""
echo "Creating TPU VM ($TPU_TYPE)..."
echo "This will take 2-3 minutes..."
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$TPU_TYPE \
  --version=tpu-vm-pt-2.0 \
  --project=$PROJECT_ID

echo ""
echo "✓ TPU VM created successfully!"

# Wait for TPU to be ready
echo ""
echo "Waiting for TPU to be ready..."
sleep 30

# Install dependencies
echo ""
echo "Installing dependencies on TPU VM..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --command="
    echo '=== Installing Python packages ===' &&
    pip install --upgrade pip &&
    pip install torch==2.0.0 &&
    pip install torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/tpu-pytorch-releases/torch_xla_wheels.html &&
    pip install pytorch-lightning &&
    pip install numpy pandas scipy scikit-learn &&
    pip install rdkit-pypi &&
    pip install matplotlib seaborn &&
    echo '✓ Packages installed'
  "

# Clone repository
echo ""
echo "Cloning FOP-Code repository..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --command="
    rm -rf FOP-Code &&
    git clone https://github.com/Aaryan-Patel2/FOP-Code.git &&
    echo '✓ Repository cloned'
  "

# Create data directory
echo ""
echo "Creating data directory..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --command="
    mkdir -p FOP-Code/data/bindingdb_data &&
    echo '✓ Data directory created'
  "

# Upload BindingDB data
echo ""
echo "Uploading BindingDB data..."
echo "This may take 5-10 minutes for a 6GB file..."
if [ -f "data/bindingdb_data/BindingDB_All.tsv" ]; then
    gcloud compute tpus tpu-vm scp \
      data/bindingdb_data/BindingDB_All.tsv \
      $TPU_NAME:~/FOP-Code/data/bindingdb_data/ \
      --zone=$ZONE \
      --project=$PROJECT_ID
    echo "✓ Data uploaded"
else
    echo "⚠ BindingDB_All.tsv not found locally at data/bindingdb_data/"
    echo "  Upload manually after setup with:"
    echo "  gcloud compute tpus tpu-vm scp data/bindingdb_data/BindingDB_All.tsv $TPU_NAME:~/FOP-Code/data/bindingdb_data/ --zone=$ZONE"
fi

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  1. SSH into TPU VM:"
echo "     gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
echo ""
echo "  2. Run training:"
echo "     cd FOP-Code"
echo "     python train_tpu_cloud.py --target kinase --epochs 50 --batch-size 128"
echo ""
echo "  3. Download trained model (from local machine):"
echo "     gcloud compute tpus tpu-vm scp $TPU_NAME:~/FOP-Code/trained_models/best_model.ckpt ./trained_models/ --zone=$ZONE"
echo ""
echo "  4. Delete TPU when done (IMPORTANT - to stop charges):"
echo "     gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE"
echo ""
echo "Cost: ~\$8/hour for v3-8 TPU"
echo ""

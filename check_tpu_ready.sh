#!/bin/bash
# Test if your local ARM64 machine is ready for Cloud TPU deployment
# Run this BEFORE creating expensive TPU resources

echo "========================================"
echo "Cloud TPU Deployment Check (ARM64)"
echo "========================================"
echo ""

# Check architecture
ARCH=$(uname -m)
echo "✓ Architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "  (Note: Not ARM64, but that's OK)"
fi
echo ""

# Check gcloud
echo "Checking gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found!"
    echo "   Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "✓ gcloud CLI installed"
GCLOUD_VERSION=$(gcloud version --format="value(version)")
echo "  Version: $GCLOUD_VERSION"
echo ""

# Check if authenticated
echo "Checking authentication..."
ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null)
if [ -z "$ACCOUNT" ]; then
    echo "❌ Not authenticated with gcloud"
    echo "   Run: gcloud auth login"
    exit 1
fi
echo "✓ Authenticated as: $ACCOUNT"
echo ""

# Check project
echo "Checking project..."
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "⚠  No default project set"
    echo "   Set with: gcloud config set project YOUR_PROJECT_ID"
else
    echo "✓ Default project: $PROJECT"
fi
echo ""

# Check if BindingDB data exists
echo "Checking data files..."
if [ -f "data/bindingdb_data/BindingDB_All.tsv" ]; then
    SIZE=$(du -h data/bindingdb_data/BindingDB_All.tsv | cut -f1)
    echo "✓ BindingDB data found ($SIZE)"
else
    echo "⚠  BindingDB data not found at: data/bindingdb_data/BindingDB_All.tsv"
    echo "   You'll need to upload this to the TPU VM"
fi
echo ""

# Check git repository
echo "Checking git repository..."
if [ -d ".git" ]; then
    REMOTE=$(git remote get-url origin 2>/dev/null)
    if [ -z "$REMOTE" ]; then
        echo "⚠  Git repository exists but no remote set"
    else
        echo "✓ Git repository: $REMOTE"
    fi
else
    echo "⚠  Not a git repository"
    echo "   Initialize with: git init && git remote add origin YOUR_REPO"
fi
echo ""

# Check TPU quota
echo "Checking TPU availability in zones..."
if [ ! -z "$PROJECT" ]; then
    # Try common TPU zones
    for ZONE in us-central1-a us-central1-b us-central1-c europe-west4-a; do
        AVAILABLE=$(gcloud compute tpus accelerator-types list --zone=$ZONE --project=$PROJECT 2>/dev/null | grep -c "v3-8")
        if [ "$AVAILABLE" -gt 0 ]; then
            echo "  ✓ $ZONE: TPUs available"
            FOUND_ZONE=1
            break
        fi
    done
    
    if [ -z "$FOUND_ZONE" ]; then
        echo "  ⚠  Could not verify TPU availability"
        echo "     Check quota in Cloud Console: https://console.cloud.google.com/iam-admin/quotas"
    fi
else
    echo "  ⚠  Skipping (no project set)"
fi
echo ""

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo ""

if command -v gcloud &> /dev/null && [ ! -z "$ACCOUNT" ]; then
    echo "✅ Your local machine is ready for Cloud TPU deployment!"
    echo ""
    echo "Next steps:"
    echo "  1. Set project (if not set):"
    echo "     export PROJECT_ID=\"your-project-id\""
    echo ""
    echo "  2. Run setup:"
    echo "     ./setup_cloud_tpu.sh"
    echo ""
    echo "  3. Or create TPU manually:"
    echo "     gcloud compute tpus tpu-vm create affinity-tpu \\"
    echo "       --zone=us-central1-a \\"
    echo "       --accelerator-type=v3-8 \\"
    echo "       --version=tpu-vm-pt-2.0"
    echo ""
    echo "Remember: Training happens ON the TPU VM, not locally!"
    echo "Your ARM64 machine just uploads code and downloads results."
else
    echo "❌ Setup incomplete. Please:"
    if ! command -v gcloud &> /dev/null; then
        echo "  - Install gcloud CLI"
    fi
    if [ -z "$ACCOUNT" ]; then
        echo "  - Run: gcloud auth login"
    fi
fi
echo ""

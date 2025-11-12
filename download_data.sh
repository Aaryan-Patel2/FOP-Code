#!/bin/bash
# Data download script for Google Colab or local use

echo "=========================================="
echo "Downloading BindingDB Dataset"
echo "=========================================="
echo ""

# Create data directory if it doesn't exist
mkdir -p data/bindingdb_data
cd data/bindingdb_data

# Check if file already exists
if [ -f "BindingDB_All.tsv" ]; then
    echo "✓ BindingDB_All.tsv already exists"
    echo "  Size: $(du -h BindingDB_All.tsv | cut -f1)"
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

echo "[1/3] Downloading BindingDB dataset..."
echo "  This is a large file (~500MB compressed, 6.3GB uncompressed)"
echo "  Download may take several minutes..."
echo ""

# Try to download from BindingDB
BINDINGDB_URL="https://www.bindingdb.org/bind/downloads/BindingDB_All_202510_tsv.zip"

if command -v wget &> /dev/null; then
    wget -c "$BINDINGDB_URL" -O BindingDB_All_202510_tsv.zip
elif command -v curl &> /dev/null; then
    curl -L -C - "$BINDINGDB_URL" -o BindingDB_All_202510_tsv.zip
else
    echo "❌ Error: Neither wget nor curl is available"
    echo "Please install wget or curl and try again"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ Download failed. Please try again or download manually from:"
    echo "   https://www.bindingdb.org/bind/downloads/"
    exit 1
fi

echo ""
echo "[2/3] Extracting dataset..."
unzip -o BindingDB_All_202510_tsv.zip

if [ $? -ne 0 ]; then
    echo "❌ Extraction failed"
    exit 1
fi

echo ""
echo "[3/3] Cleaning up..."
# Optionally remove zip file to save space
read -p "Remove zip file to save space? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    rm BindingDB_All_202510_tsv.zip
    echo "✓ Zip file removed"
fi

echo ""
echo "=========================================="
echo "✓ Download complete!"
echo "=========================================="
echo ""
echo "Dataset location: data/bindingdb_data/BindingDB_All.tsv"
echo "Dataset size: $(du -h BindingDB_All.tsv | cut -f1)"
echo ""
echo "You can now train the model:"
echo "  python3 quick_start.py"
echo "  # or"
echo "  from quick_start import AffinityPredictor"
echo "  predictor = AffinityPredictor()"
echo "  predictor.train(bindingdb_path='data/bindingdb_data/BindingDB_All.tsv')"

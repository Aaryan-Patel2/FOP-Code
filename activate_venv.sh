#!/bin/bash
# Activation script for FOP-Code virtual environment with lzma support

echo "Activating FOP-Code virtual environment..."
source /home/aaryan0302/FOP-Code/venv_new/bin/activate

echo "✓ Virtual environment activated!"
echo ""
echo "Python version: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To verify lzma is available:"
echo "  python -c 'import lzma; print(\"✓ lzma available\")'"
echo ""
echo "To train your model:"
echo "  python train_model.py"
echo ""
echo "To deactivate:"
echo "  deactivate"

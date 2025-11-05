#!/bin/bash
# Quick setup script for UDIS-D evaluation

echo "Setting up UDIS-D evaluation pipeline..."

# Create directory structure
mkdir -p data/raw/dataset/UDIS-D
mkdir -p results/udis_d

# Run download script
python src/download_udis_dataset.py

echo ""
echo "Next steps:"
echo "1. If dataset not found, download manually from:"
echo "   https://github.com/nie-lang/UnsupervisedDeepImageStitching"
echo "2. Extract to: data/raw/dataset/UDIS-D/"
echo "3. Run evaluation: python src/test_udis_batch.py"
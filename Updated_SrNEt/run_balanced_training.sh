#!/bin/bash

# Script to analyze dataset balance and run training with balanced datasets

echo "=== SteganoGan Balanced Training Script ==="

# Create necessary directories
mkdir -p plots

# Step 1: Analyze dataset distribution before balancing
echo "Step 1: Analyzing dataset distribution..."
python3 analyze_dataset.py

# Step 2: Run training with balanced dataset
echo "Step 2: Running training with balanced dataset..."
python3 train.py

# Step 3: Test model with balanced evaluation
echo "Step 3: Testing model with balanced evaluation..."
python3 test.py

echo "All done! Check plots/ directory for dataset distribution and training results."

#!/bin/bash
# Launcher script for smart hypercube batch experiments

echo "Activating conda environment..."
source ~/miniconda3/bin/activate
conda activate dfs-analyzer

echo "Starting smart batch experiments (10D-20D)..."
echo "Estimated time: ~7 hours"
echo ""

python run_hypercube_smart_batch.py

#!/bin/bash
# Launcher script for variable sample hypercube batch experiments (11D-20D)
# Estimated time: ~5.8 hours

echo "==========================================================================="
echo "VARIABLE SAMPLE BATCH: 11D-20D HYPERCUBE EXPERIMENTS"
echo "==========================================================================="
echo "Start time: $(date)"
echo ""
echo "Sample strategy:"
echo "  11D-15D: 10,000 samples (~35 minutes)"
echo "  16D-18D: 10,000 samples (~4.1 hours)"
echo "  19D-20D: 1,000 samples (~1.4 hours)"
echo ""
echo "Total estimated time: ~5.8 hours"
echo ""
echo "IMPORTANT: Make sure your computer won't go to sleep!"
echo "==========================================================================="
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate
conda activate dfs-analyzer

echo "Starting batch experiments..."
echo ""

# Run the batch script
python run_hypercube_variable_batch.py

echo ""
echo "==========================================================================="
echo "Batch completed at: $(date)"
echo "==========================================================================="

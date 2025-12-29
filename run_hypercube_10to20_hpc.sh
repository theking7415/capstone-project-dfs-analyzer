#!/bin/bash
#SBATCH --job-name=hypercube_10-20d
#SBATCH --output=hypercube_batch_%j.out
#SBATCH --error=hypercube_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=compute
# NOTE: Adjust partition name based on your HPC (might be 'batch', 'standard', 'normal', etc.)

# Print job information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=============================================="
echo

# Load conda module
module load anaconda3  # Adjust based on 'module avail' output

# Activate conda environment
source activate dfs-analyzer
# OR: conda activate dfs-analyzer

# Verify environment
echo "Python version:"
python --version
echo

echo "Checking dependencies..."
python -c "import numpy; import matplotlib; import scipy; print('✓ All dependencies available')"
echo

# Run the batch experiments
echo "Starting hypercube experiments (10D-20D)..."
echo
python run_hypercube_10to20_batch.py

# Print completion info
echo
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="

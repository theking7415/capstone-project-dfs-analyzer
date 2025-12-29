#!/bin/bash
#SBATCH --job-name=dfs_hypercube
#SBATCH --output=logs/hypercube_%A_%a.out
#SBATCH --error=logs/hypercube_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=10-20

# Example SLURM job script for running DFS experiments on HPC cluster
# This script runs experiments for hypercubes from dimension 10 to 20

# Loads required modules (adjust for your HPC environment)
module load python/3.10
# module load numpy/1.24
# module load scipy/1.10
# module load matplotlib/3.7

# Activates virtual environment if using one
# source ~/venvs/dfs-analyzer/bin/activate

# Sets up variables
DIMENSION=${SLURM_ARRAY_TASK_ID}
NUM_VERTICES=$((2**DIMENSION))

# Calculates recommended samples based on dimension
if [ $DIMENSION -le 10 ]; then
    SAMPLES_PER_VERTEX=3000
elif [ $DIMENSION -le 15 ]; then
    SAMPLES_PER_VERTEX=2000
else
    SAMPLES_PER_VERTEX=1000
fi

NUM_SAMPLES=$((SAMPLES_PER_VERTEX * NUM_VERTICES))

# Creates output directory
mkdir -p logs
mkdir -p results/hypercube_${DIMENSION}d

# Prints job information
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Dimension: ${DIMENSION}"
echo "Vertices: ${NUM_VERTICES}"
echo "Samples: ${NUM_SAMPLES}"
echo "=========================================="
echo ""

# Runs the experiment
python3 run_batch.py \
    --graph hypercube \
    --dimension ${DIMENSION} \
    --samples ${NUM_SAMPLES} \
    --seed 1832479182 \
    --output results/hypercube_${DIMENSION}d \
    --formats csv txt pickle \
    --progress 10000 \
    --no-plots

# Captures exit code
EXIT_CODE=$?

# Prints completion message
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

exit $EXIT_CODE

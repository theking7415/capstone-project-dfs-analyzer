#!/bin/bash
#PBS -N hypercube_10-20d
#PBS -o hypercube_batch.out
#PBS -e hypercube_batch.err
#PBS -l ncpus=52
#PBS -l walltime=48:00:00
#PBS -q cpu

# Print job information
echo "=============================================="
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "CPUs: $PBS_NUM_PPN"
echo "Start Time: $(date)"
echo "Working Directory: $PBS_O_WORKDIR"
echo "=============================================="
echo

# Change to submission directory
cd $PBS_O_WORKDIR

# Load Anaconda module
module load compiler/anaconda3

# Activate conda environment
source activate dfs-analyzer

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

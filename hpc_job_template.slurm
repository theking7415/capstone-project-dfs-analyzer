#!/bin/bash
#SBATCH --job-name=dfs-analyzer         # Job name
#SBATCH --output=dfs-%j.out             # Standard output (%j = job ID)
#SBATCH --error=dfs-%j.err              # Standard error
#SBATCH --time=04:00:00                 # Time limit (4 hours)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=16              # CPU cores per task
#SBATCH --mem=32G                       # Memory per node
#SBATCH --partition=standard            # Partition name (adjust for your HPC)

# =============================================================================
# DFS Graph Analyzer - HPC Job Script Template
# =============================================================================
#
# This script runs DFS analysis experiments on HPC clusters using SLURM.
#
# SETUP INSTRUCTIONS:
# 1. Copy this file and customize the SBATCH directives above for your HPC
# 2. Update the graph parameters and sample sizes below
# 3. Adjust --cpus-per-task based on your experiment size
# 4. Submit with: sbatch hpc_job_template.slurm
#
# RESOURCE RECOMMENDATIONS:
#   Dimension  | Vertices | Samples  | CPUs | Memory | Time
#   -----------|----------|----------|------|--------|-------
#   10D        | 1,024    | 100k     | 8    | 8G     | 10min
#   12D        | 4,096    | 200k     | 16   | 16G    | 20min
#   15D        | 32,768   | 1M       | 32   | 32G    | 1hr
#   18D        | 262,144  | 5M       | 64   | 64G    | 3hr
#   20D        | 1,048,576| 10M      | 64   | 128G   | 8hr
#
# =============================================================================

# Print job information
echo "========================================="
echo "DFS Graph Analyzer - HPC Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "========================================="
echo

# Load required modules (adjust for your HPC environment)
# Example for module-based HPC systems:
# module load python/3.10
# module load miniconda3

# Activate conda environment
# Option 1: If you installed via conda environment
source activate dfs-analyzer

# Option 2: If using system Python with pip
# export PATH=/home/username/.local/bin:$PATH

# Verify environment
echo "Python version:"
python --version
echo

echo "Checking graph-tool availability:"
python -c "from dfs_analyzer.core.rdfs_graphtool import is_available; print('graph-tool:', 'Available' if is_available() else 'Not available')"
echo

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Graph parameters
GRAPH_TYPE="hypercube"    # Options: hypercube, petersen
DIMENSION=15              # For hypercube: d (creates 2^d vertices)
                         # For petersen: n (creates 2n vertices with k parameter)
PETERSEN_K=2             # Only used if GRAPH_TYPE=petersen

# Experiment parameters
NUM_SAMPLES=1000000      # Number of RDFS samples (1 million)
RNG_SEED=1832479182      # Random seed for reproducibility

# Analysis focus (for v0.3.0 features)
ANALYSIS_FOCUS="full"    # Options: full, neighbors, opposite, custom

# Parallel processing
NUM_PROCESSES=$SLURM_CPUS_PER_TASK  # Use all allocated CPUs

# Output directory
OUTPUT_DIR="./hpc_results"
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

echo "Starting experiment..."
echo "Graph: $GRAPH_TYPE, Dimension: $DIMENSION"
echo "Samples: $NUM_SAMPLES"
echo "Processes: $NUM_PROCESSES"
echo

# Create Python script to run experiment
cat > run_hpc_experiment.py << 'PYTHON_SCRIPT'
import sys
import os
from dfs_analyzer.experiments.config import ExperimentConfig
from dfs_analyzer.experiments.runner import ExperimentRunner

# Get parameters from environment
graph_type = os.environ.get('GRAPH_TYPE', 'hypercube')
dimension = int(os.environ.get('DIMENSION', '10'))
petersen_k = int(os.environ.get('PETERSEN_K', '2'))
num_samples = int(os.environ.get('NUM_SAMPLES', '100000'))
rng_seed = int(os.environ.get('RNG_SEED', '1832479182'))
num_processes = int(os.environ.get('NUM_PROCESSES', '1'))
output_dir = os.environ.get('OUTPUT_DIR', './hpc_results')

# Create configuration
config = ExperimentConfig(
    graph_type=graph_type,
    dimension=dimension,
    petersen_k=petersen_k if graph_type == 'petersen' else None,
    num_samples=num_samples,
    rng_seed=rng_seed,
    export_formats=['csv', 'json', 'txt'],
    output_dir=output_dir
)

# Progress callback
def progress(current, total):
    if current % (total // 20) == 0 or current == total:  # Print every 5%
        percent = 100 * current / total
        print(f"  Progress: {current}/{total} ({percent:.1f}%)", flush=True)

# Run experiment
print(f"Running experiment with {num_processes} processes...")
runner = ExperimentRunner()
results = runner.run(config, progress_callback=progress, num_processes=num_processes)

print("\n" + "="*50)
print("EXPERIMENT COMPLETE")
print("="*50)
print(results.get_summary())
print(f"\nResults saved to: {results.output_dir}")

PYTHON_SCRIPT

# Run the experiment
python run_hpc_experiment.py

# Clean up
rm run_hpc_experiment.py

# =============================================================================
# JOB COMPLETION
# =============================================================================

echo
echo "========================================="
echo "Job completed at: $(date)"
echo "Results directory: $OUTPUT_DIR"
echo "========================================="

# Optional: Compress results for transfer
# tar -czf dfs-results-$SLURM_JOB_ID.tar.gz $OUTPUT_DIR

exit 0

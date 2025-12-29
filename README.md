# DFS Graph Analyzer

A production-ready tool for analyzing randomized depth-first search (RDFS) behavior on symmetric regular graphs. Provides empirical validation of the **(n-1)/2 expected behavior** through comprehensive statistical analysis and visualization.

## Overview

For symmetric regular graphs, the average discovery number of a vertex in randomized DFS exhibits the expected behavior of **(n-1)/2**, where **n** is the number of vertices in the graph.

This tool validates this behavior by running thousands of randomized DFS traversals and computing statistical properties of the discovery numbers across multiple graph types.

## Features

**Core Capabilities:**
- **Dual Interface** - Interactive CLI and web-based GUI (Streamlit)
- **Multiple Graph Types** - Hypercube, Generalized Petersen, Triangular Lattice, Torus Grid, Hexagonal Lattice, Complete Graph, N-Dimensional Grid, and G(n,p) Random Graphs
- **Advanced Analysis** - Full graph analysis, neighbor focus, opposite vertex analysis, custom vertex pairs
- **Statistical Analysis** - Comprehensive statistics with automated validation
- **Multiple Export Formats** - CSV, JSON, TXT, Pickle
- **Advanced Visualizations** - Per-vertex bar charts, histograms, layer analysis plots
- **High Performance** - Optional graph-tool backend (40-100x speedup) with multiprocessing support
- **HPC Ready** - SLURM job templates and SSH-accessible GUI
- **Reproducible** - Seeded RNG for consistent results

**Analysis Methods:**
1. **RDFS Sampling** - Randomized depth-first search with configurable sample sizes
2. **Laplacian Analysis** - Theoretical hitting times using graph Laplacian matrix
3. **Layer Analysis** - BFS-based distance grouping and discovery pattern analysis
4. **Distance Metrics** - L1 and L2 distance correlation studies

## Installation

### Requirements

- Python 3.10 or higher
- pip or conda

### Standard Installation

```bash
pip install -r requirements.txt
```

### High-Performance Installation (Recommended for HPC)

For 40-100x speedup with C++ backend and multiprocessing:

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate dfs-analyzer

# Verify graph-tool installation
python -c "from dfs_analyzer.core.rdfs_graphtool import is_available; print('graph-tool:', is_available())"
```

See `INSTALLATION.md` for detailed platform-specific instructions.

## Quick Start

### Interactive CLI

```bash
python run_analyzer.py
```

The CLI provides a menu-driven interface with:
- 5 analysis types (Full, Neighbors, Opposite, Custom, G(n,p) Batch)
- 8 graph types
- Configurable sample sizes and output formats
- Real-time progress tracking

### Web GUI

```bash
python run_gui.py
```

Opens browser to `http://localhost:8501` with interactive interface featuring:
- Real-time parameter configuration
- Progress bars and validation
- In-browser results display
- HPC-compatible via SSH port forwarding

See `GUI_README.md` for detailed GUI documentation.

### HPC Usage

```bash
# Setup on HPC
conda env create -f environment.yml
conda activate dfs-analyzer

# Customize and submit job
cp hpc_job_template.slurm my_job.slurm
# Edit parameters in my_job.slurm
sbatch my_job.slurm
```

See `HPC_GUIDE.md` for complete HPC deployment instructions.

## Graph Types

### Symmetric Regular Graphs

**Hypercube (d-dimensional)**
- Vertices: 2^d binary tuples (e.g., `(0,1,0)` for d=3)
- Edges: Hamming distance 1 (differ in exactly one bit)
- Degree: d (d-regular)
- Tested: Dimensions 3-19 (8 to 524,288 vertices)

**Generalized Petersen GP(n,k)**
- Vertices: 2n vertices in two rings (outer and inner)
- Edges: Sequential outer, skip-k inner, radial spokes
- Degree: 3 (3-regular)
- Tested: n=3-100, various k values

**Triangular Lattice**
- Vertices: rows × cols with axial coordinates (q, r)
- Edges: 6 neighbors per vertex (hexagonal coordination)
- Degree: 6 (6-regular)
- Topology: Torus with periodic boundaries
- Tested: Up to 100×100 (10,000 vertices)

**Torus Grid**
- Vertices: rows × cols with (row, col) coordinates
- Edges: 4 cardinal neighbors with wraparound
- Degree: 4 (4-regular)
- Topology: Torus with periodic boundaries
- Tested: Up to 100×100 (10,000 vertices)

**Hexagonal Lattice**
- Vertices: rows × cols (graphene structure)
- Edges: 3 neighbors per vertex (honeycomb pattern)
- Degree: 3 (3-regular, bipartite)
- Topology: Torus with periodic boundaries
- Tested: Up to 50×50 (2,500 vertices)

**Complete Graph K_n**
- Vertices: n labeled 0 to n-1
- Edges: All possible edges (every vertex connected to all others)
- Degree: n-1 (maximum possible)
- Tested: Up to K_100

**N-Dimensional Grid**
- Vertices: size^d d-tuples
- Edges: ±1 in each dimension with wraparound
- Degree: 2d (constant)
- Tested: 3D-10D with various sizes

### Exploratory Analysis

**G(n,p) Random Graphs (Erdos-Renyi)**
- Vertices: n labeled vertices
- Edges: Each edge appears independently with probability p
- Properties: NOT regular or symmetric
- Purpose: Comparative analysis with regular graphs
- Features: Connectivity checking, batch mode

## Output Files

Each experiment creates a timestamped directory with:

**Standard Outputs:**
- `summary.txt` - Human-readable summary with validation results
- `visualization.png` - Bar chart of mean discovery numbers (if enabled)
- `histogram.png` - Distribution by discovery number buckets (if enabled)
- `layer_analysis.png` - BFS distance layer analysis (if enabled)

**Optional Outputs (configurable):**
- `data.csv` - Per-vertex statistics (Excel-compatible)
- `detailed_stats.txt` - Comprehensive statistics per vertex
- `layer_statistics_bfs.csv` - Layer-by-layer BFS analysis
- `data.json` - Machine-readable format
- `data.pickle` - Raw Python data for further analysis

**Storage-Efficient Mode:**
By default, only `summary.txt` is generated to prevent storage issues on large experiments.

## Project Structure

```
dfs_analyzer/
├── core/                         # Core algorithms
│   ├── graphs.py                # Graph implementations (8 types)
│   ├── rdfs.py                  # RDFS algorithm (pure Python)
│   ├── rdfs_graphtool.py        # High-performance C++ backend
│   ├── statistics.py            # Statistical validation
│   ├── neighbor_analysis.py     # Immediate neighbor focus
│   ├── opposite_analysis.py     # Opposite vertex (hypercube)
│   ├── custom_vertex_analysis.py # Custom vertex pairs
│   ├── gnp_graph.py             # G(n,p) random graphs
│   └── random_walk.py           # Laplacian hitting times
├── experiments/                  # Experiment management
│   ├── config.py                # Experiment configuration
│   ├── runner.py                # Main experiment orchestration
│   ├── results.py               # Results storage/export/visualization
│   ├── neighbor_runner.py       # Neighbor analysis runner
│   ├── opposite_runner.py       # Opposite vertex runner
│   ├── custom_vertex_runner.py  # Custom pair runner
│   ├── gnp_batch_runner.py      # G(n,p) batch experiments
│   └── random_walk_runner.py    # Laplacian analysis runner
└── ui/                          # User interfaces
    ├── cli.py                   # Interactive CLI (all features)
    └── streamlit_app.py         # Web GUI (HPC-compatible)
```

## Performance

**Pure Python:**
- Suitable for graphs up to ~1000 vertices
- Standard analysis: minutes to hours depending on samples

**With graph-tool (C++ backend):**
- 40x single-core speedup
- Multiprocessing support (all graph types)
- Near-linear scaling with CPU cores (70-90% efficiency)
- HPC performance: 10-56x speedup on 16-64 cores

**Example (Hypercube 15D, 100k samples):**
- Laptop single-core: 2.8 hours
- HPC 32 cores: 3-5 minutes (34-56x speedup)

See `MULTIPROCESSING_UPDATE.md` for technical details.

## Results Summary

**Validation Accuracy:**
- Hypercubes 3D-19D: Validated with <0.01% error
- Generalized Petersen: Validated across multiple (n,k) combinations
- Lattice graphs: Validated up to 10,000 vertices
- Complete graphs: Validated up to K_100

**Key Findings:**
- The (n-1)/2 expected behavior holds consistently for all symmetric regular graphs tested
- Layer analysis reveals correlation between BFS distance and discovery order
- Histogram analysis shows tight clustering around expected value
- Non-regular graphs (G(n,p)) show different behavior patterns

## Documentation

- `README.md` - This file (overview and quick start)
- `QUICKSTART.md` - Step-by-step tutorial
- `GUI_README.md` - Web interface documentation
- `HPC_GUIDE.md` - HPC deployment guide
- `INSTALLATION.md` - Platform-specific installation
- `INSTALL_GRAPHTOOL.md` - graph-tool installation guide
- `WINDOWS_NOTES.md` - Windows-specific help
- `CHANGELOG.md` - Version history
- `SECURITY_REVIEW.md` - Security analysis

## Usage Tips

### Sample Size Recommendations

Uses proportional scaling based on graph type:
- **Hypercubes:** 2000 samples per vertex (consistent statistical precision)
- **Petersen graphs:** 1000 samples per vertex (constant degree 3)
- **Lattice graphs:** 1000 samples per vertex (constant degree)
- **Random graphs:** 2000 samples per vertex (variable degree)

### Reproducibility

- Default RNG seed (1832479182) ensures identical results across runs
- All experiment parameters saved with results
- Seeded multiprocessing for reproducible parallel execution

### HPC Best Practices

- Use graph-tool backend for 40-100x speedup
- Allocate CPUs appropriately (16-64 recommended)
- Use storage-efficient mode for large experiments
- Monitor with `squeue` and `tail -f` on output files

## Research Background

This project was developed as part of a capstone research project at Ashoka University investigating depth-first search traversal properties on symmetric regular graphs. The tool provides both empirical validation through RDFS sampling and theoretical analysis using graph Laplacian methods.

**Version:** 0.6.0
**Status:** Production-ready for academic research and publication

---

For detailed documentation, see the files listed in the Documentation section above.

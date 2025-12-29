# Installation Guide - DFS Graph Analyzer

This guide covers installation for academic researchers.

## Quick Start (Recommended for Academic Users)

### Method 1: Conda Installation (Includes graph-tool for 50-100x speedup)

```bash
# Clone repository
git clone https://github.com/theking7415/capstone-project-dfs-analyzer
cd capstone-project-dfs-analyzer

# Create conda environment with all dependencies
conda env create -f environment.yml

# Activate environment
conda activate dfs-analyzer

# Verify installation
python -c "from dfs_analyzer.core.rdfs_graphtool import is_available; print('graph-tool:', 'Available ✓' if is_available() else 'Not available')"

# Run GUI or CLI
python run_gui.py
python run_analyzer.py
```

**This is the recommended method** - includes graph-tool for maximum performance.

---

## Method 2: pip Installation (Fallback - works everywhere but slower)

If you cannot use conda (or don't need maximum performance):

```bash
# Clone repository
git clone https://github.com/theking7415/capstone-project-dfs-analyzer
cd capstone-project-dfs-analyzer

# Install dependencies
pip install -r requirements.txt

# Run
python run_gui.py
python run_analyzer.py
```

**Note:** This method does NOT include graph-tool. The software will automatically use the slower pure Python implementation.

---

## Performance Comparison

| Method | 10D Hypercube (3M samples) | graph-tool |
|--------|---------------------------|------------|
| Conda (with graph-tool) | ~3-5 minutes | ✓ |
| pip (pure Python) | ~4.5 hours | ✗ |

**For large graphs (d > 8), graph-tool is HIGHLY recommended.**

---

## Platform-Specific Instructions

### Ubuntu/Debian

**Option A: Conda (Recommended)**
```bash
# Install miniconda if not installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Then follow Quick Start above
conda env create -f environment.yml
conda activate dfs-analyzer
```

**Option B: System packages**
```bash
sudo apt-get update
sudo apt-get install python3-graph-tool python3-pip
pip install streamlit numpy scipy matplotlib networkx
```

### macOS

**Option A: Conda (Recommended)**
```bash
# Install miniconda if not installed
brew install miniconda

# Then follow Quick Start above
conda env create -f environment.yml
conda activate dfs-analyzer
```

**Option B: Homebrew**
```bash
brew install graph-tool
pip3 install -r requirements.txt
```

### Windows

**graph-tool is NOT natively supported on Windows.** You have two options:

**Option A: WSL (Windows Subsystem for Linux) - Recommended**
```bash
# 1. Install WSL2 (Windows 11/10)
wsl --install

# 2. Inside WSL, follow Ubuntu instructions above
conda env create -f environment.yml
conda activate dfs-analyzer
```

**Option B: Pure Python (No graph-tool)**
```bash
# Just use pip installation
pip install -r requirements.txt
python run_gui.py
```

*Note: Without graph-tool, large experiments will be slow. For research purposes, WSL is recommended.*

### HPC Clusters

Most HPC clusters use module systems. Here's the typical workflow:

```bash
# Load Python module (version may vary)
module load python/3.10

# Option A: Create conda environment (if conda available)
conda env create -f environment.yml
conda activate dfs-analyzer

# Option B: User installation with pip (if conda not available)
pip install --user -r requirements.txt

# If graph-tool is pre-installed as a module:
module load graph-tool
pip install --user streamlit numpy scipy matplotlib networkx

# For GUI access from local machine:
# Terminal 1 (on HPC):
python run_gui.py

# Terminal 2 (on your laptop):
ssh -L 8501:localhost:8501 user@hpc-cluster.edu
# Then open browser to: http://localhost:8501
```

**Note:** Contact your HPC system administrator if graph-tool is not available. They can install it system-wide.

---

## Verifying Installation

After installation, verify everything works:

```bash
# Check graph-tool availability
python -c "from dfs_analyzer.core.rdfs_graphtool import is_available; print('graph-tool available:', is_available())"

# Run quick test
python test_hypercube.py

# Launch GUI
python run_gui.py
```

You should see:
- `graph-tool available: True` (if conda method)
- Test passes with 0.0000% error
- GUI opens in browser

---

## Troubleshooting

### "graph-tool not found"

This is expected if you used pip installation. The software will still work, just slower.

To add graph-tool:
```bash
# Create conda environment
conda create -n dfs-analyzer python=3.10
conda activate dfs-analyzer
conda install -c conda-forge graph-tool
pip install streamlit numpy scipy matplotlib networkx
```

### "streamlit command not found" (Windows)

Use Python module syntax:
```bash
python -m streamlit run dfs_analyzer/ui/streamlit_app.py
```

Or just use the launcher:
```bash
python run_gui.py
```

### Conda environment conflicts

If you have existing conda environments:
```bash
# Remove old environment
conda env remove -n dfs-analyzer

# Recreate fresh
conda env create -f environment.yml
```

### HPC: "Cannot install packages"

On HPC, you typically don't have admin rights. Use:
```bash
pip install --user -r requirements.txt
```

Or request that your HPC admin install graph-tool system-wide.

---

## Dependencies

### Required (included in requirements.txt):
- Python ≥ 3.10
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- matplotlib ≥ 3.7.0
- streamlit ≥ 1.28.0
- networkx ≥ 3.0

### Optional (for performance):
- graph-tool ≥ 2.45 (50-100x speedup)

**Install via conda for best results:**
```bash
conda install -c conda-forge graph-tool
```

---

## Development Installation

For development/contribution:

```bash
git clone https://github.com/theking7415/capstone-project-dfs-analyzer
cd capstone-project-dfs-analyzer

# Create conda environment
conda env create -f environment.yml
conda activate dfs-analyzer

# Install in editable mode
pip install -e .
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{dfs_analyzer_2025,
  author = {Mandava, Venkat Mahesh},
  title = {DFS Graph Analyzer: Validating the (n-1)/2 Conjecture},
  year = {2025},
  institution = {Ashoka University},
  url = {https://github.com/theking7415/capstone-project-dfs-analyzer}
}
```

---

## Support

For issues:
1. Check this installation guide
2. See `README.md` and `GUI_README.md`
3. Open an issue on GitHub

**For academic researchers:** If graph-tool installation fails on your HPC cluster, contact your system administrator. They can install it system-wide.

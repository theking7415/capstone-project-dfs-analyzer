# DFS Graph Analyzer - Web GUI

Interactive web interface for exploring DFS behavior on symmetric regular graphs.

## Quick Start

### Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch GUI
python run_gui.py

# Browser opens automatically to: http://localhost:8501
```

### HPC Usage

The GUI works perfectly on HPC clusters via SSH port forwarding:

```bash
# Step 1: SSH with port forwarding
ssh -L 8501:localhost:8501 username@hpc-cluster.edu

# Step 2: Navigate to project
cd /path/to/dfs-analyzer

# Step 3: Load Python module (if needed)
module load python/3.10

# Step 4: Install dependencies (first time only)
pip install --user -r requirements.txt

# Step 5: Launch GUI
python run_gui.py
```

Then on your **local machine**, open your browser to: **http://localhost:8501**

The GUI will be running on the HPC cluster, but you access it through your local browser!

## Features

### 📊 Analysis Types

1. **Full Graph Analysis** - Analyze all vertices
2. **Immediate Neighbors** - Focus on vertices adjacent to start
3. **Opposite Vertex** - Diagonally opposite vertex (hypercube only)
4. **Custom Vertex Pair** - Specify any start and target vertices
5. **G(n,p) Batch Mode** - Analyze multiple random graphs

### 🎯 Analysis Methods

- **RDFS (Randomized DFS)** - Empirical simulations
- **Laplacian (Random Walk)** - Theoretical exact computation

### 📈 Graph Types

- **Hypercube** - d-dimensional binary graphs (2^d vertices)
- **Generalized Petersen** - GP(n,k) graphs (2n vertices)
- **G(n,p) Random** - Erdős-Rényi random graphs

### ✨ Interactive Features

- **Real-time progress bars** - See experiment progress
- **Dynamic parameter validation** - Instant feedback
- **Automatic recommendations** - Suggested sample sizes
- **Results display** - View summaries in-browser
- **File output** - Save detailed results to disk

## Interface Layout

### Sidebar (Left)
- Configuration options
- Graph type selection
- Parameter inputs
- Vertex selection (for custom pairs)
- Sampling configuration
- Advanced options

### Main Area (Right)
- Current configuration summary
- Run experiment button
- Progress indicators
- Results display

### Info Panel (Far Right)
- Conjecture explanation
- Graph type descriptions
- Method descriptions
- HPC usage instructions

## Example Workflows

### Example 1: Quick Hypercube Test

1. Select "Full Graph Analysis"
2. Choose "RDFS (Empirical)"
3. Select "Hypercube"
4. Set dimension to 5 (32 vertices)
5. Use recommended samples (≈96,000)
6. Click "Run Experiment"
7. View results in browser

### Example 2: Custom Vertex Pair

1. Select "Custom Vertex Pair"
2. Choose "RDFS (Empirical)"
3. Select "Hypercube"
4. Set dimension to 5
5. Enter start vertex: `0,0,0,0,0`
6. Enter target vertex: `1,0,1,0,1`
7. See Hamming distance: 3
8. Set samples to 1000
9. Run and view results

### Example 3: G(n,p) Batch Analysis

1. Select "G(n,p) Batch Mode"
2. Set n = 50 vertices
3. Set p = 0.10 (note: below connectivity threshold)
4. Set number of graphs = 10
5. Set samples per graph = 500
6. Run batch experiment
7. View aggregate statistics

### Example 4: Opposite Vertex (Hypercube)

1. Select "Opposite Vertex (Hypercube only)"
2. Choose "Laplacian (Theoretical)"
3. Dimension is auto-set to hypercube
4. Set dimension to 5
5. Run (instant result - no sampling needed)
6. Compare with RDFS method

## Advanced Options

Click "🔧 Advanced Options" in the sidebar to access:

- **RNG Seed** - For reproducible results (default: 1832479182)
- **Output Directory** - Where to save files (default: data_output)
- **Output Files**:
  - Save CSV - Per-vertex statistics
  - Save detailed stats - Full statistical report
  - Save plots - Visualization graphs

## Output Files

Results are saved to: `data_output/experiment-name_timestamp/`

Files generated (depending on options):
- `summary.txt` - Always generated
- `data.csv` - If "Save CSV" enabled
- `detailed_stats.txt` - If "Save detailed stats" enabled
- `visualization.png` - If "Save plots" enabled

## When to Use GUI vs CLI

### Use GUI When:
- ✅ Exploring parameters interactively
- ✅ Running small to medium experiments
- ✅ Visualizing results immediately
- ✅ Learning the tool
- ✅ Prototyping analysis

### Use CLI When:
- ✅ Running large batch jobs
- ✅ Automating experiments with scripts
- ✅ Using HPC job schedulers (SLURM, PBS)
- ✅ Processing many configurations
- ✅ Integrating with pipelines

**Both interfaces use the same core logic - choose based on your workflow!**

## Troubleshooting

### GUI won't start
```bash
# Check if streamlit is installed
python -c "import streamlit; print(streamlit.__version__)"

# If not, install it
pip install streamlit>=1.28.0
```

### Port 8501 already in use
```bash
# Use a different port
streamlit run dfs_analyzer/ui/streamlit_app.py --server.port 8502

# Then access: http://localhost:8502
```

### HPC: Cannot connect to GUI
```bash
# Verify port forwarding is active
ssh -L 8501:localhost:8501 username@hpc.edu

# Check if streamlit is running on HPC
ps aux | grep streamlit

# Check firewall (on HPC)
# May need to request firewall rule from HPC admin
```

### Slow performance on HPC
The GUI runs on the HPC **login node**, which is fine for:
- Small to medium experiments (< 10,000 samples)
- Interactive parameter exploration
- Viewing results

For **large experiments**:
- Use CLI with batch scripts
- Submit jobs to compute nodes via SLURM/PBS
- Use `run_batch.py` for automation

## Comparison: GUI vs CLI vs Batch

| Feature | GUI | CLI | Batch |
|---------|-----|-----|-------|
| Interactive | ✅ Yes | ✅ Yes | ❌ No |
| Visual feedback | ✅ Yes | ⚠️ Text | ❌ No |
| Parameter exploration | ✅ Easy | ⚠️ Manual | ❌ Scripted |
| Large experiments | ⚠️ Limited | ✅ Yes | ✅ Yes |
| HPC compatible | ✅ Yes* | ✅ Yes | ✅ Yes |
| Automation | ❌ No | ⚠️ Scripts | ✅ Yes |
| Learning curve | ✅ Easy | ⚠️ Medium | ⚠️ Medium |

*Via SSH port forwarding

## Tips for Best Results

1. **Start small** - Test with small dimensions first
2. **Use recommendations** - Follow suggested sample sizes
3. **Save incrementally** - Enable CSV/detailed stats for important runs
4. **Compare methods** - Run both RDFS and Laplacian to validate
5. **Use custom pairs** - Explore interesting vertex combinations
6. **Batch G(n,p)** - Generate multiple graphs for robust statistics

## Support

For issues or questions:
- Check documentation: `README.md`, `QUICKSTART.md`
- Review examples in `test_*.py` files
- Check code comments in `dfs_analyzer/` modules

## Citation

If you use this tool in research, please cite the project appropriately.

Author: Venkat Mahesh Mandava
Institution: Ashoka University
Year: 2025

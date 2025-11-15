# Quick Start Guide - DFS Graph Analyzer

## Installation

1. **Check Python version** (requires Python 3.10+):
   ```bash
   python3 --version
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python3 test_cli_automated.py
   ```

   You should see: `✓ CLI backend test completed successfully!`

## Running the Interactive CLI

```bash
python3 run_analyzer.py
```

## Your First Experiment

When the CLI starts, you'll see:

```
======================================================================
                        DFS GRAPH ANALYZER
======================================================================

MAIN MENU
1. Run new experiment
2. Help & Documentation
3. About
4. Exit

Enter your choice:
```

### Step-by-Step:

1. **Press `1`** to run a new experiment
2. **Enter dimension**: Try `3` for a quick test (8 vertices)
3. **Enter samples**: Try `1000` for a quick test
4. **Advanced options?**: Press `n` for no (uses defaults)
5. **Confirm**: Press `y` to proceed

The experiment will run and show a progress bar:

```
Progress: [████████████████████████████████] 100% (1000/1000)

======================================================================
EXPERIMENT RESULTS
======================================================================
Graph: Hypercube 3D (8 vertices)
Samples: 1000
...
✓ CONJECTURE VALIDATED
======================================================================
```

## Viewing Results

Results are saved in `data_output/` directory. Each experiment creates a folder with:

- **summary.txt** - Quick summary
- **data.csv** - Open in Excel
- **visualization.png** - Bar chart

## Example Experiments

### Quick Test (30 seconds)
- Dimension: 3
- Samples: 1000

### Standard Analysis (2-3 minutes)
- Dimension: 5
- Samples: 10000

### Publication Quality (10-15 minutes)
- Dimension: 6
- Samples: 25000

## Tips

1. **Start small** - Try dimension 3-4 first to get familiar
2. **More samples = more accurate** but takes longer
3. **Use defaults** - The default settings work well for most cases
4. **Check outputs** - Open the CSV file in Excel to see per-vertex statistics

## Troubleshooting

### Import Errors
Make sure you're in the correct directory:
```bash
cd "/mnt/c/Users/mahes/Desktop/Ashoka/Capstone project/My data"
```

### Permission Errors
Make sure scripts are executable:
```bash
chmod +x run_analyzer.py
```

### Module Not Found
Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Next Steps

1. Read `README.md` for detailed documentation
2. Try the Help menu (option 2) in the CLI
3. Experiment with different dimensions and sample sizes
4. Check the `examples/` directory (coming soon)

## Getting Help

- Press `2` in the main menu for built-in help
- Check `README.md` for full documentation
- View generated files in `data_output/` for examples

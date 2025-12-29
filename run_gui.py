#!/usr/bin/env python3
"""
Launch the DFS Graph Analyzer web GUI.

This script launches the Streamlit web interface for interactive exploration
of DFS experiments on symmetric regular graphs.

Usage:
    python run_gui.py

HPC Usage:
    # Terminal 1: SSH with port forwarding
    ssh -L 8501:localhost:8501 user@hpc.edu
    cd /path/to/dfs-analyzer
    python run_gui.py

    # Terminal 2 / Browser on local machine:
    # Navigate to: http://localhost:8501

The GUI provides an interactive interface for:
- Selecting graph types and parameters
- Choosing analysis focus (full graph, neighbors, opposite, custom pairs)
- Running experiments with real-time progress
- Viewing results and summaries

For batch jobs or HPC scripting, use the CLI instead:
    python run_analyzer.py     # Interactive CLI
    python run_batch.py ...    # Batch mode
"""

import os
import sys
from pathlib import Path

# Ensures we're in the correct directory
project_root = Path(__file__).parent
os.chdir(project_root)

# Adds project to path
sys.path.insert(0, str(project_root))

# Checks if streamlit is installed
try:
    import streamlit
except ImportError:
    print("Error: Streamlit is not installed.")
    print("\nPlease install required dependencies:")
    print("  pip install -r requirements.txt")
    print("\nOr install streamlit directly:")
    print("  pip install streamlit")
    sys.exit(1)

# Launches Streamlit app
if __name__ == "__main__":
    # Gets the path to the streamlit app
    app_path = project_root / "dfs_analyzer" / "ui" / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: Cannot find Streamlit app at {app_path}")
        sys.exit(1)

    print("=" * 70)
    print("DFS GRAPH ANALYZER - WEB GUI".center(70))
    print("=" * 70)
    print()
    print("Starting Streamlit server...")
    print()
    print("Navigate to: http://localhost:8501")
    print("(Browser will not open automatically in WSL)")
    print()
    print("HPC Users: Make sure you set up SSH port forwarding:")
    print("  ssh -L 8501:localhost:8501 user@hpc.edu")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    # Runs streamlit using Python module syntax (Windows/WSL-compatible)
    # This works even if streamlit is not on PATH
    # --server.headless=true disables browser auto-opening to prevent WSL errors
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless=true"
    ])

#!/usr/bin/env python3
"""
Launcher script for DFS Graph Analyzer CLI.
"""

import sys
import os

# Add the current directory to path so we can import dfs_analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dfs_analyzer.ui.cli import main

if __name__ == "__main__":
    main()

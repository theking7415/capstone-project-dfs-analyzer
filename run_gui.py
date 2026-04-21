#!/usr/bin/env python3
"""
Launch the Random DFS Graph Analyzer desktop GUI.

Opens a native PyQt6 window — no browser required.

Usage:
    python run_gui.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.resolve()
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("Error: PyQt6 is not installed.")
    print("\nInstall dependencies:")
    print("  pip install -r requirements_desktop.txt")
    sys.exit(1)

if __name__ == "__main__":
    from dfs_analyzer.ui.qt_app import main
    main()

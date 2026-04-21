# Random DFS Graph Analyzer

A desktop application for empirically investigating randomized depth-first search (RDFS) behavior on symmetric regular graphs. Built to validate the **(n-1)/2 conjecture**: the average discovery number of any vertex in a random DFS tends to (n-1)/2 as the graph grows.

## Overview

For symmetric regular graphs, the average position at which a vertex is discovered during a randomized DFS is conjectured to approach **(n-1)/2**, where **n** is the number of vertices. This tool runs thousands of RDFS traversals and computes per-vertex and per-layer statistics to test this empirically across many graph families.

## Features

- **Native desktop GUI** built with PyQt6 — no browser required
- **Multiple graph types** — Hypercube, Generalized Petersen, Triangular Lattice, Torus Grid, Complete Graph, N-Dimensional Grid, and G(n,p) Random Graphs
- **Analysis modes** — Full graph, neighbor focus, opposite vertex (hypercube), custom vertex pairs
- **Layer analysis** — BFS-distance grouping to study how discovery order correlates with graph distance
- **Post-experiment tools** — Deviation analysis, sigmoid fitting, layer variance analysis, three-vertex distribution
- **Visualizations** — Per-vertex bar charts, histograms, layer plots, embedded directly in the app
- **Export** — CSV, JSON, TXT, Pickle

## Requirements

- Python 3.10 or higher
- Windows, macOS, or Linux

## Installation

```bash
pip install -r requirements_desktop.txt
```

## Running

```bash
python run_gui.py
```

## Building a Standalone Windows Executable

```bash
pip install pyinstaller
build_windows.bat
```

The executable will be at `dist\RandomDFSAnalyzer\RandomDFSAnalyzer.exe`. No Python installation required to run it.

## Graph Types

| Graph | Vertices | Degree |
|---|---|---|
| Hypercube (d-dim) | 2^d | d |
| Generalized Petersen GP(n,k) | 2n | 3 |
| Triangular Lattice (rows x cols) | rows x cols | 6 |
| Torus Grid (rows x cols) | rows x cols | 4 |
| Complete Graph K_n | n | n-1 |
| N-Dimensional Grid | size^d | 2d |
| G(n,p) Random Graph | n | varies |

## Project Structure

```
dfs_analyzer/
├── core/
│   ├── graphs.py                # Graph implementations
│   ├── rdfs.py                  # RDFS algorithm
│   ├── statistics.py            # Statistical analysis
│   ├── deviation_analysis.py    # Layer deviation analysis
│   ├── sigmoid_fitting.py       # Sigmoid curve fitting
│   └── layer_variance.py        # Layer variance analysis
├── experiments/
│   ├── config.py                # Experiment configuration
│   ├── runner.py                # Experiment orchestration
│   ├── results.py               # Results storage and export
│   └── three_vertex_runner.py   # Three-vertex distribution
└── ui/
    ├── qt_app.py                # PyQt6 desktop GUI
    └── cli.py                   # Command-line interface
```

## Research Background

Developed as a capstone research project at Ashoka University. The project investigates depth-first search traversal properties on symmetric regular graphs and provides empirical validation of the (n-1)/2 conjecture across graph families including hypercubes (3D–19D), Petersen graphs, and lattice graphs.

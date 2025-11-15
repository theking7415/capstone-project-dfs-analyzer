"""
DFS Graph Analyzer

A tool for validating the (n-1)/2 conjecture on symmetric regular graphs
using randomized depth-first search (RDFS).

The conjecture states that for large symmetric regular graphs, the average
discovery number of a node in random DFS tends to (n-1)/2, where n is the
number of nodes in the graph.
"""

__version__ = "0.1.0"
__author__ = "Venkat Mahesh Mandava"

from dfs_analyzer.core.graphs import Graph, Hypercube

__all__ = ["Graph", "Hypercube", "__version__"]

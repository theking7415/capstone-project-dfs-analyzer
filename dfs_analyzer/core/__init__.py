"""
Core algorithms and data structures for DFS graph analysis.
"""

from dfs_analyzer.core.graphs import Graph, Hypercube, GeneralizedPetersen, TriangularLattice, TorusGrid, HexagonalLattice, CompleteGraph, NDGrid
from dfs_analyzer.core.gnp_graph import ErdosRenyiGraph
from dfs_analyzer.core.rdfs import rdfs, collect_statistics, get_summary_stats

__all__ = [
    "Graph",
    "Hypercube",
    "GeneralizedPetersen",
    "TriangularLattice",
    "TorusGrid",
    "HexagonalLattice",
    "CompleteGraph",
    "NDGrid",
    "ErdosRenyiGraph",
    "rdfs",
    "collect_statistics",
    "get_summary_stats",
]

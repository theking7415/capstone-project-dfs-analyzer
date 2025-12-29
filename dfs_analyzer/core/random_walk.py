"""
Random walk analysis using graph Laplacian.

Computes expected hitting times and costs using the Laplacian matrix approach,
providing a theoretical complement to the empirical RDFS analysis.
"""

import numpy as np
import scipy.sparse.linalg as spla
import networkx as nx
from typing import TypeVar, Generic, Dict

from dfs_analyzer.core.graphs import Graph

Vertex = TypeVar("Vertex")


def compute_laplacian_hitting_times(
    graph: Graph[Vertex],
    target_vertex: Vertex = None
) -> Dict[Vertex, float]:
    """
    Computes expected hitting times using the Laplacian matrix.

    Uses the "single-factor" method with sparse LU decomposition to compute
    expected hitting times from all vertices to a target vertex.

    Args:
        graph: Graph instance to analyze.
        target_vertex: Target (sink) node. If None, uses the start vertex.

    Returns:
        Dictionary mapping each vertex to its expected hitting time to target.
    """
    # Uses start vertex if target not specified
    if target_vertex is None:
        target_vertex = graph.get_start_vertex()

    # Creates NetworkX graph for Laplacian computation
    nx_graph = _convert_to_networkx(graph)

    # Gets sorted list of all nodes for consistent ordering
    nodelist = sorted(list(nx_graph.nodes()))
    N = len(nodelist)

    # Gets the graph Laplacian matrix
    L = nx.laplacian_matrix(nx_graph, nodelist=nodelist).astype(np.float64)

    # Finds the index of the target node
    target_idx = nodelist.index(target_vertex)

    # Creates the minor of the Laplacian by removing target row and column
    minor_nodes_indices = [i for i in range(N) if i != target_idx]
    L_minor = L[minor_nodes_indices, :][:, minor_nodes_indices]

    # Factorizes L_minor using sparse LU decomposition
    lu_factor = spla.splu(L_minor.tocsc())

    # Computes effective resistances
    effective_resistances = np.zeros(N - 1)
    for i in range(N - 1):
        # Creates the standard basis vector e_x
        e_x = np.zeros(N - 1)
        e_x[i] = 1.0

        # Solves the system using the pre-computed LU factorization
        y = lu_factor.solve(e_x)

        # Extracts effective resistance from diagonal entry
        effective_resistances[i] = y[i]

    # Assembles the "current" vector b
    b = 1.0 / effective_resistances

    # Solves the final linear system L_minor * phi = b
    phi_costs = lu_factor.solve(b)

    # The full cost array includes cost from target to itself (0)
    all_costs = np.zeros(N)
    all_costs[minor_nodes_indices] = phi_costs

    # Creates dictionary mapping nodes to hitting times
    hitting_times = {}
    for i, node in enumerate(nodelist):
        hitting_times[node] = all_costs[i]

    return hitting_times


def get_laplacian_summary_stats(
    hitting_times: Dict[Vertex, float]
) -> Dict[str, float]:
    """
    Computes summary statistics from Laplacian hitting times.

    Args:
        hitting_times: Dictionary mapping vertices to hitting times.

    Returns:
        Dictionary with mean, std, min, max statistics.
    """
    # Extracts hitting time values
    values = list(hitting_times.values())

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def _convert_to_networkx(graph: Graph[Vertex]) -> nx.Graph:
    """
    Converts custom Graph to NetworkX graph.

    Args:
        graph: Graph instance to convert.

    Returns:
        NetworkX Graph object.
    """
    # Creates empty NetworkX graph
    nx_graph = nx.Graph()

    # Gets all vertices by doing a traversal
    visited = set()
    stack = [graph.get_start_vertex()]

    while stack:
        vertex = stack.pop()
        if vertex in visited:
            continue

        visited.add(vertex)

        # Adds neighbors as edges
        for neighbor in graph.get_adj_list(vertex):
            nx_graph.add_edge(vertex, neighbor)
            if neighbor not in visited:
                stack.append(neighbor)

    return nx_graph

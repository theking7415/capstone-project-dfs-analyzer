"""
Randomized Depth-First Search (RDFS) algorithm and statistics collection.

Implements the core RDFS algorithm used to validate the expected behavior.
Performs depth-first search with randomized neighbor ordering.

Performance:
- Automatically uses graph-tool backend if available (50-100x faster)
- Falls back to pure Python implementation if graph-tool not installed
- Install graph-tool: conda install -c conda-forge graph-tool
"""

import pickle
from collections import defaultdict
from typing import Any, Callable, Optional, TypeVar

import numpy as np
from scipy import stats

from dfs_analyzer.core.graphs import Graph

# Tries to import high-performance graph-tool backend
try:
    from dfs_analyzer.core.rdfs_graphtool import (
        is_available as graphtool_available,
        collect_statistics_graphtool,
        rdfs_graphtool
    )
    USE_GRAPHTOOL = graphtool_available()
except ImportError:
    USE_GRAPHTOOL = False

# Defines default RNG seed for reproducible results
DEFAULT_SEED = 1832479182
# Creates default random number generator with fixed seed
RNG = np.random.default_rng(DEFAULT_SEED)

# Defines generic vertex type for type safety
Vertex = TypeVar("Vertex")


def rdfs(
    G: Graph[Vertex],
    v: Vertex,
    *,
    dist_stats: Optional[dict[Vertex, list[int]]] = None,
    rng=RNG,
) -> Optional[list[list[Vertex]]]:
    """
    Performs Randomized Depth-First Search on a graph.

    Uses stack-based approach with randomized neighbor ordering.
    Tracks discovery number (visit order) for each vertex.

    Args:
        G: Specifies the graph to traverse.
        v: Specifies the starting vertex.
        dist_stats: Accumulates discovery numbers across multiple runs.
                   Returns visited order if None.
        rng: Provides random number generator for reproducibility.

    Returns:
        List of [parent, child] pairs if dist_stats is None.
        None if dist_stats is provided (results stored in dist_stats).
    """
    # Tracks which vertices have been visited
    visited: set[Vertex] = set()
    # Stores (vertex, parent) pairs for processing
    process_stack: list[tuple[Vertex, Vertex]] = []
    # Counts discovery order of vertices
    index: int = 0

    # Creates visited order list if not accumulating statistics
    if dist_stats is None:
        visited_order: list[list[Vertex]] = []

    # Adds starting vertex to process stack
    process_stack.append((v, v))

    # Continues until all reachable vertices are processed
    while process_stack:
        # Removes next vertex from stack
        current_node, parent = process_stack.pop()

        # Processes vertex if not yet visited
        if current_node not in visited:
            # Marks vertex as visited
            visited.add(current_node)

            # Records visit in appropriate data structure
            if dist_stats is None:
                visited_order.append([parent, current_node])
            else:
                dist_stats[current_node].append(index)

            # Increments discovery counter
            index += 1

            # Gets all neighbors of current vertex
            neighbors = G.get_adj_list(current_node)
            # Filters to unvisited neighbors only
            unvisited_neighbors = [
                neighbor for neighbor in neighbors if neighbor not in visited
            ]

            # Randomizes neighbor order for non-deterministic traversal
            rng.shuffle(unvisited_neighbors)

            # Adds unvisited neighbors to stack for processing
            for neighbor in unvisited_neighbors:
                process_stack.append((neighbor, current_node))

    # Returns visited order if not accumulating statistics
    if dist_stats is None:
        return visited_order

    return None


def collect_statistics(
    G: Graph[Vertex],
    num_samples: int,
    *,
    rng=RNG,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    num_processes: Optional[int] = None,
) -> defaultdict[Vertex, list[int]]:
    """
    Collects discovery number statistics by running RDFS multiple times.

    Runs RDFS num_samples times to build distribution of discovery numbers.

    Performance: Automatically uses graph-tool backend if available (50-100x faster).

    Args:
        G: Specifies the graph to analyze.
        num_samples: Determines how many RDFS runs to perform.
        rng: Provides random number generator for reproducibility.
        progress_callback: Reports progress with (current_sample, total_samples).
        num_processes: Number of parallel processes (graph-tool only). If None, uses all CPUs.

    Returns:
        Dictionary mapping each vertex to list of its discovery numbers.
    """
    # Uses graph-tool backend if available for massive speedup
    if USE_GRAPHTOOL:
        return collect_statistics_graphtool(G, num_samples, rng=rng, progress_callback=progress_callback, num_processes=num_processes)

    # Pure Python fallback implementation
    # Creates dictionary to store discovery numbers for each vertex
    dist_stats: defaultdict[Vertex, list[int]] = defaultdict(list)
    # Gets starting vertex from graph
    start_vertex = G.get_start_vertex()

    # Runs RDFS the specified number of times
    for i in range(num_samples):
        # Performs one RDFS run and accumulates results
        rdfs(G, start_vertex, dist_stats=dist_stats, rng=rng)

        # Calls progress callback if provided
        if progress_callback is not None:
            progress_callback(i + 1, num_samples)

    return dist_stats


def get_summary_stats(
    dist_stats: defaultdict[Vertex, list[int]]
) -> dict[Vertex, Any]:
    """
    Computes summary statistics for discovery number distributions.

    Uses scipy.stats.describe to compute mean, variance, min, max,
    skewness, and kurtosis for each vertex.

    Args:
        dist_stats: Maps vertices to lists of discovery numbers.

    Returns:
        Dictionary mapping vertices to scipy.stats.DescribeResult objects.
    """
    # Creates dictionary to store summary statistics
    summary_stats = {}
    # Computes statistics for each vertex
    for v, vals in dist_stats.items():
        summary_stats[v] = stats.describe(vals)
    return summary_stats


def save_statistics(
    G: Graph[Vertex], *, samples: int, output_dir: str = "data"
) -> str:
    """
    Runs RDFS analysis and saves results to pickle file.

    Convenience function combining analysis and storage.

    Args:
        G: Specifies the graph to analyze.
        samples: Determines number of RDFS samples to collect.
        output_dir: Specifies directory to save results.

    Returns:
        Path to the saved pickle file.
    """
    import os

    # Creates output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Collects statistics from multiple RDFS runs
    dist_stats = collect_statistics(G, samples)
    # Constructs filename from graph description and sample count
    filename = f"rdfs-{G.desc()}-{samples}-samples.pickle"
    # Combines directory and filename
    filepath = os.path.join(output_dir, filename)

    # Saves graph and statistics to pickle file
    with open(filepath, "wb") as f:
        pickle.dump((G, dist_stats), f)

    return filepath


def rdfs_till_first_backtrack(G: Graph[Vertex], v: Vertex, *, rng=RNG) -> int:
    """
    Runs RDFS until first backtrack occurs.

    Stops when encountering a vertex with no unvisited neighbors.
    Returns count of vertices visited before backtracking.

    Args:
        G: Specifies the graph to traverse.
        v: Specifies the starting vertex.
        rng: Provides random number generator.

    Returns:
        Number of vertices visited before first backtrack.
    """
    # Tracks visited vertices
    visited: set[Vertex] = set()
    # Stores vertices to process
    process_stack: list[Vertex] = []
    # Counts vertices visited
    index: int = 0

    # Adds starting vertex to stack
    process_stack.append(v)

    # Continues until stack is empty
    while process_stack:
        # Gets next vertex from stack
        current_node = process_stack.pop()

        # Processes if not yet visited
        if current_node not in visited:
            # Marks as visited
            visited.add(current_node)
            # Increments visit counter
            index += 1

            # Gets neighbors
            neighbors = G.get_adj_list(current_node)
            # Filters to unvisited neighbors
            unvisited_neighbors = [
                neighbor for neighbor in neighbors if neighbor not in visited
            ]

            # Returns count if no unvisited neighbors (backtrack point)
            if not unvisited_neighbors:
                return index

            # Randomizes neighbor order
            rng.shuffle(unvisited_neighbors)

            # Adds unvisited neighbors to stack
            for neighbor in unvisited_neighbors:
                process_stack.append(neighbor)

    return index
